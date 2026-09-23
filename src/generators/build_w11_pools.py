"""Build common-roster W11 synthetic pools from frozen generator fits."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "vendor" / "parquet"))
import pandas as pd
from scipy.ndimage import gaussian_filter
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "generators"))
sys.path.insert(0, str(ROOT / "src" / "analysis"))
sys.path.insert(0, str(ROOT / "src" / "pilot"))
from run_w09_factorial import (  # noqa: E402
    RUN_ROOT as W09_RUN_ROOT,
    atomic_json,
    condition_tensor,
    image_tensor,
    read_csv,
    sha256_file,
    verify_inputs,
    verify_protocol,
)
from materialize_w09_validation_arrays import build_model  # noqa: E402
from run_w06_diffusion_pilot import CompactConditionalUNet, cosine_schedule  # noqa: E402
from run_w11_diffusion import RUN_ROOT as DIFFUSION_RUN_ROOT, config as diffusion_config, stream_seed  # noqa: E402
from metal_spatter_pinn.inference import render_field  # noqa: E402


RUN_ROOT = ROOT / "runs" / "new_study" / "W11_pools"
EVIDENCE_ROOT = ROOT / "evidence" / "stronger_generator"
ARMS = ("NB", "NF", "NP", "NS", "ND")


def object_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def donor_plan(rows: list[dict[str, str]], conditions: np.ndarray, seed: int,
               target_rows: list[dict[str, str]] | None = None,
               target_conditions: np.ndarray | None = None) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    dims = np.asarray([0, 1, 2, 7, 8, 9, 10])
    donor_a, donor_b, alpha, epsilon = [], [], [], []
    targets = rows if target_rows is None else target_rows
    target_values = conditions if target_conditions is None else target_conditions
    for index, row in enumerate(targets):
        candidates = np.asarray([j for j, other in enumerate(rows)
                                 if other["material"] == row["material"] and other["view"] == row["view"]])
        if len(candidates) == 0:
            raise RuntimeError("No same-material/view donor")
        distances = np.mean((conditions[candidates][:, dims] - target_values[index, dims]) ** 2, axis=1)
        nearest = candidates[np.lexsort((candidates, distances))[:min(64, len(candidates))]]
        selected = rng.choice(nearest, size=2, replace=True)
        donor_a.append(selected[0])
        donor_b.append(selected[1])
        alpha.append(0.2 + 0.6 * rng.random())
        epsilon.append(rng.standard_normal(16))
    return {
        "donor_a": np.asarray(donor_a, dtype=np.int32),
        "donor_b": np.asarray(donor_b, dtype=np.int32),
        "alpha": np.asarray(alpha, dtype=np.float32),
        "epsilon": np.asarray(epsilon, dtype=np.float32),
    }


def atomic_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npy", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        with temporary.open("wb") as handle:
            np.save(handle, array, allow_pickle=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@torch.inference_mode()
def field_pool(checkpoint_path: Path, bank_images: torch.Tensor, bank_conditions: torch.Tensor,
               target_conditions: torch.Tensor,
               plan: dict[str, np.ndarray], device: torch.device, batch_size: int = 32) -> np.ndarray:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = build_model(checkpoint, device)
    posterior = []
    for start in range(0, len(bank_images), batch_size):
        mu, _ = model.encode(bank_images[start:start + batch_size].to(device), bank_conditions[start:start + batch_size].to(device))
        posterior.append(mu)
    bank = torch.cat(posterior)
    a = torch.from_numpy(plan["donor_a"]).to(device)
    b = torch.from_numpy(plan["donor_b"]).to(device)
    alpha = torch.from_numpy(plan["alpha"]).to(device)[:, None]
    epsilon = torch.from_numpy(plan["epsilon"]).to(device)
    latent = alpha * bank[a] + (1 - alpha) * bank[b] + 0.08 * epsilon
    chunks = []
    for start in range(0, len(target_conditions), batch_size):
        prediction = render_field(model, target_conditions[start:start + batch_size].to(device),
                                  latent[start:start + batch_size], model.image_size)
        chunks.append((prediction.clamp(0, 1) * 255).round().to(torch.uint8).cpu().numpy())
    del model, checkpoint, bank, latent
    torch.cuda.empty_cache()
    return np.concatenate(chunks)


def blend_pool(images: torch.Tensor, plan: dict[str, np.ndarray]) -> np.ndarray:
    source = images.numpy()
    alpha = plan["alpha"][:, None, None, None]
    blended = alpha * source[plan["donor_a"]] + (1 - alpha) * source[plan["donor_b"]]
    return np.rint(np.clip(blended, 0, 1) * 255).astype(np.uint8)


def ddim_timesteps(count: int = 50) -> np.ndarray:
    values = np.rint(np.linspace(999, 0, count)).astype(np.int64)
    if len(np.unique(values)) != count or values[0] != 999 or values[-1] != 0:
        raise RuntimeError("Invalid 50-step DDIM schedule")
    return values


@torch.inference_mode()
def diffusion_pool(checkpoint_path: Path, conditions: torch.Tensor, seed: int,
                   device: torch.device, batch_size: int = 16) -> np.ndarray:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = CompactConditionalUNet().to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    cumulative = torch.cumprod(1 - cosine_schedule(1000).to(device), dim=0)
    timesteps = ddim_timesteps()
    chunks = []
    for start in range(0, len(conditions), batch_size):
        current = conditions[start:start + batch_size].to(device)
        rng = torch.Generator(device=device).manual_seed(stream_seed(seed, start, "ddim_initial_noise"))
        sample = torch.randn((len(current), 1, 64, 64), generator=rng, device=device)
        for index, timestep in enumerate(timesteps):
            t = torch.full((len(current),), int(timestep), dtype=torch.long, device=device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                predicted_noise = model(sample, t, current).float()
            alpha = cumulative[int(timestep)]
            predicted_clean = (sample - (1 - alpha).sqrt() * predicted_noise) / alpha.sqrt()
            if index + 1 == len(timesteps):
                sample = predicted_clean
            else:
                previous = cumulative[int(timesteps[index + 1])]
                sample = previous.sqrt() * predicted_clean + (1 - previous).sqrt() * predicted_noise
        chunks.append(((sample.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8).cpu().numpy())
    del model, checkpoint
    torch.cuda.empty_cache()
    return np.concatenate(chunks)


def smoothing_choice() -> dict[str, Any]:
    path = EVIDENCE_ROOT.parent / "prior_sweep" / "smoothing_match.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 2:
        raise RuntimeError("Expected Gaussian and heat smoothing decisions")
    for row in rows:
        row["strength"] = float(row["strength"])
        row["absolute_relative_gap"] = float(row["absolute_relative_gap"])
    chosen = min(rows, key=lambda row: (row["absolute_relative_gap"], row["strength"], row["control_family"]))
    chosen["equivalent_gaussian_sigma_px"] = (
        chosen["strength"] if chosen["control_family"] == "gaussian_sigma_px" else math.sqrt(2 * chosen["strength"])
    )
    return chosen


def build_seed(seed: int) -> dict[str, Any]:
    lock, protocol_hash = verify_protocol()
    if seed not in lock["seed_rosters"]["primary_full_pipeline"]:
        raise ValueError("Pool seed outside frozen full-pipeline roster")
    diffusion_receipt = json.loads((EVIDENCE_ROOT / "W11_DIFFUSION_COMPLETENESS.json").read_text(encoding="utf-8"))
    if diffusion_receipt["status"] != "COMPLETE":
        raise RuntimeError("W11 diffusion fits are incomplete")
    inputs = verify_inputs()
    rows = read_csv(ROOT / inputs["outputs"]["train"]["path"])
    validation_rows = read_csv(ROOT / inputs["outputs"]["validation"]["path"])
    images, conditions_t = image_tensor(rows), condition_tensor(rows)
    validation_conditions_t = condition_tensor(validation_rows)
    conditions = conditions_t.numpy().astype(np.float64)
    plan = donor_plan(rows, conditions, seed)
    quality_plan = donor_plan(rows, conditions, seed + 1000003, validation_rows,
                              validation_conditions_t.numpy().astype(np.float64))
    directory = RUN_ROOT / f"s{seed}"
    summary_path = directory / "summary.json"
    smooth = smoothing_choice()
    identity = {
        "schema_version": 1, "seed": seed, "protocol_sha256": protocol_hash,
        "source_sha256": sha256_file(Path(__file__)), "input_receipt_sha256": sha256_file(ROOT / "data" / "manifests" / "w09" / "input_receipt.json"),
        "field_checkpoints": {arm: sha256_file(W09_RUN_ROOT / f"{factor}_s{seed}" / "best.pt")
                              for arm, factor in (("NF", "F00"), ("NP", "F11"), ("NS_BASE", "F01"))},
        "diffusion_checkpoint": sha256_file(DIFFUSION_RUN_ROOT / f"ND_s{seed}" / "final.pt"),
        "smoothing_choice": smooth, "label_policy": lock["label_policy"],
    }
    identity_hash = object_hash(identity)
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary["identity_sha256"] != identity_hash:
            raise RuntimeError("Completed pool identity mismatch")
        return summary
    directory.mkdir(parents=True, exist_ok=True)
    atomic_json(directory / "identity.json", identity)
    plan_path = directory / "donor_plan.npz"
    atomic_npz(plan_path, **plan)
    quality_plan_path = directory / "quality_donor_plan.npz"
    atomic_npz(quality_plan_path, **quality_plan)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" or torch.cuda.get_device_name(device) != "NVIDIA GeForce RTX 5080":
        raise RuntimeError("Frozen local RTX 5080 is required")
    pools = {
        "NB": blend_pool(images, plan),
        "NF": field_pool(W09_RUN_ROOT / f"F00_s{seed}" / "best.pt", images, conditions_t, conditions_t, plan, device),
        "NP": field_pool(W09_RUN_ROOT / f"F11_s{seed}" / "best.pt", images, conditions_t, conditions_t, plan, device),
        "NS": field_pool(W09_RUN_ROOT / f"F01_s{seed}" / "best.pt", images, conditions_t, conditions_t, plan, device),
        "ND": diffusion_pool(DIFFUSION_RUN_ROOT / f"ND_s{seed}" / "final.pt", conditions_t, seed, device),
    }
    quality_pools = {
        "NB": blend_pool(images, quality_plan),
        "NF": field_pool(W09_RUN_ROOT / f"F00_s{seed}" / "best.pt", images, conditions_t, validation_conditions_t, quality_plan, device),
        "NP": field_pool(W09_RUN_ROOT / f"F11_s{seed}" / "best.pt", images, conditions_t, validation_conditions_t, quality_plan, device),
        "NS": field_pool(W09_RUN_ROOT / f"F01_s{seed}" / "best.pt", images, conditions_t, validation_conditions_t, quality_plan, device),
        "ND": diffusion_pool(DIFFUSION_RUN_ROOT / f"ND_s{seed}" / "final.pt", validation_conditions_t, seed + 1000003, device),
    }
    sigma = float(smooth["equivalent_gaussian_sigma_px"])
    if sigma > 0:
        pools["NS"] = np.rint(np.clip(gaussian_filter(pools["NS"].astype(np.float32), sigma=(0, 0, sigma, sigma), mode="reflect"), 0, 255)).astype(np.uint8)
        quality_pools["NS"] = np.rint(np.clip(gaussian_filter(quality_pools["NS"].astype(np.float32), sigma=(0, 0, sigma, sigma), mode="reflect"), 0, 255)).astype(np.uint8)
    artifacts = {"donor_plan.npz": sha256_file(plan_path), "quality_donor_plan.npz": sha256_file(quality_plan_path)}
    for arm, array in pools.items():
        if array.shape != (len(rows), 1, 64, 64) or array.dtype != np.uint8:
            raise RuntimeError(f"Invalid {arm} pool array")
        path = directory / f"{arm}_images_uint8.npy"
        atomic_npy(path, array)
        artifacts[path.name] = sha256_file(path)
        quality_array = quality_pools[arm]
        if quality_array.shape != (len(validation_rows), 1, 64, 64) or quality_array.dtype != np.uint8:
            raise RuntimeError(f"Invalid {arm} quality array")
        quality_path = directory / f"quality_{arm}_images_uint8.npy"
        atomic_npy(quality_path, quality_array)
        artifacts[quality_path.name] = sha256_file(quality_path)
    summary = {
        "schema_version": 1, "status": "COMPLETED", "seed": seed, "images_per_arm": len(rows),
        "quality_images_per_arm": len(validation_rows),
        "arms": list(ARMS), "identity_sha256": identity_hash, "artifact_hashes": artifacts,
        "test_payload_accessed": False, "label_policy": "inherited_bbox_region",
        "physical_label_semantics_verified": False,
    }
    atomic_json(summary_path, summary)
    return summary


def summarize() -> dict[str, Any]:
    lock, protocol_hash = verify_protocol()
    rows, missing = [], []
    for seed in lock["seed_rosters"]["primary_full_pipeline"]:
        path = RUN_ROOT / f"s{seed}" / "summary.json"
        if path.exists():
            rows.append(json.loads(path.read_text(encoding="utf-8")))
        else:
            missing.append(seed)
    status = "COMPLETE" if len(rows) == 10 and not missing else "INCOMPLETE"
    receipt = {"schema_version": 1, "status": status, "protocol_sha256": protocol_hash,
               "planned_seeds": 10, "completed_seeds": len(rows), "missing_seeds": missing,
               "images_per_arm_seed": 3584, "quality_images_per_arm_seed": 1024,
               "quality_data_role": "validation", "arms": list(ARMS), "test_payload_accessed": False,
               "label_validity_status": "SCOPE_LIMITED_INHERITED_OPERATIONAL_TARGET_ONLY",
               "human_review": "EXCLUDED_BY_USER_SCOPE"}
    atomic_json(EVIDENCE_ROOT / "W11_POOL_COMPLETENESS.json", receipt)
    if status == "COMPLETE":
        inputs = verify_inputs()
        source_rows = read_csv(ROOT / inputs["outputs"]["train"]["path"])
        manifest_rows = []
        hashes = {}
        for seed in lock["seed_rosters"]["primary_full_pipeline"]:
            directory = RUN_ROOT / f"s{seed}"
            for arm in ARMS:
                array_path = directory / f"{arm}_images_uint8.npy"
                hashes[str(array_path.relative_to(ROOT)).replace("\\", "/")] = sha256_file(array_path)
                quality_path = directory / f"quality_{arm}_images_uint8.npy"
                hashes[str(quality_path.relative_to(ROOT)).replace("\\", "/")] = sha256_file(quality_path)
                for index, source in enumerate(source_rows):
                    manifest_rows.append({
                        "sample_id": f"{arm}_s{seed}_{source['sample_id']}", "pool_arm": arm,
                        "generator_seed": seed, "array_path": str(array_path.relative_to(ROOT)).replace("\\", "/"),
                        "array_index": index, "target_sample_id": source["sample_id"],
                        "specimen": source["specimen"], "view": source["view"], "material": source["material"],
                        "bbox_x_px": source["bbox_x_px"], "bbox_y_px": source["bbox_y_px"],
                        "bbox_width_px": source["bbox_width_px"], "bbox_height_px": source["bbox_height_px"],
                        "label_source": "inherited_target", "label_unit": "inherited_bbox_region",
                        "physical_semantics_verified": False, "source_kind": "synthetic",
                    })
        manifest_path = EVIDENCE_ROOT / "pool_manifest.parquet"
        pd.DataFrame(manifest_rows).to_parquet(manifest_path, index=False)
        atomic_json(EVIDENCE_ROOT / "source_array_hashes.json", {"schema_version": 1, "arrays": hashes,
                    "pool_manifest_sha256": sha256_file(manifest_path)})
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--preflight", action="store_true")
    action.add_argument("--run-one", action="store_true")
    action.add_argument("--run-all", action="store_true")
    action.add_argument("--summarize", action="store_true")
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    if args.preflight:
        lock, _ = verify_protocol()
        result = {"status": "PASS", "seeds": lock["seed_rosters"]["primary_full_pipeline"],
                  "arms": list(ARMS), "test_access": "PROHIBITED"}
    elif args.run_one:
        if args.seed is None:
            parser.error("--run-one requires --seed")
        result = build_seed(args.seed)
    elif args.run_all:
        for seed in diffusion_config()["seeds"]:
            build_seed(seed)
        result = summarize()
    else:
        result = summarize()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
