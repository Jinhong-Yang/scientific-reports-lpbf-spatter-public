"""Run the ten frozen compact conditional diffusion fits for W11."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "generators"))
sys.path.insert(0, str(ROOT / "src" / "pilot"))
from run_w09_factorial import (  # noqa: E402
    atomic_json,
    condition_tensor,
    image_tensor,
    object_hash,
    read_csv,
    sha256_file,
    verify_inputs,
    verify_protocol,
)
from run_w06_diffusion_pilot import (  # noqa: E402
    CompactConditionalUNet,
    cosine_schedule,
    fixed_validation_loss,
)


CONFIG_PATH = ROOT / "configs" / "w11_diffusion_full.json"
DIFFUSION_SPEC_PATH = ROOT / "configs" / "diffusion_spec.json"
RUN_ROOT = ROOT / "runs" / "new_study" / "W11_diffusion"
EVIDENCE_ROOT = ROOT / "evidence" / "stronger_generator"


def config() -> dict[str, Any]:
    value = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    spec = json.loads(DIFFUSION_SPEC_PATH.read_text(encoding="utf-8"))
    lock, _ = verify_protocol()
    if value["status"] != "DERIVED_FROM_FROZEN_PROTOCOL" or value["test_access"] != "PROHIBITED":
        raise RuntimeError("W11 diffusion configuration is not frozen-pretest derived")
    if value["optimizer_updates"] != lock["generator"]["compact_diffusion"]["optimizer_updates_per_replicate"]:
        raise RuntimeError("Diffusion update count differs from the protocol")
    if value["sampling_rule"] != "DDIM_eta_0_exactly_50_steps":
        raise RuntimeError("Diffusion sampling rule differs from the protocol")
    if value["seeds"] != lock["seed_rosters"]["primary_full_pipeline"]:
        raise RuntimeError("Diffusion seeds differ from the primary full-pipeline roster")
    if value["batch_size"] != spec["optimization_pilot"]["batch_size"]:
        raise RuntimeError("Diffusion batch differs from the approved technical path")
    return value


def stream_seed(seed: int, update: int, stream: str) -> int:
    raw = f"SR-LPBF-N2-LOCK-20260914-A001|ND|{seed}|{update}|{stream}".encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big") % (2**63 - 1)


def atomic_torch(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".pt", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def identity(seed: int, protocol_hash: str, inputs: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1, "run_id": f"W11_ND_s{seed}", "evidence_layer": "N2_NEW_STUDY",
        "seed": seed, "model_id": "ND_compact_conditional_ddpm_v0",
        "protocol_sha256": protocol_hash, "config_sha256": sha256_file(CONFIG_PATH),
        "diffusion_spec_sha256": sha256_file(DIFFUSION_SPEC_PATH),
        "pilot_source_sha256": sha256_file(ROOT / "src" / "pilot" / "run_w06_diffusion_pilot.py"),
        "source_sha256": sha256_file(Path(__file__)),
        "dataset_sha256": object_hash({key: item["sha256"] for key, item in inputs["outputs"].items()}),
        "data_roles": ["train", "validation"], "test_access": "PROHIBITED",
    }


def train_one(seed: int) -> dict[str, Any]:
    value = config()
    if seed not in value["seeds"]:
        raise ValueError("Seed is outside the frozen W11 roster")
    _, protocol_hash = verify_protocol()
    inputs = verify_inputs()
    run_identity = identity(seed, protocol_hash, inputs)
    identity_hash = object_hash(run_identity)
    directory = RUN_ROOT / f"ND_s{seed}"
    summary_path = directory / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary["identity_sha256"] != identity_hash:
            raise RuntimeError("Completed W11 identity mismatch")
        return summary
    directory.mkdir(parents=True, exist_ok=True)
    atomic_json(directory / "identity.json", run_identity)
    atomic_json(directory / "resolved_config.json", value)

    train_rows = read_csv(ROOT / inputs["outputs"]["train"]["path"])
    validation_rows = read_csv(ROOT / inputs["outputs"]["validation"]["path"])
    train_images, train_conditions = image_tensor(train_rows), condition_tensor(train_rows)
    validation_images, validation_conditions = image_tensor(validation_rows), condition_tensor(validation_rows)
    train_images = train_images * 2 - 1
    validation_images = validation_images * 2 - 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" or torch.cuda.get_device_name(device) != "NVIDIA GeForce RTX 5080":
        raise RuntimeError("Frozen local RTX 5080 is required")
    torch.manual_seed(stream_seed(seed, 0, "model"))
    torch.cuda.manual_seed_all(stream_seed(seed, 0, "model"))
    model = CompactConditionalUNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=value["learning_rate"], weight_decay=value["weight_decay"])
    cumulative = torch.cumprod(1 - cosine_schedule(1000).to(device), dim=0)
    update, elapsed_prior = 0, 0.0
    milestones: list[dict[str, float | int]] = []
    resume_path = directory / "resume.pt"
    if resume_path.exists():
        saved = torch.load(resume_path, map_location=device, weights_only=False)
        if saved["identity_sha256"] != identity_hash:
            raise RuntimeError("W11 resume identity mismatch")
        model.load_state_dict(saved["model_state"])
        optimizer.load_state_dict(saved["optimizer"])
        update, elapsed_prior = int(saved["update"]), float(saved["elapsed_seconds"])
        milestones = saved["validation_milestones"]
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    losses: list[float] = []
    failure_path = directory / "failure.json"
    try:
        while update < value["optimizer_updates"]:
            next_update = update + 1
            cpu_rng = torch.Generator().manual_seed(stream_seed(seed, next_update, "indices"))
            indices = torch.randint(len(train_images), (value["batch_size"],), generator=cpu_rng)
            clean = train_images[indices].to(device)
            condition = train_conditions[indices].to(device)
            cuda_rng = torch.Generator(device=device).manual_seed(stream_seed(seed, next_update, "diffusion"))
            timesteps = torch.randint(0, 1000, (value["batch_size"],), generator=cuda_rng, device=device)
            noise = torch.randn(clean.shape, generator=cuda_rng, device=device)
            alpha = cumulative[timesteps].view(-1, 1, 1, 1)
            noisy = alpha.sqrt() * clean + (1 - alpha).sqrt() * noise
            model.train()
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                predicted = model(noisy, timesteps, condition)
                loss = torch.nn.functional.mse_loss(predicted.float(), noise.float())
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite diffusion loss at update {next_update}")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), value["gradient_clip_norm"])
            if not torch.isfinite(norm):
                raise FloatingPointError(f"Nonfinite diffusion gradient at update {next_update}")
            optimizer.step()
            update = next_update
            losses.append(float(loss.detach()))
            if update % value["validation_every_updates"] == 0:
                metric = fixed_validation_loss(model, validation_images, validation_conditions, cumulative, device)
                milestones.append({"update": update, "fixed_t500_epsilon_mse": metric})
            if update % value["checkpoint_every_updates"] == 0:
                torch.cuda.synchronize()
                atomic_torch(resume_path, {
                    "model_state": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "update": update, "elapsed_seconds": elapsed_prior + time.perf_counter() - started,
                    "validation_milestones": milestones, "identity_sha256": identity_hash,
                })
            if update % 1000 == 0:
                print(json.dumps({"seed": seed, "update": update, "loss": losses[-1]}), flush=True)
        torch.cuda.synchronize()
        atomic_torch(directory / "final.pt", {"model_state": model.state_dict(), "update": update,
            "model_id": value["model_id"], "identity_sha256": identity_hash})
        elapsed = elapsed_prior + time.perf_counter() - started
        artifacts = ["identity.json", "resolved_config.json", "resume.pt", "final.pt"]
        summary = {
            "schema_version": 1, "status": "COMPLETED", "seed": seed,
            "optimizer_updates": update, "elapsed_seconds": elapsed, "gpu_hours": elapsed / 3600,
            "validation_milestones": milestones,
            "recent_training_loss_mean": (sum(losses[-100:]) / min(100, len(losses))) if losses else None,
            "peak_allocated_cuda_bytes": torch.cuda.max_memory_allocated(),
            "identity_sha256": identity_hash, "test_payload_accessed": False,
            "artifact_hashes": {name: sha256_file(directory / name) for name in artifacts},
        }
        atomic_json(summary_path, summary)
        failure_path.unlink(missing_ok=True)
        return summary
    except BaseException as exc:
        atomic_json(failure_path, {"status": "FAILED_OR_INTERRUPTED", "seed": seed,
            "update_last_atomically_completed": update, "error_type": type(exc).__name__,
            "error": str(exc), "test_payload_accessed": False})
        raise
    finally:
        del model, optimizer
        torch.cuda.empty_cache()


def summarize() -> dict[str, Any]:
    _, protocol_hash = verify_protocol()
    completed, failures, missing = [], [], []
    for seed in config()["seeds"]:
        directory = RUN_ROOT / f"ND_s{seed}"
        if (directory / "summary.json").exists():
            completed.append(json.loads((directory / "summary.json").read_text(encoding="utf-8")))
        elif (directory / "failure.json").exists():
            failures.append(json.loads((directory / "failure.json").read_text(encoding="utf-8")))
        else:
            missing.append(f"ND_s{seed}")
    receipt = {
        "schema_version": 1, "status": "COMPLETE" if len(completed) == 10 and not failures and not missing else "INCOMPLETE",
        "protocol_sha256": protocol_hash, "planned": 10, "completed": len(completed),
        "failed_or_interrupted": len(failures), "missing": len(missing),
        "missing_run_ids": missing, "failure_records": failures,
        "cumulative_gpu_hours": sum(item["gpu_hours"] for item in completed), "test_payload_accessed": False,
    }
    atomic_json(EVIDENCE_ROOT / "W11_DIFFUSION_COMPLETENESS.json", receipt)
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
        result = {"status": "PASS", "planned": len(config()["seeds"]), "test_access": "PROHIBITED"}
    elif args.run_one:
        if args.seed is None:
            parser.error("--run-one requires --seed")
        result = train_one(args.seed)
    elif args.run_all:
        for seed in config()["seeds"]:
            train_one(seed)
        result = summarize()
    else:
        result = summarize()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
