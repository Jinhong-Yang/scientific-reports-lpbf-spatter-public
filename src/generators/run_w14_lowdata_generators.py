"""Fit the 27 frozen training-only W14 NP generators and build paired NB/NP pools."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "vendor" / "parquet"))
import pandas as pd  # noqa: E402

sys.path.insert(0, str(ROOT / "src" / "generators"))
import run_w09_factorial as w09  # noqa: E402
from build_w11_pools import atomic_npy, atomic_npz, blend_pool, donor_plan, field_pool  # noqa: E402


CONFIG_PATH = ROOT / "configs" / "w14_low_data.json"
W09_CONFIG_PATH = ROOT / "configs" / "generator_factorial.json"
DETECTOR_CONFIG_PATH = ROOT / "configs" / "w12_detector_core.json"
SUBSET_ROOT = ROOT / "data" / "manifests" / "w14"
RUN_ROOT = ROOT / "runs" / "new_study" / "W14_low_data_generators"
EVIDENCE_ROOT = ROOT / "evidence" / "low_data"


def load_config() -> dict[str, Any]:
    value = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if value["status"] != "FROZEN_BEFORE_W14_OUTCOMES" or value["held_out_test_access"] != "PROHIBITED_DURING_SUBSET_SELECTION_TRAINING_AND_MODEL_SELECTION":
        raise RuntimeError("W14 generator configuration is not frozen and test-locked")
    return value


def object_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def generator_seed(realization: str, size: int, pipeline_seed: int) -> int:
    return w09.stream_seed(pipeline_seed, f"w14_generator_{realization}_n{size}")


def subset_path_for(spec: dict[str, Any]) -> Path:
    if spec.get("manifest_path"):
        return ROOT / spec["manifest_path"]
    return SUBSET_ROOT / f"{spec['realization']}_n{spec['subset_size']}.csv"


def roster() -> list[dict[str, Any]]:
    config = load_config()
    values = []
    for realization in config["subset_realizations"]:
        for size in config["subset_sizes_specimens"]:
            for pipeline_seed in config["pipeline_replicates"]:
                values.append({
                    "realization": realization["id"],
                    "subset_selection_seed": realization["selection_seed"],
                    "subset_size": size,
                    "pipeline_seed": pipeline_seed,
                    "generator_seed": generator_seed(realization["id"], size, pipeline_seed),
                    "run_id": f"{realization['id']}_n{size}_p{pipeline_seed}",
                })
    if len(values) != 27 or len({value["run_id"] for value in values}) != 27:
        raise RuntimeError("W14 generator roster must contain 27 unique fits")
    return values


def run_identity(spec: dict[str, Any], subset_path: Path, protocol_hash: str) -> dict[str, Any]:
    identity = {
        "schema_version": 1,
        "evidence_layer": "N2_NEW_STUDY",
        **spec,
        "arm": "NP_low_data_refit",
        "protocol_sha256": protocol_hash,
        "w14_config_sha256": w09.sha256_file(CONFIG_PATH),
        "w09_generator_config_sha256": w09.sha256_file(W09_CONFIG_PATH),
        "subset_manifest_sha256": w09.sha256_file(subset_path),
        "label_policy_sha256": w09.sha256_file(ROOT / "configs" / "LABEL_POLICY_LOCK.yaml"),
        "environment_sha256": w09.sha256_file(ROOT / "ENVIRONMENT_LOCK.json"),
        "source_sha256": w09.sha256_file(Path(__file__)),
        "data_roles": ["subset_train", "fixed_validation"],
        "test_access": "PROHIBITED",
        "full_data_generator_reused": False,
    }
    return {**identity, "identity_sha256": object_hash(identity)}


def train_one(spec: dict[str, Any]) -> dict[str, Any]:
    config = load_config()
    lock, protocol_hash = w09.verify_protocol()
    w09.verify_inputs()
    base_config = json.loads(W09_CONFIG_PATH.read_text(encoding="utf-8"))
    subset_path = subset_path_for(spec)
    rows = w09.read_csv(subset_path)
    expected_specimens = int(spec.get("training_specimens", spec["subset_size"]))
    if len(rows) != expected_specimens * 32 or len({row["specimen"] for row in rows}) != expected_specimens or any(row["split"] != "train" for row in rows):
        raise RuntimeError("W14 subset manifest contract changed")
    identity = run_identity(spec, subset_path, protocol_hash)
    directory = RUN_ROOT / spec["run_id"]
    summary_path = directory / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary["identity_sha256"] != identity["identity_sha256"]:
            raise RuntimeError("Completed W14 generator identity changed")
        for relative, digest in summary["artifact_hashes"].items():
            if w09.sha256_file(directory / relative) != digest:
                raise RuntimeError(f"Changed W14 generator artifact: {relative}")
        return summary
    directory.mkdir(parents=True, exist_ok=True)
    w09.atomic_json(directory / "identity.json", identity)
    weights = base_config["arms"]["F11"]
    resolved = {
        "schema_version": 1,
        "protocol_id": lock["protocol_id"],
        "arm": "NP",
        "weights": weights,
        "model": base_config["model"],
        "training": base_config["training"],
        "checkpoint": base_config["checkpoint"],
        **spec,
    }
    w09.atomic_json(directory / "resolved_config.json", resolved)
    validation_receipt = w09.verify_inputs()
    checkpoint_policy = spec.get("checkpoint_policy", "fixed_w09_validation_minimum")
    validation_rows = (
        rows if checkpoint_policy == "fixed_final_epoch_10_no_oof_selection"
        else w09.read_csv(ROOT / validation_receipt["outputs"]["validation"]["path"])
    )
    images, conditions = w09.image_tensor(rows), w09.condition_tensor(rows)
    validation_images, validation_conditions = w09.image_tensor(validation_rows), w09.condition_tensor(validation_rows)
    seed = int(spec["generator_seed"])
    w09.seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" or torch.cuda.get_device_name(device) != "NVIDIA GeForce RTX 5080":
        raise RuntimeError("W14 generator refits require the frozen local RTX 5080")
    model_config = w09.ModelConfig(
        condition_dim=base_config["condition_dimension"],
        latent_dim=base_config["model"]["latent_dimension"],
        hidden_dim=base_config["model"]["hidden_dimension"],
        fourier_frequencies=tuple(base_config["model"]["fourier_frequencies"]),
        residual_blocks=base_config["model"]["residual_blocks"],
        decoder_type=base_config["model"]["decoder_type"],
        conditional_prior=base_config["model"]["conditional_prior"],
    )
    model = w09.ConditionalVariationalField(model_config, base_config["image_size"]).to(device)
    training = base_config["training"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=training["learning_rate"], weight_decay=training["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=training["epochs"], eta_min=training["scheduler_eta_min"])
    grid = w09.make_coordinate_grid(64, device)
    moment_grid = w09.make_coordinate_grid(training["moment_grid_size"], device)
    batch_size = training["batch_size"]
    steps_per_epoch = len(images) // batch_size
    resume_path = directory / "resume.pt"
    history: list[dict[str, Any]] = []
    epoch, best_epoch, best_mse, elapsed_prior = 0, -1, float("inf"), 0.0
    if resume_path.exists():
        saved = torch.load(resume_path, map_location=device, weights_only=False)
        if saved["identity_sha256"] != identity["identity_sha256"]:
            raise RuntimeError("W14 generator resume identity changed")
        model.load_state_dict(saved["model_state"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        history, epoch = saved["history"], saved["epoch"]
        best_epoch, best_mse, elapsed_prior = saved["best_epoch"], saved["best_mse"], saved["elapsed_seconds"]
        w09.restore_rng(saved["rng"])
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()

    def save_resume() -> None:
        w09.atomic_torch(resume_path, {
            **model.checkpoint_payload(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "rng": w09.rng_state(),
            "epoch": epoch,
            "history": history,
            "best_epoch": best_epoch,
            "best_mse": best_mse,
            "elapsed_seconds": elapsed_prior + time.perf_counter() - started,
            "identity_sha256": identity["identity_sha256"],
        })

    if not resume_path.exists():
        save_resume()
    try:
        while epoch < training["epochs"]:
            model.train()
            order = torch.randperm(len(images), generator=w09.generator(seed, "image_order", epoch))
            totals = {key: 0.0 for key in ("loss", "reconstruction", "kl", "proxy", "boundary", "moment")}
            for batch_index in range(steps_per_epoch):
                indices = order[batch_index * batch_size:(batch_index + 1) * batch_size]
                image = images[indices].to(device)
                condition = conditions[indices].to(device)
                step = epoch * steps_per_epoch + batch_index
                noise, pixels, collocation, boundary = w09.draw_inputs(
                    seed, step, batch_size, model_config.latent_dim, training["pixel_samples"],
                    training["collocation_samples"], training["boundary_samples"], device,
                )
                mu, logvar = model.encode(image, condition)
                latent = mu + torch.exp(0.5 * logvar) * noise
                coordinates = grid[pixels]
                prediction = model.decode(coordinates, condition, latent)
                targets = torch.gather(image[:, 0].reshape(batch_size, -1, 1), 1, pixels[..., None])
                reconstruction = w09.weighted_reconstruction(prediction, targets)
                kl = w09.kl_divergence(mu, logvar)
                regularizers = w09.regularizer_components(
                    model.decoder, condition, latent, collocation, boundary,
                    moment_grid[None].expand(batch_size, -1, -1),
                )
                warmup = min(1.0, (epoch + 1) / training["physics_warmup_epochs"])
                loss = reconstruction + training["kl_weight"] * kl + warmup * sum(weights[name] * regularizers[name] for name in weights)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite W14 generator loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), training["gradient_clip_norm"], error_if_nonfinite=True)
                optimizer.step()
                for key, value in {"loss": loss, "reconstruction": reconstruction, "kl": kl, **regularizers}.items():
                    totals[key] += float(value.detach())
            scheduler.step()
            measured = w09.validation_mse(model, validation_images, validation_conditions, device)
            record = {"epoch": epoch + 1, **{key: value / steps_per_epoch for key, value in totals.items()}, "validation_mse": measured}
            history.append(record)
            should_save = (
                epoch + 1 == training["epochs"]
                if checkpoint_policy == "fixed_final_epoch_10_no_oof_selection"
                else measured < best_mse
            )
            if should_save:
                best_mse, best_epoch = measured, epoch + 1
                w09.atomic_torch(directory / "best.pt", {
                    **model.checkpoint_payload(),
                    "epoch": best_epoch,
                    "validation_mse": best_mse,
                    "arm": "NP",
                    "seed": seed,
                    "identity_sha256": identity["identity_sha256"],
                })
            epoch += 1
            save_resume()
            print(json.dumps({"W14_generator": spec["run_id"], **record}), flush=True)
        torch.cuda.synchronize()
        w09.atomic_csv(directory / "history.csv", history, list(history[0]))
        elapsed = elapsed_prior + time.perf_counter() - started
        artifacts = ("identity.json", "resolved_config.json", "history.csv", "best.pt", "resume.pt")
        summary = {
            "schema_version": 1,
            "status": "COMPLETED",
            **spec,
            "optimizer_updates": epoch * steps_per_epoch,
            "epochs": epoch,
            "best_epoch": best_epoch,
            "best_validation_mse": best_mse,
            "checkpoint_policy": checkpoint_policy,
            "gpu_hours": elapsed / 3600,
            "identity_sha256": identity["identity_sha256"],
            "full_data_generator_reused": False,
            "test_payload_accessed": False,
            "artifact_hashes": {name: w09.sha256_file(directory / name) for name in artifacts},
        }
        w09.atomic_json(summary_path, summary)
        return summary
    except BaseException as error:
        w09.atomic_json(directory / "failure.json", {
            "status": "FAILED_OR_INTERRUPTED",
            **spec,
            "epoch_last_atomically_completed": epoch,
            "error_type": type(error).__name__,
            "error": str(error),
            "seed_replaced": False,
            "test_payload_accessed": False,
        })
        raise
    finally:
        del model, optimizer, scheduler
        torch.cuda.empty_cache()


def build_pools(spec: dict[str, Any], summary: dict[str, Any]) -> dict[str, Any]:
    directory = RUN_ROOT / spec["run_id"]
    pool_summary_path = directory / "pool_summary.json"
    subset_path = subset_path_for(spec)
    rows = w09.read_csv(subset_path)
    checkpoint_path = directory / "best.pt"
    identity = {
        "schema_version": 1,
        "run_id": spec["run_id"],
        "generator_identity_sha256": summary["identity_sha256"],
        "generator_checkpoint_sha256": w09.sha256_file(checkpoint_path),
        "subset_manifest_sha256": w09.sha256_file(subset_path),
        "pool_seed": w09.stream_seed(spec["pipeline_seed"], f"w14_pool_{spec['realization']}_n{spec['subset_size']}"),
        "arms": ["NB", "NP"],
    }
    identity_hash = object_hash(identity)
    pool_identity_path = directory / "pool_identity.json"
    if pool_identity_path.exists():
        prior_identity = json.loads(pool_identity_path.read_text(encoding="utf-8"))
        if prior_identity != {**identity, "identity_sha256": identity_hash}:
            raise RuntimeError("Incomplete W14 pool identity changed")
    else:
        w09.atomic_json(pool_identity_path, {**identity, "identity_sha256": identity_hash})
    if pool_summary_path.exists():
        prior = json.loads(pool_summary_path.read_text(encoding="utf-8"))
        if prior["identity_sha256"] != identity_hash:
            raise RuntimeError("Completed W14 pool identity changed")
        for name, digest in prior["artifact_hashes"].items():
            if w09.sha256_file(directory / name) != digest:
                raise RuntimeError("Changed W14 pool artifact")
        return prior
    images, conditions_t = w09.image_tensor(rows), w09.condition_tensor(rows)
    plan = donor_plan(rows, conditions_t.numpy().astype(np.float64), int(identity["pool_seed"]))
    plan_path = directory / "donor_plan.npz"
    atomic_npz(plan_path, **plan)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nb = blend_pool(images, plan)
    np_pool = field_pool(checkpoint_path, images, conditions_t, conditions_t, plan, device)
    artifacts = {
        "pool_identity.json": w09.sha256_file(pool_identity_path),
        "donor_plan.npz": w09.sha256_file(plan_path),
    }
    for arm, array in (("NB", nb), ("NP", np_pool)):
        if array.shape != (len(rows), 1, 64, 64) or array.dtype != np.uint8:
            raise RuntimeError("Invalid W14 pool array")
        path = directory / f"{arm}_images_uint8.npy"
        atomic_npy(path, array)
        artifacts[path.name] = w09.sha256_file(path)
    result = {
        "schema_version": 1,
        "status": "COMPLETED",
        **spec,
        "images_per_arm": len(rows),
        "selected_detector_rows_per_arm": len(rows) // 4,
        "identity_sha256": identity_hash,
        "full_data_pool_reused": False,
        "artifact_hashes": artifacts,
        "test_payload_accessed": False,
    }
    w09.atomic_json(pool_summary_path, result)
    return result


def summarize() -> dict[str, Any]:
    _, protocol_hash = w09.verify_protocol()
    completed, pools, missing, failures = [], [], [], []
    for spec in roster():
        directory = RUN_ROOT / spec["run_id"]
        if (directory / "summary.json").exists():
            completed.append(json.loads((directory / "summary.json").read_text(encoding="utf-8")))
        else:
            missing.append(spec["run_id"])
        if (directory / "pool_summary.json").exists():
            pools.append(json.loads((directory / "pool_summary.json").read_text(encoding="utf-8")))
        if (directory / "failure.json").exists():
            failures.append(str((directory / "failure.json").relative_to(ROOT)).replace("\\", "/"))
    status = "COMPLETE" if len(completed) == len(pools) == 27 and not missing else "INCOMPLETE"
    manifest_rows = []
    if status == "COMPLETE":
        for spec in roster():
            subset = w09.read_csv(SUBSET_ROOT / f"{spec['realization']}_n{spec['subset_size']}.csv")
            for arm in ("NB", "NP"):
                path = RUN_ROOT / spec["run_id"] / f"{arm}_images_uint8.npy"
                for index, source in enumerate(subset):
                    manifest_rows.append({
                        "run_id": spec["run_id"],
                        "realization": spec["realization"],
                        "subset_size": spec["subset_size"],
                        "pipeline_seed": spec["pipeline_seed"],
                        "generator_seed": spec["generator_seed"],
                        "arm": arm,
                        "array_path": str(path.relative_to(ROOT)).replace("\\", "/"),
                        "array_sha256": w09.sha256_file(path),
                        "array_index": index,
                        "target_sample_id": source["sample_id"],
                        "specimen": source["specimen"],
                        "view": source["view"],
                        "label_source": "inherited_target",
                        "label_unit": "inherited_bbox_region",
                    })
        manifest_path = EVIDENCE_ROOT / "low_data_pool_manifest.parquet"
        pd.DataFrame(manifest_rows).to_parquet(manifest_path, index=False)
    else:
        manifest_path = None
    receipt = {
        "schema_version": 1,
        "status": status,
        "protocol_sha256": protocol_hash,
        "config_sha256": w09.sha256_file(CONFIG_PATH),
        "planned_generator_fits": 27,
        "completed_generator_fits": len(completed),
        "completed_pools": len(pools),
        "missing_run_ids": missing,
        "failure_records": failures,
        "cumulative_gpu_hours": sum(float(row["gpu_hours"]) for row in completed),
        "full_data_generator_or_pool_reused": False,
        "pool_manifest": str(manifest_path.relative_to(ROOT)).replace("\\", "/") if manifest_path else None,
        "pool_manifest_sha256": w09.sha256_file(manifest_path) if manifest_path else None,
        "human_review": "EXCLUDED_BY_USER_SCOPE",
        "test_payload_accessed": False,
    }
    w09.atomic_json(EVIDENCE_ROOT / "W14_GENERATOR_COMPLETENESS.json", receipt)
    return receipt


def accounted_gpu_hours() -> float:
    candidates = (
        (ROOT / "evidence/generator_factorial/W09_COMPLETENESS.json", ("cumulative_generator_gpu_hours",)),
        (ROOT / "evidence/prior_sweep/W10_SWEEP_COMPLETENESS.json", ("cumulative_gpu_hours",)),
        (ROOT / "evidence/stronger_generator/W11_DIFFUSION_COMPLETENESS.json", ("cumulative_gpu_hours",)),
        (ROOT / "evidence/detector_core/CORE_COMPLETENESS.json", ("training_gpu_hours", "validation_gpu_hours")),
        (ROOT / "evidence/resolution/W15_COMPLETENESS.json", ("evaluation_gpu_hours",)),
    )
    total = 0.0
    for path, keys in candidates:
        if path.exists():
            value = json.loads(path.read_text(encoding="utf-8"))
            total += sum(float(value.get(key, 0.0)) for key in keys)
    return total + float(summarize()["cumulative_gpu_hours"])


def run_all() -> dict[str, Any]:
    ceiling = float(json.loads(DETECTOR_CONFIG_PATH.read_text(encoding="utf-8"))["compute"]["cumulative_project_ceiling_gpu_hours"])
    for spec in roster():
        directory = RUN_ROOT / spec["run_id"]
        if (directory / "summary.json").exists() and (directory / "pool_summary.json").exists():
            continue
        if not (directory / "summary.json").exists():
            used = accounted_gpu_hours()
            if used + 0.75 >= ceiling:
                result = summarize()
                result["status"] = "PAUSED_COMPUTE_CEILING_GUARD"
                result["accounted_gpu_hours"] = used
                w09.atomic_json(EVIDENCE_ROOT / "W14_GENERATOR_COMPLETENESS.json", result)
                return result
        summary = train_one(spec)
        build_pools(spec, summary)
        summarize()
    return summarize()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--preflight", action="store_true")
    action.add_argument("--run-all", action="store_true")
    action.add_argument("--run-one", action="store_true")
    action.add_argument("--summarize", action="store_true")
    parser.add_argument("--realization")
    parser.add_argument("--size", type=int)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    if args.preflight:
        config = load_config()
        subset_receipt = json.loads((EVIDENCE_ROOT / "W14_SUBSET_RECEIPT.json").read_text(encoding="utf-8"))
        result = {"status": "PASS" if subset_receipt["status"] == "PASS" and len(roster()) == 27 else "FAIL",
                  "generator_fits": len(roster()), "config_sha256": w09.sha256_file(CONFIG_PATH),
                  "test_access": config["held_out_test_access"]}
    elif args.run_all:
        result = run_all()
    elif args.run_one:
        matches = [spec for spec in roster() if spec["realization"] == args.realization and spec["subset_size"] == args.size and spec["pipeline_seed"] == args.seed]
        if len(matches) != 1:
            raise ValueError("Requested W14 generator is outside the frozen roster")
        result = build_pools(matches[0], train_one(matches[0]))
    else:
        result = summarize()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] != "FAIL" else 1


if __name__ == "__main__":
    raise SystemExit(main())
