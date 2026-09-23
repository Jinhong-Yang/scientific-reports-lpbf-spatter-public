"""Run the frozen W12 core detector trajectories without held-out test access."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import sys
import tempfile
import time
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "vendor" / "parquet"))
import pandas as pd  # noqa: E402

sys.path.insert(0, str(ROOT / "src" / "generators"))
from run_w09_factorial import (  # noqa: E402
    atomic_json,
    atomic_torch,
    read_csv,
    seed_everything,
    sha256_file,
    stream_seed,
    verify_inputs,
    verify_protocol,
)

sys.path.insert(0, str(ROOT / "src" / "detectors"))
from detector_common import (  # noqa: E402
    DetectorRows,
    ShuffledStream,
    build_model,
    official_coco_metrics,
    per_specimen_metrics,
    predict,
    write_gzip_json,
)


CONFIG_PATH = ROOT / "configs" / "w12_detector_core.json"
RUN_ROOT = ROOT / "runs" / "new_study" / "W12_detector_core"
EVIDENCE_ROOT = ROOT / "evidence" / "detector_core"
SOURCE_ROOT = ROOT / "historical" / "metal_spatter_pinn"
POOL_EVIDENCE_ROOT = ROOT / "evidence" / "stronger_generator"
POOL_RUN_ROOT = ROOT / "runs" / "new_study" / "W11_pools"


def load_config() -> dict[str, Any]:
    value = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if value["status"] != "FROZEN_BEFORE_W12_DETECTOR_OUTCOMES":
        raise RuntimeError("W12 detector configuration is not frozen")
    if value["data_roles"]["held_out_test_access"] != "PROHIBITED_DURING_TRAINING_AND_SELECTION":
        raise RuntimeError("W12 test lock differs")
    return value


def object_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".parquet", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def relative_real_row(row: dict[str, str], source_kind: str = "real", prefix: str = "") -> dict[str, Any]:
    value: dict[str, Any] = dict(row)
    value["sample_id"] = prefix + row["sample_id"]
    value["target_sample_id"] = row["sample_id"]
    value["source_kind"] = source_kind
    value["image_path"] = str(Path("historical") / "metal_spatter_pinn" / Path(row["image_path"].replace("\\", "/"))).replace("\\", "/")
    return value


def trajectory_roster(config: dict[str, Any]) -> list[dict[str, Any]]:
    output = []
    primary = config["trajectory_roster"]["primary"]
    for arm in primary["arms"]:
        for seed in primary["pipeline_seeds"]:
            output.append({
                "block": "primary",
                "family": primary["family"],
                "arm": arm,
                "detector_seed": int(seed),
                "pool_seed": int(seed),
                "run_id": f"primary_{primary['family']}_{arm}_s{seed}",
            })
    secondary = config["trajectory_roster"]["secondary"]
    for arm in secondary["arms"]:
        for seed in secondary["detector_seeds"]:
            output.append({
                "block": "secondary",
                "family": secondary["family"],
                "arm": arm,
                "detector_seed": int(seed),
                "pool_seed": int(secondary["pool_seed_mapping"][str(seed)]),
                "run_id": f"secondary_{secondary['family']}_{arm}_s{seed}",
            })
    if len(output) != config["trajectory_roster"]["total"] or len({row["run_id"] for row in output}) != len(output):
        raise RuntimeError("W12 trajectory roster does not contain 100 unique runs")
    return output


def preflight() -> dict[str, Any]:
    config = load_config()
    lock, protocol_hash = verify_protocol()
    inputs = verify_inputs()
    quality_path = POOL_EVIDENCE_ROOT / "W11_QUALITY_COMPLETENESS.json"
    quality = json.loads(quality_path.read_text(encoding="utf-8")) if quality_path.exists() else None
    weights = {}
    for family, model in config["models"].items():
        path = ROOT / model["weights_path"]
        digest = sha256_file(path) if path.is_file() else ""
        weights[family] = {
            "path": str(path.relative_to(ROOT)).replace("\\", "/"),
            "exists": path.is_file(),
            "sha256": digest,
            "prefix_match": digest.startswith(model["weights_sha256_prefix"]),
        }
    checks = {
        "protocol_detector_total_244": lock["approved_trajectory_matrix"]["detector_total"] == 244,
        "w12_core_total_100": len(trajectory_roster(config)) == 100,
        "input_receipt_test_free": inputs["test_rows_written"] == 0 and inputs["test_image_or_label_payload_accessed"] is False,
        "w11_quality_complete": quality is not None and quality["status"] == "PASS_WITH_SCOPE_LIMITATION" and quality["test_payload_accessed"] is False,
        "label_scope_bound": quality is not None and quality["label_validity_status"] == "SCOPE_LIMITED_INHERITED_OPERATIONAL_TARGET_ONLY",
        "weight_hashes": all(value["prefix_match"] for value in weights.values()),
        "ratio_is_preapproved_fallback": config["ratio_policy"]["selection_status"] == "PREAPPROVED_FALLBACK" and config["ratio_policy"]["selected_synthetic_to_base_real_roster_ratio"] == 0.25,
        "test_locked": lock["execution_gates"]["test_data_access_authorized_now"] is False,
    }
    return {
        "schema_version": 1,
        "status": "PASS" if all(checks.values()) else "WAITING_FOR_W11" if not checks["w11_quality_complete"] else "FAIL",
        "protocol_sha256": protocol_hash,
        "config_sha256": sha256_file(CONFIG_PATH),
        "checks": checks,
        "weights": weights,
        "test_payload_accessed": False,
    }


def load_training_rows(spec: dict[str, Any], config: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, str]]:
    inputs = verify_inputs()
    train_path = ROOT / inputs["outputs"]["train"]["path"]
    source = read_csv(train_path)
    if len(source) != 3584:
        raise RuntimeError("W12 base-real roster differs")
    rows = [relative_real_row(row) for row in source]
    hashes = {row["image_path"]: row["image_sha256"] for row in rows}
    arm = spec["arm"]
    if arm in {"N0", "N1"}:
        return rows, hashes
    selected = list(range(0, len(source), 4))
    if len(selected) != config["ratio_policy"]["selected_additional_rows"]:
        raise RuntimeError("W12 every-fourth-row ratio subset differs")
    if arm == "NR":
        additional = [relative_real_row(source[index], "additional_real", f"NR_extra_s{spec['pool_seed']}_") for index in selected]
        rows.extend(additional)
        hashes.update({row["image_path"]: row["image_sha256"] for row in additional})
        return rows, hashes
    if arm not in {"NB", "NF", "NP", "NS", "ND"}:
        raise ValueError(f"Unknown W12 arm {arm}")
    array_path = POOL_RUN_ROOT / f"s{spec['pool_seed']}" / f"{arm}_images_uint8.npy"
    summary_path = POOL_RUN_ROOT / f"s{spec['pool_seed']}" / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary["artifact_hashes"][array_path.name] != sha256_file(array_path):
        raise RuntimeError("W12 pool array differs from W11 summary")
    relative_array = str(array_path.relative_to(ROOT)).replace("\\", "/")
    for index in selected:
        target = source[index]
        rows.append({
            **target,
            "sample_id": f"{arm}_s{spec['pool_seed']}_{target['sample_id']}",
            "target_sample_id": target["sample_id"],
            "source_kind": "synthetic",
            "array_path": relative_array,
            "array_index": index,
        })
    return rows, hashes


def validation_dataset(family: str, _spec: dict[str, Any] | None = None) -> tuple[DetectorRows, list[dict[str, Any]]]:
    inputs = verify_inputs()
    values = read_csv(ROOT / inputs["outputs"]["validation"]["path"])
    rows = [relative_real_row(row) for row in values]
    hashes = {row["image_path"]: row["image_sha256"] for row in rows}
    return DetectorRows(rows, family, ROOT, augment_real=False, expected_sha256=hashes), rows


def rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def artifact_hashes(directory: Path) -> dict[str, str]:
    return {
        str(path.relative_to(ROOT)).replace("\\", "/"): sha256_file(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path.name not in {"completion.json"}
    }


def verify_artifact_hashes(values: dict[str, str]) -> None:
    if not values:
        raise RuntimeError("Empty W12 artifact ledger")
    for relative, digest in values.items():
        path = ROOT / relative
        if not path.is_file() or sha256_file(path) != digest:
            raise RuntimeError(f"Changed W12 artifact: {relative}")


def evaluate_milestone(
    model: torch.nn.Module,
    dataset: DetectorRows,
    device: torch.device,
    directory: Path,
    spec: dict[str, Any],
    step: int,
    checkpoint_sha256: str,
) -> dict[str, Any]:
    destination = directory / "validation" / f"milestone_{step}"
    receipt_path = destination / "receipt.json"
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt["checkpoint_sha256"] != checkpoint_sha256:
            raise RuntimeError("Validation receipt checkpoint changed")
        verify_artifact_hashes(receipt["artifact_hashes"])
        return receipt
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f"milestone_{step}_attempt_", dir=destination.parent))
    started = time.perf_counter()
    try:
        records = predict(model, dataset, device, batch_size=4)
        metrics = official_coco_metrics(records)
        predictions_path = temporary / "predictions.json.gz"
        metrics_path = temporary / "metrics.json"
        write_gzip_json(predictions_path, records)
        atomic_json(metrics_path, {
            "schema_version": 1,
            "run_id": spec["run_id"],
            "block": spec["block"],
            "family": spec["family"],
            "arm": spec["arm"],
            "detector_seed": spec["detector_seed"],
            "pool_seed": spec["pool_seed"],
            "optimizer_step": step,
            "checkpoint_sha256": checkpoint_sha256,
            "split": spec.get("evaluation_split", "validation"),
            **metrics,
            "test_payload_accessed": False,
        })
        if step == 8960:
            per_specimen = pd.DataFrame(per_specimen_metrics(records))
            per_specimen.insert(0, "run_id", spec["run_id"])
            per_specimen.insert(1, "family", spec["family"])
            per_specimen.insert(2, "arm", spec["arm"])
            per_specimen.insert(3, "detector_seed", spec["detector_seed"])
            atomic_parquet(temporary / "per_specimen_metrics.parquet", per_specimen)
        files = {
            str(path.name): sha256_file(path)
            for path in sorted(temporary.iterdir())
            if path.is_file()
        }
        receipt = {
            "schema_version": 1,
            "status": "COMPLETED",
            "checkpoint_sha256": checkpoint_sha256,
            "optimizer_step": step,
            "metrics": metrics,
            "elapsed_seconds": time.perf_counter() - started,
            "test_payload_accessed": False,
            "files": files,
        }
        atomic_json(temporary / "receipt.json", receipt)
        if destination.exists():
            raise RuntimeError("Validation destination appeared during atomic evaluation")
        os.replace(temporary, destination)
        receipt["artifact_hashes"] = {
            str(path.relative_to(ROOT)).replace("\\", "/"): sha256_file(path)
            for path in destination.iterdir()
            if path.is_file() and path.name != "receipt.json"
        }
        atomic_json(receipt_path, receipt)
        return receipt
    except BaseException:
        # The uniquely named attempt directory is deliberately preserved.
        raise


def run_identity(spec: dict[str, Any], config: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    _, protocol_hash = verify_protocol()
    inputs = verify_inputs()
    weight_path = ROOT / config["models"][spec["family"]]["weights_path"]
    identity = {
        "schema_version": 1,
        "run": spec,
        "protocol_sha256": protocol_hash,
        "config_sha256": sha256_file(CONFIG_PATH),
        "source_sha256": sha256_file(Path(__file__)),
        "common_source_sha256": sha256_file(ROOT / "src" / "detectors" / "detector_common.py"),
        "train_manifest_sha256": inputs["outputs"]["train"]["sha256"],
        "validation_manifest_sha256": inputs["outputs"]["validation"]["sha256"],
        "weights_sha256": sha256_file(weight_path),
        "training_roster_sha256": object_hash([
            [row["sample_id"], row["source_kind"], row.get("array_path"), row.get("array_index")]
            for row in rows
        ]),
        "pool_summary_sha256": (
            sha256_file(POOL_RUN_ROOT / f"s{spec['pool_seed']}" / "summary.json")
            if spec["arm"] in {"NB", "NF", "NP", "NS", "ND"} else None
        ),
        "environment_lock_sha256": sha256_file(ROOT / "ENVIRONMENT_LOCK.json"),
        "test_payload_accessed": False,
    }
    return {**identity, "identity_sha256": object_hash(identity)}


def train_one(spec: dict[str, Any]) -> dict[str, Any]:
    config = load_config()
    gate = preflight()
    if gate["status"] != "PASS":
        raise RuntimeError(f"W12 preflight is {gate['status']}")
    rows, expected_hashes = load_training_rows(spec, config)
    base_rows = int(spec.get("base_rows", 3584))
    additional_rows = int(spec.get("additional_rows", 896))
    expected_count = base_rows if spec["arm"] in {"N0", "N1"} else base_rows + additional_rows
    if len(rows) != expected_count:
        raise RuntimeError("W12 training roster count differs")
    identity = run_identity(spec, config, rows)
    directory = RUN_ROOT / spec["run_id"]
    completion_path = directory / "completion.json"
    if completion_path.exists():
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        if completion["identity_sha256"] != identity["identity_sha256"]:
            raise RuntimeError("Completed W12 run identity changed")
        verify_artifact_hashes(completion["artifact_hashes"])
        return completion
    identity_path = directory / "identity.json"
    if identity_path.exists():
        prior = json.loads(identity_path.read_text(encoding="utf-8"))
        if prior != identity:
            raise RuntimeError("Incomplete W12 run is bound to different source/config/input identity")
    else:
        directory.mkdir(parents=True, exist_ok=True)
        atomic_json(identity_path, identity)
    if shutil.disk_usage(ROOT).free < 20_000_000_000:
        raise RuntimeError("Less than 20 GB free before W12 trajectory")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" or torch.cuda.get_device_name(device) != "NVIDIA GeForce RTX 5080":
        raise RuntimeError("W12 requires the frozen local RTX 5080")
    seed_everything(spec["detector_seed"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.reset_peak_memory_stats()
    augment_real = spec["arm"] != "N0"
    training = DetectorRows(rows, spec["family"], ROOT, augment_real=augment_real, expected_sha256=expected_hashes)
    validation, _ = validation_dataset(spec["family"], spec)
    sampler = ShuffledStream(len(training), stream_seed(spec["detector_seed"], "w12_sampler"))
    weight_path = ROOT / config["models"][spec["family"]]["weights_path"]
    model = build_model(spec["family"], config, weight_path).to(device)
    optimization = config["optimization"]
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=optimization["learning_rate"],
        momentum=optimization["momentum"],
        weight_decay=optimization["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=optimization["learning_rate_step_updates"],
        gamma=optimization["learning_rate_gamma"],
    )
    progress: dict[str, Any] = {
        "step": 0,
        "real_seen": 0,
        "additional_real_seen": 0,
        "synthetic_seen": 0,
        "loss_sum": 0.0,
        "loss_count": 0,
        "training_elapsed_seconds": 0.0,
        "validation_elapsed_seconds": 0.0,
        "milestones": {},
        "evaluations": {},
    }
    resume_path = directory / "resume.pt"
    if resume_path.exists():
        state = torch.load(resume_path, map_location="cpu", weights_only=False)
        if state["identity_sha256"] != identity["identity_sha256"]:
            raise RuntimeError("W12 resume identity changed")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        sampler.load_state_dict(state["sampler"])
        progress = state["progress"]
        restore_rng(state["rng"])

    def state_payload() -> dict[str, Any]:
        return {
            "identity_sha256": identity["identity_sha256"],
            "run": spec,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "sampler": sampler.state_dict(),
            "rng": rng_state(),
            "progress": progress,
        }

    def save_resume() -> None:
        atomic_torch(resume_path, state_payload())

    if not resume_path.exists():
        save_resume()
    milestones = set(int(value) for value in optimization["validation_milestones"])
    try:
        # Finish any validation transaction whose training checkpoint was already committed.
        for step_text, metadata in sorted(progress["milestones"].items(), key=lambda item: int(item[0])):
            if step_text not in progress["evaluations"]:
                checkpoint = torch.load(ROOT / metadata["path"], map_location="cpu", weights_only=False)
                model.load_state_dict(checkpoint["model"])
                receipt = evaluate_milestone(model, validation, device, directory, spec, int(step_text), metadata["sha256"])
                progress["evaluations"][step_text] = receipt
                progress["validation_elapsed_seconds"] += receipt["elapsed_seconds"]
                save_resume()
        for step in range(int(progress["step"]) + 1, optimization["optimizer_updates"] + 1):
            started = time.perf_counter()
            indices = sampler.take(optimization["micro_batch"])
            batch = [training[index] for index in indices]
            counts = {kind: sum(meta["source_kind"] == kind for _, _, meta in batch)
                      for kind in ("real", "additional_real", "synthetic")}
            model.train()
            images = [image.to(device) for image, _, _ in batch]
            targets = [{name: value.to(device) for name, value in target.items()} for _, target, _ in batch]
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                losses = model(images, targets)
                loss = sum(losses.values())
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite W12 detector loss")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), optimization["gradient_clip_norm"])
            if not torch.isfinite(norm):
                raise FloatingPointError("Nonfinite W12 detector gradient norm")
            optimizer.step()
            scheduler.step()
            torch.cuda.synchronize()
            progress["step"] = step
            progress["real_seen"] += counts["real"]
            progress["additional_real_seen"] += counts["additional_real"]
            progress["synthetic_seen"] += counts["synthetic"]
            progress["loss_sum"] += float(loss.detach())
            progress["loss_count"] += 1
            progress["training_elapsed_seconds"] += time.perf_counter() - started
            if step in milestones:
                checkpoint_path = directory / "checkpoints" / f"milestone_{step}.pt"
                atomic_torch(checkpoint_path, state_payload())
                checkpoint_sha256 = sha256_file(checkpoint_path)
                progress["milestones"][str(step)] = {
                    "path": str(checkpoint_path.relative_to(ROOT)).replace("\\", "/"),
                    "sha256": checkpoint_sha256,
                    "step": step,
                    "real_seen": progress["real_seen"],
                    "additional_real_seen": progress["additional_real_seen"],
                    "synthetic_seen": progress["synthetic_seen"],
                    "mean_loss_since_prior_milestone": progress["loss_sum"] / progress["loss_count"],
                    "learning_rate": optimizer.param_groups[0]["lr"],
                }
                save_resume()
                receipt = evaluate_milestone(model, validation, device, directory, spec, step, checkpoint_sha256)
                progress["evaluations"][str(step)] = receipt
                progress["validation_elapsed_seconds"] += receipt["elapsed_seconds"]
                progress["loss_sum"] = 0.0
                progress["loss_count"] = 0
                save_resume()
                print(json.dumps({
                    "W12": spec["run_id"],
                    "step": step,
                    "COCO_AP": receipt["metrics"]["COCO_AP"],
                    "real_seen": progress["real_seen"],
                    "additional_real_seen": progress["additional_real_seen"],
                    "synthetic_seen": progress["synthetic_seen"],
                }), flush=True)
            elif step % optimization["resume_interval_updates"] == 0:
                save_resume()
        if set(progress["milestones"]) != {str(value) for value in milestones} or set(progress["evaluations"]) != {str(value) for value in milestones}:
            raise RuntimeError("W12 trajectory ended with incomplete milestones")
        total_seen = progress["real_seen"] + progress["additional_real_seen"] + progress["synthetic_seen"]
        expected_total_seen = optimization["optimizer_updates"] * optimization["micro_batch"]
        if total_seen != expected_total_seen:
            raise RuntimeError("W12 exposure accounting mismatch")
        audit_sampler = ShuffledStream(len(rows), stream_seed(spec["detector_seed"], "w12_sampler"))
        expected_indices = audit_sampler.take(expected_total_seen)
        expected_exposure = Counter(rows[index]["source_kind"] for index in expected_indices)
        expected_exposure = {
            kind: int(expected_exposure.get(kind, 0))
            for kind in ("real", "additional_real", "synthetic")
        }
        observed_exposure = {
            "real": progress["real_seen"],
            "additional_real": progress["additional_real_seen"],
            "synthetic": progress["synthetic_seen"],
        }
        if observed_exposure != expected_exposure:
            raise RuntimeError(
                f"Detector exposure count mismatch: observed={observed_exposure}, expected={expected_exposure}"
            )
        completion = {
            "schema_version": 1,
            "status": "COMPLETED",
            **spec,
            "identity_sha256": identity["identity_sha256"],
            "optimizer_updates": progress["step"],
            "real_seen": progress["real_seen"],
            "additional_real_seen": progress["additional_real_seen"],
            "synthetic_seen": progress["synthetic_seen"],
            "training_gpu_hours": progress["training_elapsed_seconds"] / 3600.0,
            "validation_gpu_hours": progress["validation_elapsed_seconds"] / 3600.0,
            "peak_cuda_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_cuda_memory_reserved_bytes": torch.cuda.max_memory_reserved(),
            "primary_checkpoint_step": 8960,
            "primary_checkpoint_sha256": progress["milestones"]["8960"]["sha256"],
            "final_validation_metrics": progress["evaluations"]["8960"]["metrics"],
            "test_payload_accessed": False,
        }
        completion["artifact_hashes"] = artifact_hashes(directory)
        atomic_json(completion_path, completion)
        return completion
    except BaseException as error:
        failure = {
            "schema_version": 1,
            "status": "FAILED_OR_INTERRUPTED",
            **spec,
            "identity_sha256": identity["identity_sha256"],
            "step_last_atomically_resumable": progress["step"],
            "error_type": type(error).__name__,
            "error": str(error),
            "seed_replaced": False,
            "test_payload_accessed": False,
        }
        atomic_json(directory / f"failure_{time.time_ns()}.json", failure)
        raise
    finally:
        del model, optimizer, scheduler
        torch.cuda.empty_cache()


def summarize() -> dict[str, Any]:
    config = load_config()
    rows, failures, missing = [], [], []
    for spec in trajectory_roster(config):
        directory = RUN_ROOT / spec["run_id"]
        completion = directory / "completion.json"
        if completion.exists():
            value = json.loads(completion.read_text(encoding="utf-8"))
            if value["status"] != "COMPLETED" or value["test_payload_accessed"] is not False:
                raise RuntimeError("Invalid W12 completion receipt")
            rows.append(value)
        else:
            missing.append(spec["run_id"])
            failures.extend(str(path.relative_to(ROOT)).replace("\\", "/") for path in directory.glob("failure_*.json"))
    status = "COMPLETE" if len(rows) == 100 and not missing else "INCOMPLETE"
    manifest = []
    for row in rows:
        manifest.append({
            "run_id": row["run_id"],
            "block": row["block"],
            "family": row["family"],
            "arm": row["arm"],
            "detector_seed": row["detector_seed"],
            "pool_seed": row["pool_seed"],
            "optimizer_updates": row["optimizer_updates"],
            "real_seen": row["real_seen"],
            "additional_real_seen": row["additional_real_seen"],
            "synthetic_seen": row["synthetic_seen"],
            "training_gpu_hours": row["training_gpu_hours"],
            "validation_gpu_hours": row["validation_gpu_hours"],
            "primary_checkpoint_sha256": row["primary_checkpoint_sha256"],
            **{f"validation_{key}": value for key, value in row["final_validation_metrics"].items()},
        })
    EVIDENCE_ROOT.mkdir(parents=True, exist_ok=True)
    manifest_path = EVIDENCE_ROOT / "trajectory_manifest.parquet"
    if manifest:
        atomic_parquet(manifest_path, pd.DataFrame(manifest).sort_values(["block", "family", "arm", "detector_seed"]))
    receipt = {
        "schema_version": 1,
        "status": status,
        "planned_trajectories": 100,
        "completed_trajectories": len(rows),
        "primary_completed": sum(row["block"] == "primary" for row in rows),
        "secondary_completed": sum(row["block"] == "secondary" for row in rows),
        "missing_run_ids": missing,
        "failure_records": failures,
        "training_gpu_hours": sum(row["training_gpu_hours"] for row in rows),
        "validation_gpu_hours": sum(row["validation_gpu_hours"] for row in rows),
        "config_sha256": sha256_file(CONFIG_PATH),
        "test_payload_accessed": False,
        "trajectory_manifest": str(manifest_path.relative_to(ROOT)).replace("\\", "/") if manifest else None,
        "trajectory_manifest_sha256": sha256_file(manifest_path) if manifest else None,
    }
    atomic_json(EVIDENCE_ROOT / "CORE_COMPLETENESS.json", receipt)
    return receipt


def completed_project_gpu_hours() -> float:
    total = 0.0
    for path, key in (
        (ROOT / "evidence" / "generator_factorial" / "W09_COMPLETENESS.json", "cumulative_generator_gpu_hours"),
        (ROOT / "evidence" / "prior_sweep" / "W10_SWEEP_COMPLETENESS.json", "cumulative_gpu_hours"),
        (ROOT / "evidence" / "stronger_generator" / "W11_DIFFUSION_COMPLETENESS.json", "cumulative_gpu_hours"),
    ):
        if path.exists():
            value = json.loads(path.read_text(encoding="utf-8"))
            total += float(value.get(key, 0.0))
    receipt = summarize()
    total += float(receipt["training_gpu_hours"]) + float(receipt["validation_gpu_hours"])
    return total


def run_all(session_hours: float) -> dict[str, Any]:
    if not 0 < session_hours < 12:
        raise ValueError("W12 session-hours must be positive and below 12")
    config = load_config()
    started = time.perf_counter()
    for spec in trajectory_roster(config):
        if (RUN_ROOT / spec["run_id"] / "completion.json").exists():
            continue
        if time.perf_counter() - started >= session_hours * 3600:
            result = summarize()
            result["status"] = "PAUSED_SESSION_LIMIT"
            result["session_hours_limit"] = session_hours
            return result
        used = completed_project_gpu_hours()
        if used + 0.75 >= float(config["compute"]["cumulative_project_ceiling_gpu_hours"]):
            result = summarize()
            result["status"] = "PAUSED_COMPUTE_CEILING_GUARD"
            result["accounted_gpu_hours"] = used
            return result
        train_one(spec)
        summarize()
    return summarize()


def select_spec(block: str, arm: str, seed: int) -> dict[str, Any]:
    matches = [row for row in trajectory_roster(load_config())
               if row["block"] == block and row["arm"] == arm and row["detector_seed"] == seed]
    if len(matches) != 1:
        raise ValueError("Requested W12 trajectory is outside the frozen roster")
    return matches[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--preflight", action="store_true")
    action.add_argument("--run-one", action="store_true")
    action.add_argument("--run-all", action="store_true")
    action.add_argument("--summarize", action="store_true")
    parser.add_argument("--block", choices=("primary", "secondary"))
    parser.add_argument("--arm")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--session-hours", type=float, default=11.5)
    args = parser.parse_args()
    if args.preflight:
        result = preflight()
    elif args.run_one:
        if args.block is None or args.arm is None or args.seed is None:
            parser.error("--run-one requires --block, --arm, and --seed")
        result = train_one(select_spec(args.block, args.arm, args.seed))
    elif args.run_all:
        result = run_all(args.session_hours)
    else:
        result = summarize()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] not in {"FAIL"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
