"""Run 36 frozen development-only W16 grouped out-of-fold trajectories."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "vendor" / "parquet"))
import pandas as pd  # noqa: E402

sys.path.insert(0, str(ROOT / "src" / "generators"))
from run_w09_factorial import atomic_json, read_csv, sha256_file, verify_inputs, verify_protocol  # noqa: E402

sys.path.insert(0, str(ROOT / "src" / "detectors"))
import run_w12_detectors as w12  # noqa: E402

CONFIG_PATH = ROOT / "configs" / "w16_grouped_internal.json"
DETECTOR_CONFIG_PATH = ROOT / "configs" / "w12_detector_core.json"
FOLD_ROOT = ROOT / "data" / "manifests" / "w16"
GENERATOR_ROOT = ROOT / "runs" / "new_study" / "W16_grouped_generators"
RUN_ROOT = ROOT / "runs" / "new_study" / "W16_grouped_detectors"
EVIDENCE_ROOT = ROOT / "evidence" / "grouped_internal"


def object_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def load_config() -> dict[str, Any]:
    value = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if value["status"] != "FROZEN_BEFORE_W16_OUTCOMES" or not value["held_out_test_access"].startswith("PROHIBITED"):
        raise RuntimeError("W16 grouped detector configuration is not frozen and test-locked")
    return value


def detector_config() -> dict[str, Any]:
    value = json.loads(DETECTOR_CONFIG_PATH.read_text(encoding="utf-8"))
    if value["status"] != "FROZEN_BEFORE_W12_DETECTOR_OUTCOMES":
        raise RuntimeError("Shared detector configuration is not frozen")
    return value


def trajectory_roster() -> list[dict[str, Any]]:
    config = load_config()
    values = []
    for seed in config["partition_seeds"]:
        for fold in range(1, int(config["folds_per_partition"]) + 1):
            path = FOLD_ROOT / f"seed{seed}_fold{fold}_train.csv"
            rows = read_csv(path)
            base_rows = len(rows)
            for arm in config["arms"]:
                values.append({
                    "block": "grouped_internal",
                    "family": config["detector_family"],
                    "arm": arm,
                    "partition_seed": int(seed),
                    "fold": fold,
                    "detector_seed": int(seed),
                    "pipeline_seed": int(seed),
                    "pool_seed": int(seed),
                    "base_rows": base_rows,
                    "additional_rows": base_rows // 4,
                    "evaluation_split": "development_oof",
                    "run_id": f"seed{seed}_fold{fold}_{arm}",
                })
    if len(values) != 36 or len({value["run_id"] for value in values}) != 36:
        raise RuntimeError("W16 detector roster must contain 36 unique trajectories")
    return values


def preflight() -> dict[str, Any]:
    config = load_config()
    lock, protocol_hash = verify_protocol()
    inputs = verify_inputs()
    folds_path = EVIDENCE_ROOT / "W16_FOLD_RECEIPT.json"
    generators_path = EVIDENCE_ROOT / "W16_GENERATOR_COMPLETENESS.json"
    low_path = ROOT / "evidence" / "low_data" / "W14_DETECTOR_COMPLETENESS.json"
    folds = json.loads(folds_path.read_text(encoding="utf-8")) if folds_path.exists() else None
    generators = json.loads(generators_path.read_text(encoding="utf-8")) if generators_path.exists() else None
    low = json.loads(low_path.read_text(encoding="utf-8")) if low_path.exists() else None
    checks = {
        "approved_grouped_36": lock["approved_trajectory_matrix"]["grouped_internal_validation"] == config["detector_trajectories"] == len(trajectory_roster()) == 36,
        "folds_complete": folds is not None and folds["status"] == "PASS" and folds["held_out_test_rows_read"] == 0,
        "fold_generators_complete": generators is not None and generators["status"] == "COMPLETE" and generators["completed_pools"] == 9,
        "w14_complete_before_grouped": low is not None and low["status"] == "COMPLETE",
        "input_receipt_test_free": inputs["test_rows_written"] == 0 and inputs["test_image_or_label_payload_accessed"] is False,
        "test_locked": lock["execution_gates"]["test_data_access_authorized_now"] is False,
        "shared_updates_8960": detector_config()["optimization"]["optimizer_updates"] == config["detector_updates"] == 8960,
    }
    waiting = {"fold_generators_complete", "w14_complete_before_grouped"}
    status = "PASS" if all(checks.values()) else "WAITING_FOR_PREREQUISITES" if all(
        value for key, value in checks.items() if key not in waiting
    ) else "FAIL"
    return {
        "schema_version": 1,
        "status": status,
        "checks": checks,
        "protocol_sha256": protocol_hash,
        "config_sha256": sha256_file(CONFIG_PATH),
        "test_payload_accessed": False,
    }


def fold_path(spec: dict[str, Any], role: str) -> Path:
    return FOLD_ROOT / f"seed{spec['partition_seed']}_fold{spec['fold']}_{role}.csv"


def load_training_rows(spec: dict[str, Any], _config: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, str]]:
    path = fold_path(spec, "train")
    source = read_csv(path)
    if len(source) != spec["base_rows"] or any(row["split"] != "train" for row in source):
        raise RuntimeError("W16 outer-training manifest differs")
    rows = [w12.relative_real_row(row) for row in source]
    hashes = {row["image_path"]: row["image_sha256"] for row in rows}
    if spec["arm"] == "N1":
        return rows, hashes
    selected = list(range(0, len(source), 4))
    if len(selected) != spec["additional_rows"]:
        raise RuntimeError("W16 every-fourth-row ratio subset differs")
    if spec["arm"] == "NR":
        additional = [w12.relative_real_row(source[index], "additional_real", f"NR_{spec['run_id']}_") for index in selected]
        rows.extend(additional)
        hashes.update({row["image_path"]: row["image_sha256"] for row in additional})
        return rows, hashes
    if spec["arm"] not in {"NB", "NP"}:
        raise ValueError(f"Unknown W16 arm {spec['arm']}")
    generator_run = GENERATOR_ROOT / f"seed{spec['partition_seed']}_fold{spec['fold']}"
    array_path = generator_run / f"{spec['arm']}_images_uint8.npy"
    pool_summary_path = generator_run / "pool_summary.json"
    summary = json.loads(pool_summary_path.read_text(encoding="utf-8"))
    if summary["artifact_hashes"][array_path.name] != sha256_file(array_path):
        raise RuntimeError("W16 fold pool differs from its summary")
    relative_array = str(array_path.relative_to(ROOT)).replace("\\", "/")
    for index in selected:
        target = source[index]
        rows.append({
            **target,
            "sample_id": f"{spec['arm']}_{spec['run_id']}_{target['sample_id']}",
            "target_sample_id": target["sample_id"],
            "source_kind": "synthetic",
            "array_path": relative_array,
            "array_index": index,
        })
    return rows, hashes


def oof_dataset(family: str, spec: dict[str, Any] | None = None):
    if spec is None:
        raise RuntimeError("W16 OOF evaluation requires a trajectory specification")
    values = read_csv(fold_path(spec, "oof"))
    if any(row["split"] != "oof" for row in values):
        raise RuntimeError("W16 OOF manifest role differs")
    rows = [w12.relative_real_row(row) for row in values]
    hashes = {row["image_path"]: row["image_sha256"] for row in rows}
    return w12.DetectorRows(rows, family, ROOT, augment_real=False, expected_sha256=hashes), rows


def run_identity(spec: dict[str, Any], _config: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    _, protocol_hash = verify_protocol()
    shared = detector_config()
    weight_path = ROOT / shared["models"][spec["family"]]["weights_path"]
    generator_summary = GENERATOR_ROOT / f"seed{spec['partition_seed']}_fold{spec['fold']}" / "pool_summary.json"
    identity = {
        "schema_version": 1,
        "run": spec,
        "protocol_sha256": protocol_hash,
        "w16_config_sha256": sha256_file(CONFIG_PATH),
        "detector_config_sha256": sha256_file(DETECTOR_CONFIG_PATH),
        "source_sha256": sha256_file(Path(__file__)),
        "shared_runner_sha256": sha256_file(ROOT / "src" / "detectors" / "run_w12_detectors.py"),
        "common_source_sha256": sha256_file(ROOT / "src" / "detectors" / "detector_common.py"),
        "outer_train_manifest_sha256": sha256_file(fold_path(spec, "train")),
        "outer_oof_manifest_sha256": sha256_file(fold_path(spec, "oof")),
        "weights_sha256": sha256_file(weight_path),
        "training_roster_sha256": object_hash([[row["sample_id"], row["source_kind"], row.get("array_path"), row.get("array_index")] for row in rows]),
        "pool_summary_sha256": sha256_file(generator_summary) if spec["arm"] in {"NB", "NP"} else None,
        "environment_lock_sha256": sha256_file(ROOT / "ENVIRONMENT_LOCK.json"),
        "full_data_generator_or_pool_reused": False,
        "test_payload_accessed": False,
    }
    return {**identity, "identity_sha256": object_hash(identity)}


def configure_shared_runner() -> None:
    w12.RUN_ROOT = RUN_ROOT
    w12.load_config = detector_config
    w12.preflight = preflight
    w12.load_training_rows = load_training_rows
    w12.validation_dataset = oof_dataset
    w12.run_identity = run_identity


def train_one(spec: dict[str, Any]) -> dict[str, Any]:
    configure_shared_runner()
    return w12.train_one(spec)


def summarize() -> dict[str, Any]:
    completed, missing, failures = [], [], []
    for spec in trajectory_roster():
        directory = RUN_ROOT / spec["run_id"]
        path = directory / "completion.json"
        if path.exists():
            value = json.loads(path.read_text(encoding="utf-8"))
            if value["status"] != "COMPLETED" or value["test_payload_accessed"] is not False:
                raise RuntimeError("Invalid W16 detector completion receipt")
            completed.append(value)
        else:
            missing.append(spec["run_id"])
            failures.extend(str(path.relative_to(ROOT)).replace("\\", "/") for path in directory.glob("failure_*.json"))
    manifest_rows = [{
        "run_id": row["run_id"],
        "partition_seed": row["partition_seed"],
        "fold": row["fold"],
        "arm": row["arm"],
        "training_specimens": row["base_rows"] // 32,
        "oof_specimens": len(read_csv(fold_path(row, "oof"))) // 32,
        "real_seen": row["real_seen"],
        "additional_real_seen": row["additional_real_seen"],
        "synthetic_seen": row["synthetic_seen"],
        "training_gpu_hours": row["training_gpu_hours"],
        "validation_gpu_hours": row["validation_gpu_hours"],
        "primary_checkpoint_sha256": row["primary_checkpoint_sha256"],
        **{f"oof_{key}": value for key, value in row["final_validation_metrics"].items()},
    } for row in completed]
    EVIDENCE_ROOT.mkdir(parents=True, exist_ok=True)
    manifest_path = EVIDENCE_ROOT / "grouped_detector_manifest.parquet"
    if manifest_rows:
        w12.atomic_parquet(manifest_path, pd.DataFrame(manifest_rows).sort_values(["partition_seed", "fold", "arm"]))
    receipt = {
        "schema_version": 1,
        "status": "COMPLETE" if len(completed) == 36 and not missing else "INCOMPLETE",
        "planned_trajectories": 36,
        "completed_trajectories": len(completed),
        "missing_run_ids": missing,
        "failure_records": failures,
        "training_gpu_hours": sum(float(row["training_gpu_hours"]) for row in completed),
        "evaluation_gpu_hours": sum(float(row["validation_gpu_hours"]) for row in completed),
        "scope": "internal_protocol_frozen_replication_only",
        "external_validity_claim": "PROHIBITED",
        "config_sha256": sha256_file(CONFIG_PATH),
        "held_out_test_payload_accessed": False,
        "trajectory_manifest": str(manifest_path.relative_to(ROOT)).replace("\\", "/") if manifest_rows else None,
        "trajectory_manifest_sha256": sha256_file(manifest_path) if manifest_rows else None,
    }
    atomic_json(EVIDENCE_ROOT / "W16_DETECTOR_COMPLETENESS.json", receipt)
    return receipt


def accounted_gpu_hours() -> float:
    total = 0.0
    candidates = [
        (ROOT / "evidence" / "generator_factorial" / "W09_COMPLETENESS.json", ("cumulative_generator_gpu_hours",)),
        (ROOT / "evidence" / "prior_sweep" / "W10_SWEEP_COMPLETENESS.json", ("cumulative_gpu_hours",)),
        (ROOT / "evidence" / "stronger_generator" / "W11_DIFFUSION_COMPLETENESS.json", ("cumulative_gpu_hours",)),
        (ROOT / "evidence" / "detector_core" / "CORE_COMPLETENESS.json", ("training_gpu_hours", "validation_gpu_hours")),
        (ROOT / "evidence" / "resolution" / "W15_COMPLETENESS.json", ("evaluation_gpu_hours",)),
        (ROOT / "evidence" / "low_data" / "W14_GENERATOR_COMPLETENESS.json", ("cumulative_gpu_hours",)),
        (ROOT / "evidence" / "low_data" / "W14_DETECTOR_COMPLETENESS.json", ("training_gpu_hours", "validation_gpu_hours")),
        (EVIDENCE_ROOT / "W16_GENERATOR_COMPLETENESS.json", ("cumulative_gpu_hours",)),
    ]
    for path, keys in candidates:
        if path.exists():
            value = json.loads(path.read_text(encoding="utf-8"))
            total += sum(float(value.get(key, 0.0)) for key in keys)
    current = summarize()
    return total + float(current["training_gpu_hours"]) + float(current["evaluation_gpu_hours"])


def run_all(session_hours: float) -> dict[str, Any]:
    if not 0 < session_hours < 12:
        raise ValueError("W16 session-hours must be positive and below 12")
    gate = preflight()
    if gate["status"] != "PASS":
        raise RuntimeError(f"W16 detector preflight is {gate['status']}")
    started = time.perf_counter()
    ceiling = float(detector_config()["compute"]["cumulative_project_ceiling_gpu_hours"])
    for spec in trajectory_roster():
        if (RUN_ROOT / spec["run_id"] / "completion.json").exists():
            continue
        if time.perf_counter() - started >= session_hours * 3600:
            result = summarize()
            result["status"] = "PAUSED_SESSION_LIMIT"
            return result
        used = accounted_gpu_hours()
        if used + 0.75 >= ceiling:
            result = summarize()
            result["status"] = "PAUSED_COMPUTE_CEILING_GUARD"
            result["accounted_gpu_hours"] = used
            return result
        train_one(spec)
        summarize()
    return summarize()


def select_spec(partition_seed: int, fold: int, arm: str) -> dict[str, Any]:
    matches = [value for value in trajectory_roster() if value["partition_seed"] == partition_seed and value["fold"] == fold and value["arm"] == arm]
    if len(matches) != 1:
        raise ValueError("Requested W16 detector is outside the frozen roster")
    return matches[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--preflight", action="store_true")
    action.add_argument("--run-all", action="store_true")
    action.add_argument("--run-one", action="store_true")
    action.add_argument("--summarize", action="store_true")
    parser.add_argument("--partition-seed", type=int)
    parser.add_argument("--fold", type=int)
    parser.add_argument("--arm")
    parser.add_argument("--session-hours", type=float, default=11.5)
    args = parser.parse_args()
    if args.preflight:
        result = preflight()
    elif args.run_all:
        result = run_all(args.session_hours)
    elif args.run_one:
        if args.partition_seed is None or args.fold is None or args.arm is None:
            parser.error("--run-one requires --partition-seed, --fold, and --arm")
        result = train_one(select_spec(args.partition_seed, args.fold, args.arm))
    else:
        result = summarize()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] != "FAIL" else 1


if __name__ == "__main__":
    raise SystemExit(main())
