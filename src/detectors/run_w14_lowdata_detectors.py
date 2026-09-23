"""Run the frozen W14 nested low-data detector trajectories without test access."""
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


CONFIG_PATH = ROOT / "configs" / "w14_low_data.json"
DETECTOR_CONFIG_PATH = ROOT / "configs" / "w12_detector_core.json"
SUBSET_ROOT = ROOT / "data" / "manifests" / "w14"
GENERATOR_ROOT = ROOT / "runs" / "new_study" / "W14_low_data_generators"
RUN_ROOT = ROOT / "runs" / "new_study" / "W14_low_data_detectors"
EVIDENCE_ROOT = ROOT / "evidence" / "low_data"


def object_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def load_w14_config() -> dict[str, Any]:
    value = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if value["status"] != "FROZEN_BEFORE_W14_OUTCOMES":
        raise RuntimeError("W14 configuration is not frozen")
    if value["held_out_test_access"] != "PROHIBITED_DURING_SUBSET_SELECTION_TRAINING_AND_MODEL_SELECTION":
        raise RuntimeError("W14 held-out test lock differs")
    return value


def detector_config() -> dict[str, Any]:
    value = json.loads(DETECTOR_CONFIG_PATH.read_text(encoding="utf-8"))
    if value["status"] != "FROZEN_BEFORE_W12_DETECTOR_OUTCOMES":
        raise RuntimeError("Shared detector contract is not frozen")
    return value


def trajectory_roster() -> list[dict[str, Any]]:
    config = load_w14_config()
    values: list[dict[str, Any]] = []
    for realization in config["subset_realizations"]:
        for size in config["subset_sizes_specimens"]:
            base_rows = int(size) * 32
            for arm in config["arms"]:
                for seed in config["pipeline_replicates"]:
                    values.append({
                        "block": "low_data",
                        "family": "fasterrcnn",
                        "arm": arm,
                        "realization": realization["id"],
                        "subset_selection_seed": int(realization["selection_seed"]),
                        "subset_size": int(size),
                        "detector_seed": int(seed),
                        "pipeline_seed": int(seed),
                        "pool_seed": int(seed),
                        "base_rows": base_rows,
                        "additional_rows": base_rows // 4,
                        "run_id": f"{realization['id']}_n{size}_{arm}_p{seed}",
                    })
    if len(values) != 108 or len({value["run_id"] for value in values}) != 108:
        raise RuntimeError("W14 detector roster must contain 108 unique trajectories")
    return values


def preflight() -> dict[str, Any]:
    config = load_w14_config()
    lock, protocol_hash = verify_protocol()
    inputs = verify_inputs()
    subset_path = EVIDENCE_ROOT / "W14_SUBSET_RECEIPT.json"
    generator_path = EVIDENCE_ROOT / "W14_GENERATOR_COMPLETENESS.json"
    core_path = ROOT / "evidence" / "detector_core" / "CORE_COMPLETENESS.json"
    subset = json.loads(subset_path.read_text(encoding="utf-8")) if subset_path.exists() else None
    generators = json.loads(generator_path.read_text(encoding="utf-8")) if generator_path.exists() else None
    core = json.loads(core_path.read_text(encoding="utf-8")) if core_path.exists() else None
    checks = {
        "protocol_detector_total_244": lock["approved_trajectory_matrix"]["detector_total"] == 244,
        "w14_total_108": len(trajectory_roster()) == 108 and config["detector_trajectories"] == 108,
        "subset_rosters_complete": subset is not None and subset["status"] == "PASS",
        "subset_generator_pools_complete": generators is not None and generators["status"] == "COMPLETE" and generators["completed_pools"] == 27,
        "w12_core_complete_before_low_data": core is not None and core["status"] == "COMPLETE" and core["completed_trajectories"] == 100,
        "input_receipt_test_free": inputs["test_rows_written"] == 0 and inputs["test_image_or_label_payload_accessed"] is False,
        "test_locked": lock["execution_gates"]["test_data_access_authorized_now"] is False,
        "shared_updates_8960": detector_config()["optimization"]["optimizer_updates"] == config["detector_updates"] == 8960,
    }
    waiting = {"subset_generator_pools_complete", "w12_core_complete_before_low_data"}
    status = "PASS" if all(checks.values()) else "WAITING_FOR_PREREQUISITES" if all(
        value for key, value in checks.items() if key not in waiting
    ) else "FAIL"
    return {
        "schema_version": 1,
        "status": status,
        "checks": checks,
        "protocol_sha256": protocol_hash,
        "w14_config_sha256": sha256_file(CONFIG_PATH),
        "detector_config_sha256": sha256_file(DETECTOR_CONFIG_PATH),
        "test_payload_accessed": False,
    }


def load_training_rows(spec: dict[str, Any], _config: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, str]]:
    subset_path = SUBSET_ROOT / f"{spec['realization']}_n{spec['subset_size']}.csv"
    source = read_csv(subset_path)
    if len(source) != spec["base_rows"] or len({row["specimen"] for row in source}) != spec["subset_size"]:
        raise RuntimeError("W14 subset roster differs")
    if any(row["split"] != "train" for row in source):
        raise RuntimeError("W14 detector subset contains a non-training row")
    rows = [w12.relative_real_row(row) for row in source]
    hashes = {row["image_path"]: row["image_sha256"] for row in rows}
    arm = spec["arm"]
    if arm == "N1":
        return rows, hashes
    selected = list(range(0, len(source), 4))
    if len(selected) != spec["additional_rows"]:
        raise RuntimeError("W14 every-fourth-row subset differs")
    if arm == "NR":
        additional = [
            w12.relative_real_row(source[index], "additional_real", f"NR_{spec['run_id']}_")
            for index in selected
        ]
        rows.extend(additional)
        hashes.update({row["image_path"]: row["image_sha256"] for row in additional})
        return rows, hashes
    if arm not in {"NB", "NP"}:
        raise ValueError(f"Unknown W14 arm {arm}")
    generator_run = GENERATOR_ROOT / f"{spec['realization']}_n{spec['subset_size']}_p{spec['pipeline_seed']}"
    array_path = generator_run / f"{arm}_images_uint8.npy"
    summary_path = generator_run / "pool_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary["artifact_hashes"][array_path.name] != sha256_file(array_path):
        raise RuntimeError("W14 synthetic pool differs from its frozen summary")
    relative_array = str(array_path.relative_to(ROOT)).replace("\\", "/")
    for index in selected:
        target = source[index]
        rows.append({
            **target,
            "sample_id": f"{arm}_{spec['run_id']}_{target['sample_id']}",
            "target_sample_id": target["sample_id"],
            "source_kind": "synthetic",
            "array_path": relative_array,
            "array_index": index,
        })
    return rows, hashes


def run_identity(spec: dict[str, Any], _config: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    _, protocol_hash = verify_protocol()
    inputs = verify_inputs()
    shared = detector_config()
    subset_path = SUBSET_ROOT / f"{spec['realization']}_n{spec['subset_size']}.csv"
    weight_path = ROOT / shared["models"][spec["family"]]["weights_path"]
    generator_summary = (
        GENERATOR_ROOT / f"{spec['realization']}_n{spec['subset_size']}_p{spec['pipeline_seed']}" / "pool_summary.json"
    )
    identity = {
        "schema_version": 1,
        "run": spec,
        "protocol_sha256": protocol_hash,
        "w14_config_sha256": sha256_file(CONFIG_PATH),
        "detector_config_sha256": sha256_file(DETECTOR_CONFIG_PATH),
        "source_sha256": sha256_file(Path(__file__)),
        "shared_runner_sha256": sha256_file(ROOT / "src" / "detectors" / "run_w12_detectors.py"),
        "common_source_sha256": sha256_file(ROOT / "src" / "detectors" / "detector_common.py"),
        "subset_manifest_sha256": sha256_file(subset_path),
        "validation_manifest_sha256": inputs["outputs"]["validation"]["sha256"],
        "weights_sha256": sha256_file(weight_path),
        "training_roster_sha256": object_hash([
            [row["sample_id"], row["source_kind"], row.get("array_path"), row.get("array_index")]
            for row in rows
        ]),
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
    w12.run_identity = run_identity


def train_one(spec: dict[str, Any]) -> dict[str, Any]:
    configure_shared_runner()
    return w12.train_one(spec)


def summarize() -> dict[str, Any]:
    completed: list[dict[str, Any]] = []
    missing: list[str] = []
    failures: list[str] = []
    for spec in trajectory_roster():
        directory = RUN_ROOT / spec["run_id"]
        path = directory / "completion.json"
        if path.exists():
            value = json.loads(path.read_text(encoding="utf-8"))
            if value["status"] != "COMPLETED" or value["test_payload_accessed"] is not False:
                raise RuntimeError("Invalid W14 detector completion receipt")
            completed.append(value)
        else:
            missing.append(spec["run_id"])
            failures.extend(str(path.relative_to(ROOT)).replace("\\", "/") for path in directory.glob("failure_*.json"))
    manifest_rows = [{
        "run_id": row["run_id"],
        "realization": row["realization"],
        "subset_size": row["subset_size"],
        "arm": row["arm"],
        "pipeline_seed": row["pipeline_seed"],
        "base_rows": row["base_rows"],
        "additional_rows": row["additional_rows"],
        "real_seen": row["real_seen"],
        "additional_real_seen": row["additional_real_seen"],
        "synthetic_seen": row["synthetic_seen"],
        "training_gpu_hours": row["training_gpu_hours"],
        "validation_gpu_hours": row["validation_gpu_hours"],
        "primary_checkpoint_sha256": row["primary_checkpoint_sha256"],
        **{f"validation_{key}": value for key, value in row["final_validation_metrics"].items()},
    } for row in completed]
    EVIDENCE_ROOT.mkdir(parents=True, exist_ok=True)
    manifest_path = EVIDENCE_ROOT / "low_data_detector_manifest.parquet"
    if manifest_rows:
        w12.atomic_parquet(manifest_path, pd.DataFrame(manifest_rows).sort_values(["subset_size", "realization", "arm", "pipeline_seed"]))
    receipt = {
        "schema_version": 1,
        "status": "COMPLETE" if len(completed) == 108 and not missing else "INCOMPLETE",
        "planned_trajectories": 108,
        "completed_trajectories": len(completed),
        "missing_run_ids": missing,
        "failure_records": failures,
        "training_gpu_hours": sum(float(row["training_gpu_hours"]) for row in completed),
        "validation_gpu_hours": sum(float(row["validation_gpu_hours"]) for row in completed),
        "full_data_generator_or_pool_reused": False,
        "config_sha256": sha256_file(CONFIG_PATH),
        "test_payload_accessed": False,
        "trajectory_manifest": str(manifest_path.relative_to(ROOT)).replace("\\", "/") if manifest_rows else None,
        "trajectory_manifest_sha256": sha256_file(manifest_path) if manifest_rows else None,
    }
    atomic_json(EVIDENCE_ROOT / "W14_DETECTOR_COMPLETENESS.json", receipt)
    return receipt


def accounted_gpu_hours() -> float:
    paths = [
        (ROOT / "evidence" / "generator_factorial" / "W09_COMPLETENESS.json", "cumulative_generator_gpu_hours"),
        (ROOT / "evidence" / "prior_sweep" / "W10_SWEEP_COMPLETENESS.json", "cumulative_gpu_hours"),
        (ROOT / "evidence" / "stronger_generator" / "W11_DIFFUSION_COMPLETENESS.json", "cumulative_gpu_hours"),
        (ROOT / "evidence" / "detector_core" / "CORE_COMPLETENESS.json", None),
        (ROOT / "evidence" / "resolution" / "W15_COMPLETENESS.json", "evaluation_gpu_hours"),
        (EVIDENCE_ROOT / "W14_GENERATOR_COMPLETENESS.json", "cumulative_gpu_hours"),
    ]
    total = 0.0
    for path, key in paths:
        if not path.exists():
            continue
        value = json.loads(path.read_text(encoding="utf-8"))
        total += float(value.get(key, 0.0)) if key else float(value.get("training_gpu_hours", 0.0)) + float(value.get("validation_gpu_hours", 0.0))
    current = summarize()
    return total + float(current["training_gpu_hours"]) + float(current["validation_gpu_hours"])


def run_all(session_hours: float) -> dict[str, Any]:
    if not 0 < session_hours < 12:
        raise ValueError("W14 session-hours must be positive and below 12")
    gate = preflight()
    if gate["status"] != "PASS":
        raise RuntimeError(f"W14 preflight is {gate['status']}")
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


def select_spec(realization: str, size: int, arm: str, seed: int) -> dict[str, Any]:
    matches = [value for value in trajectory_roster() if value["realization"] == realization and value["subset_size"] == size and value["arm"] == arm and value["pipeline_seed"] == seed]
    if len(matches) != 1:
        raise ValueError("Requested W14 detector trajectory is outside the frozen roster")
    return matches[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--preflight", action="store_true")
    action.add_argument("--run-all", action="store_true")
    action.add_argument("--run-one", action="store_true")
    action.add_argument("--summarize", action="store_true")
    parser.add_argument("--realization")
    parser.add_argument("--size", type=int)
    parser.add_argument("--arm")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--session-hours", type=float, default=11.5)
    args = parser.parse_args()
    if args.preflight:
        result = preflight()
    elif args.run_all:
        result = run_all(args.session_hours)
    elif args.run_one:
        if None in {args.realization, args.size, args.arm, args.seed}:
            parser.error("--run-one requires --realization, --size, --arm, and --seed")
        result = train_one(select_spec(args.realization, args.size, args.arm, args.seed))
    else:
        result = summarize()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] != "FAIL" else 1


if __name__ == "__main__":
    raise SystemExit(main())
