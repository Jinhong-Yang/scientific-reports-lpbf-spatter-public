"""Fit nine training-fold-only W16 NP generators and build NB/NP pools."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "generators"))
import run_w09_factorial as w09  # noqa: E402
import run_w14_lowdata_generators as w14  # noqa: E402

CONFIG_PATH = ROOT / "configs" / "w16_grouped_internal.json"
DETECTOR_CONFIG_PATH = ROOT / "configs" / "w12_detector_core.json"
FOLD_ROOT = ROOT / "data" / "manifests" / "w16"
RUN_ROOT = ROOT / "runs" / "new_study" / "W16_grouped_generators"
EVIDENCE_ROOT = ROOT / "evidence" / "grouped_internal"


def load_config() -> dict[str, Any]:
    value = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if value["status"] != "FROZEN_BEFORE_W16_OUTCOMES" or not value["held_out_test_access"].startswith("PROHIBITED"):
        raise RuntimeError("W16 grouped generator configuration is not frozen and test-locked")
    return value


def object_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def roster() -> list[dict[str, Any]]:
    config = load_config()
    values = []
    for seed in config["partition_seeds"]:
        for fold in range(1, int(config["folds_per_partition"]) + 1):
            manifest = FOLD_ROOT / f"seed{seed}_fold{fold}_train.csv"
            rows = w09.read_csv(manifest)
            specimens = len({row["specimen"] for row in rows})
            values.append({
                "partition_seed": int(seed),
                "fold": fold,
                "realization": f"seed{seed}_fold{fold}",
                "subset_size": specimens,
                "training_specimens": specimens,
                "pipeline_seed": int(seed),
                "generator_seed": w09.stream_seed(int(seed), f"w16_generator_fold{fold}"),
                "manifest_path": str(manifest.relative_to(ROOT)).replace("\\", "/"),
                "checkpoint_policy": "fixed_final_epoch_10_no_oof_selection",
                "run_id": f"seed{seed}_fold{fold}",
            })
    if len(values) != 9 or len({value["run_id"] for value in values}) != 9:
        raise RuntimeError("W16 generator roster must contain nine unique refits")
    return values


def subset_path_for(spec: dict[str, Any]) -> Path:
    return ROOT / spec["manifest_path"]


def run_identity(spec: dict[str, Any], subset_path: Path, protocol_hash: str) -> dict[str, Any]:
    identity = {
        "schema_version": 1,
        "evidence_layer": "N2_NEW_STUDY",
        **spec,
        "arm": "NP_grouped_outer_training_refit",
        "protocol_sha256": protocol_hash,
        "w16_config_sha256": w09.sha256_file(CONFIG_PATH),
        "w09_generator_config_sha256": w09.sha256_file(ROOT / "configs" / "generator_factorial.json"),
        "fold_train_manifest_sha256": w09.sha256_file(subset_path),
        "shared_generator_source_sha256": w09.sha256_file(ROOT / "src" / "generators" / "run_w14_lowdata_generators.py"),
        "source_sha256": w09.sha256_file(Path(__file__)),
        "label_policy_sha256": w09.sha256_file(ROOT / "configs" / "LABEL_POLICY_LOCK.yaml"),
        "environment_sha256": w09.sha256_file(ROOT / "ENVIRONMENT_LOCK.json"),
        "data_roles": ["outer_training_fold_only"],
        "checkpoint_selection": "fixed_final_epoch_10_no_oof_selection",
        "test_access": "PROHIBITED",
        "full_data_generator_reused": False,
    }
    return {**identity, "identity_sha256": object_hash(identity)}


def configure_shared_runner() -> None:
    w14.CONFIG_PATH = CONFIG_PATH
    w14.RUN_ROOT = RUN_ROOT
    w14.load_config = load_config
    w14.subset_path_for = subset_path_for
    w14.run_identity = run_identity


def preflight() -> dict[str, Any]:
    config = load_config()
    lock, protocol_hash = w09.verify_protocol()
    inputs = w09.verify_inputs()
    fold_path = EVIDENCE_ROOT / "W16_FOLD_RECEIPT.json"
    core_path = ROOT / "evidence" / "detector_core" / "CORE_COMPLETENESS.json"
    low_path = ROOT / "evidence" / "low_data" / "W14_DETECTOR_COMPLETENESS.json"
    fold_receipt = json.loads(fold_path.read_text(encoding="utf-8")) if fold_path.exists() else None
    core = json.loads(core_path.read_text(encoding="utf-8")) if core_path.exists() else None
    low = json.loads(low_path.read_text(encoding="utf-8")) if low_path.exists() else None
    checks = {
        "folds_complete": fold_receipt is not None and fold_receipt["status"] == "PASS" and len(fold_receipt["artifacts"]) == 18,
        "nine_generator_refits": len(roster()) == 9 and config["generator_refits"]["NP"] == 9,
        "w12_core_complete": core is not None and core["status"] == "COMPLETE",
        "w14_detector_complete": low is not None and low["status"] == "COMPLETE",
        "input_receipt_test_free": inputs["test_rows_written"] == 0 and inputs["test_image_or_label_payload_accessed"] is False,
        "test_locked": lock["execution_gates"]["test_data_access_authorized_now"] is False,
    }
    prerequisites = {"w12_core_complete", "w14_detector_complete"}
    status = "PASS" if all(checks.values()) else "WAITING_FOR_PREREQUISITES" if all(
        value for key, value in checks.items() if key not in prerequisites
    ) else "FAIL"
    return {
        "schema_version": 1,
        "status": status,
        "checks": checks,
        "protocol_sha256": protocol_hash,
        "config_sha256": w09.sha256_file(CONFIG_PATH),
        "test_payload_accessed": False,
    }


def train_one(spec: dict[str, Any]) -> dict[str, Any]:
    configure_shared_runner()
    return w14.build_pools(spec, w14.train_one(spec))


def summarize() -> dict[str, Any]:
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
    receipt = {
        "schema_version": 1,
        "status": "COMPLETE" if len(completed) == len(pools) == 9 and not missing else "INCOMPLETE",
        "planned_generator_fits": 9,
        "completed_generator_fits": len(completed),
        "completed_pools": len(pools),
        "missing_run_ids": missing,
        "failure_records": failures,
        "cumulative_gpu_hours": sum(float(row["gpu_hours"]) for row in completed),
        "checkpoint_policy": "fixed_final_epoch_10_no_oof_selection",
        "full_data_generator_or_pool_reused": False,
        "config_sha256": w09.sha256_file(CONFIG_PATH),
        "test_payload_accessed": False,
    }
    EVIDENCE_ROOT.mkdir(parents=True, exist_ok=True)
    w09.atomic_json(EVIDENCE_ROOT / "W16_GENERATOR_COMPLETENESS.json", receipt)
    return receipt


def accounted_gpu_hours() -> float:
    candidates = (
        (ROOT / "evidence/generator_factorial/W09_COMPLETENESS.json", ("cumulative_generator_gpu_hours",)),
        (ROOT / "evidence/prior_sweep/W10_SWEEP_COMPLETENESS.json", ("cumulative_gpu_hours",)),
        (ROOT / "evidence/stronger_generator/W11_DIFFUSION_COMPLETENESS.json", ("cumulative_gpu_hours",)),
        (ROOT / "evidence/detector_core/CORE_COMPLETENESS.json", ("training_gpu_hours", "validation_gpu_hours")),
        (ROOT / "evidence/resolution/W15_COMPLETENESS.json", ("evaluation_gpu_hours",)),
        (ROOT / "evidence/low_data/W14_GENERATOR_COMPLETENESS.json", ("cumulative_gpu_hours",)),
        (ROOT / "evidence/low_data/W14_DETECTOR_COMPLETENESS.json", ("training_gpu_hours", "validation_gpu_hours")),
    )
    total = 0.0
    for path, keys in candidates:
        if path.exists():
            value = json.loads(path.read_text(encoding="utf-8"))
            total += sum(float(value.get(key, 0.0)) for key in keys)
    return total + float(summarize()["cumulative_gpu_hours"])


def run_all() -> dict[str, Any]:
    gate = preflight()
    if gate["status"] != "PASS":
        raise RuntimeError(f"W16 generator preflight is {gate['status']}")
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
                w09.atomic_json(EVIDENCE_ROOT / "W16_GENERATOR_COMPLETENESS.json", result)
                return result
        train_one(spec)
        summarize()
    return summarize()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--preflight", action="store_true")
    action.add_argument("--run-all", action="store_true")
    action.add_argument("--run-one", action="store_true")
    action.add_argument("--summarize", action="store_true")
    parser.add_argument("--partition-seed", type=int)
    parser.add_argument("--fold", type=int)
    args = parser.parse_args()
    if args.preflight:
        result = preflight()
    elif args.run_all:
        result = run_all()
    elif args.run_one:
        matches = [value for value in roster() if value["partition_seed"] == args.partition_seed and value["fold"] == args.fold]
        if len(matches) != 1:
            raise ValueError("Requested W16 generator is outside the frozen roster")
        result = train_one(matches[0])
    else:
        result = summarize()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] != "FAIL" else 1


if __name__ == "__main__":
    raise SystemExit(main())
