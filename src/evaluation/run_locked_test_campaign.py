"""Freeze pretest evidence, unlock, and run the single fixed N2 test campaign."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
from datetime import datetime, timezone
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "vendor" / "parquet"))
import pandas as pd  # noqa: E402

sys.path.insert(0, str(ROOT / "src" / "generators"))
from run_w09_factorial import atomic_json, read_csv, sha256_file, verify_inputs, verify_protocol  # noqa: E402

sys.path.insert(0, str(ROOT / "src" / "detectors"))
import run_w12_detectors as w12  # noqa: E402
import run_w14_lowdata_detectors as w14  # noqa: E402
from detector_common import DetectorRows, build_model, official_coco_metrics, per_specimen_metrics, predict, write_gzip_json  # noqa: E402

CONFIG_PATH = ROOT / "configs" / "w17_test_campaign.json"
PRETEST_ROOT = ROOT / "evidence" / "pretest"
EVIDENCE_ROOT = ROOT / "evidence" / "test_campaign"
RUN_ROOT = ROOT / "runs" / "new_study" / "W17_test_campaign"
TEST_MANIFEST_PATH = ROOT / "data" / "manifests" / "w17" / "test.csv"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def object_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def load_config() -> dict[str, Any]:
    value = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if value["status"] != "FROZEN_BEFORE_FIRST_N2_TEST_PAYLOAD_ACCESS":
        raise RuntimeError("W17 test campaign is not frozen")
    if value["total_checkpoint_evaluations"] != 208:
        raise RuntimeError("W17 test roster count differs")
    return value


def required_receipts() -> dict[str, tuple[Path, set[str]]]:
    return {
        "W09": (ROOT / "evidence" / "generator_factorial" / "W09_COMPLETENESS.json", {"COMPLETE"}),
        "W10": (ROOT / "evidence" / "prior_sweep" / "W10_SWEEP_COMPLETENESS.json", {"COMPLETE"}),
        "W11_diffusion": (ROOT / "evidence" / "stronger_generator" / "W11_DIFFUSION_COMPLETENESS.json", {"COMPLETE"}),
        "W11_pools": (ROOT / "evidence" / "stronger_generator" / "W11_POOL_COMPLETENESS.json", {"COMPLETE"}),
        "W11_quality": (ROOT / "evidence" / "stronger_generator" / "W11_QUALITY_COMPLETENESS.json", {"PASS_WITH_SCOPE_LIMITATION"}),
        "W12": (ROOT / "evidence" / "detector_core" / "CORE_COMPLETENESS.json", {"COMPLETE"}),
        "W14_generators": (ROOT / "evidence" / "low_data" / "W14_GENERATOR_COMPLETENESS.json", {"COMPLETE"}),
        "W14_detectors": (ROOT / "evidence" / "low_data" / "W14_DETECTOR_COMPLETENESS.json", {"COMPLETE"}),
        "W14_integrity": (ROOT / "evidence" / "low_data" / "W14_FULL_INTEGRITY_AUDIT.json", {"PASS"}),
        "W15": (ROOT / "evidence" / "resolution" / "W15_COMPLETENESS.json", {"PASS"}),
        "W16_folds": (ROOT / "evidence" / "grouped_internal" / "W16_FOLD_RECEIPT.json", {"PASS"}),
        "W16_generators": (ROOT / "evidence" / "grouped_internal" / "W16_GENERATOR_COMPLETENESS.json", {"COMPLETE"}),
        "W16_detectors": (ROOT / "evidence" / "grouped_internal" / "W16_DETECTOR_COMPLETENESS.json", {"COMPLETE"}),
        "W16_integrity": (ROOT / "evidence" / "grouped_internal" / "W16_FULL_INTEGRITY_AUDIT.json", {"PASS"}),
    }


def inspect_required_receipts() -> dict[str, dict[str, Any]]:
    checks: dict[str, dict[str, Any]] = {}
    for name, (path, accepted) in required_receipts().items():
        if not path.exists():
            checks[name] = {"status": "MISSING", "accepted": False, "test_payload_accessed": None}
            continue
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            checks[name] = {"status": "INVALID_JSON", "accepted": False, "test_payload_accessed": None}
            continue
        status = value.get("status")
        test_access = value.get("test_payload_accessed", value.get("held_out_test_payload_accessed", False))
        checks[name] = {
            "status": status,
            "accepted": status in accepted and test_access is False,
            "test_payload_accessed": test_access,
        }
    return checks


def test_roster() -> list[dict[str, Any]]:
    values = []
    for spec in w12.trajectory_roster(w12.load_config()):
        values.append({
            "source_block": "W12_core",
            "run_id": spec["run_id"],
            "family": spec["family"],
            "arm": spec["arm"],
            "detector_seed": spec["detector_seed"],
            "pipeline_seed": spec.get("pool_seed", spec["detector_seed"]),
            "checkpoint_path": str((w12.RUN_ROOT / spec["run_id"] / "checkpoints" / "milestone_8960.pt").relative_to(ROOT)).replace("\\", "/"),
            "completion_path": str((w12.RUN_ROOT / spec["run_id"] / "completion.json").relative_to(ROOT)).replace("\\", "/"),
        })
    for spec in w14.trajectory_roster():
        values.append({
            "source_block": "W14_low_data",
            "run_id": spec["run_id"],
            "family": spec["family"],
            "arm": spec["arm"],
            "detector_seed": spec["detector_seed"],
            "pipeline_seed": spec["pipeline_seed"],
            "realization": spec["realization"],
            "subset_size": spec["subset_size"],
            "checkpoint_path": str((w14.RUN_ROOT / spec["run_id"] / "checkpoints" / "milestone_8960.pt").relative_to(ROOT)).replace("\\", "/"),
            "completion_path": str((w14.RUN_ROOT / spec["run_id"] / "completion.json").relative_to(ROOT)).replace("\\", "/"),
        })
    if len(values) != 208 or len({(row["source_block"], row["run_id"]) for row in values}) != 208:
        raise RuntimeError("W17 test campaign roster must contain 208 unique checkpoints")
    return values


def source_hashes() -> dict[str, str]:
    paths = [
        CONFIG_PATH,
        ROOT / "configs" / "PROTOCOL_LOCK.yaml",
        ROOT / "configs" / "w12_detector_core.json",
        ROOT / "configs" / "w14_low_data.json",
        ROOT / "configs" / "w15_resolution.json",
        ROOT / "configs" / "w16_grouped_internal.json",
        ROOT / "ENVIRONMENT_LOCK.json",
        ROOT / "src" / "detectors" / "detector_common.py",
        ROOT / "src" / "detectors" / "run_w12_detectors.py",
        ROOT / "src" / "detectors" / "run_w14_lowdata_detectors.py",
        ROOT / "src" / "detectors" / "run_w16_grouped_detectors.py",
        ROOT / "src" / "analysis" / "analyze_w15_resolution.py",
        ROOT / "configs" / "w17_statistics.json",
        ROOT / "src" / "analysis" / "run_w17_statistics.py",
        Path(__file__),
    ]
    return {str(path.relative_to(ROOT)).replace("\\", "/"): sha256_file(path) for path in paths}


def freeze_pretest() -> dict[str, Any]:
    load_config()
    lock, protocol_hash = verify_protocol()
    inputs = verify_inputs()
    if inputs["test_rows_written"] != 0 or inputs["test_image_or_label_payload_accessed"] is not False:
        raise RuntimeError("Pretest W09 input receipt indicates test payload access")
    receipt_hashes: dict[str, str] = {}
    completion_times: list[str] = []
    for name, (path, statuses) in required_receipts().items():
        if not path.exists():
            raise RuntimeError(f"Missing pretest receipt: {name}")
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("status") not in statuses:
            raise RuntimeError(f"Incomplete pretest receipt: {name}={value.get('status')}")
        if value.get("test_payload_accessed", value.get("held_out_test_payload_accessed", False)) is not False:
            raise RuntimeError(f"Pretest receipt reports test payload access: {name}")
        receipt_hashes[str(path.relative_to(ROOT)).replace("\\", "/")] = sha256_file(path)
        for key in ("completed_at", "finished_at", "updated_at"):
            if value.get(key):
                completion_times.append(str(value[key]))
    checkpoint_rows = []
    for spec in test_roster():
        completion_path = ROOT / spec["completion_path"]
        checkpoint_path = ROOT / spec["checkpoint_path"]
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        digest = sha256_file(checkpoint_path)
        if completion["status"] != "COMPLETED" or completion["primary_checkpoint_step"] != 8960 or completion["primary_checkpoint_sha256"] != digest:
            raise RuntimeError(f"Invalid final checkpoint binding: {spec['source_block']}/{spec['run_id']}")
        checkpoint_rows.append({**spec, "checkpoint_sha256": digest, "completion_sha256": sha256_file(completion_path)})
    freeze = {
        "schema_version": 1,
        "status": "PRETEST_COMPLETE",
        "created_at": utc_now(),
        "protocol_sha256": protocol_hash,
        "test_campaign_config_sha256": sha256_file(CONFIG_PATH),
        "receipt_hashes": receipt_hashes,
        "source_hashes": source_hashes(),
        "checkpoint_roster": checkpoint_rows,
        "checkpoint_roster_sha256": object_hash(checkpoint_rows),
        "checkpoint_count": len(checkpoint_rows),
        "known_pretest_completion_times": sorted(completion_times),
        "protocol_incident": "docs/PROTOCOL_INCIDENT_001_TEST_IDENTIFIER_DISPLAY.md",
        "test_image_label_prediction_or_metric_payload_accessed": False,
    }
    PRETEST_ROOT.mkdir(parents=True, exist_ok=True)
    path = PRETEST_ROOT / "PRETEST_FREEZE.json"
    if path.exists():
        prior = json.loads(path.read_text(encoding="utf-8"))
        comparable = {key: value for key, value in freeze.items() if key != "created_at"}
        prior_comparable = {key: value for key, value in prior.items() if key != "created_at"}
        if comparable != prior_comparable:
            raise RuntimeError("Existing pretest freeze differs")
        return prior
    atomic_json(path, freeze)
    return freeze


def verify_freeze(freeze: dict[str, Any]) -> None:
    if freeze.get("status") != "PRETEST_COMPLETE" or freeze.get("checkpoint_count") != 208:
        raise RuntimeError("Pretest freeze is incomplete")
    for relative, digest in freeze["receipt_hashes"].items():
        if sha256_file(ROOT / relative) != digest:
            raise RuntimeError(f"Changed pretest receipt: {relative}")
    for relative, digest in freeze["source_hashes"].items():
        if sha256_file(ROOT / relative) != digest:
            raise RuntimeError(f"Changed pretest source: {relative}")
    for row in freeze["checkpoint_roster"]:
        if sha256_file(ROOT / row["checkpoint_path"]) != row["checkpoint_sha256"]:
            raise RuntimeError(f"Changed frozen checkpoint: {row['checkpoint_path']}")
    if object_hash(freeze["checkpoint_roster"]) != freeze["checkpoint_roster_sha256"]:
        raise RuntimeError("Pretest checkpoint roster hash differs")


def unlock() -> dict[str, Any]:
    freeze_path = PRETEST_ROOT / "PRETEST_FREEZE.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    verify_freeze(freeze)
    unlock_path = PRETEST_ROOT / "TEST_UNLOCK.json"
    expected = {
        "schema_version": 1,
        "status": "AUTHORIZED_FOR_SINGLE_LOCKED_CAMPAIGN",
        "campaign_id": load_config()["campaign_id"],
        "pretest_freeze_sha256": sha256_file(freeze_path),
        "checkpoint_roster_sha256": freeze["checkpoint_roster_sha256"],
        "checkpoint_count": 208,
        "model_or_threshold_selection_from_test": "PROHIBITED",
    }
    if unlock_path.exists():
        prior = json.loads(unlock_path.read_text(encoding="utf-8"))
        if {key: prior.get(key) for key in expected} != expected:
            raise RuntimeError("Existing test unlock differs")
        return prior
    value = {**expected, "unlocked_at": utc_now(), "test_payload_accessed_at_write": False}
    atomic_json(unlock_path, value)
    return value


def require_unlock() -> tuple[dict[str, Any], dict[str, Any]]:
    freeze_path = PRETEST_ROOT / "PRETEST_FREEZE.json"
    unlock_path = PRETEST_ROOT / "TEST_UNLOCK.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    unlock_value = json.loads(unlock_path.read_text(encoding="utf-8"))
    verify_freeze(freeze)
    if unlock_value.get("pretest_freeze_sha256") != sha256_file(freeze_path) or unlock_value.get("status") != "AUTHORIZED_FOR_SINGLE_LOCKED_CAMPAIGN":
        raise RuntimeError("Test unlock does not bind the current pretest freeze")
    return freeze, unlock_value


def atomic_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".csv", delete=False, mode="w", encoding="utf-8", newline="") as handle:
        temporary = Path(handle.name)
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def materialize_test_manifest() -> tuple[list[dict[str, str]], str]:
    freeze, unlock_value = require_unlock()
    access_path = EVIDENCE_ROOT / "TEST_ACCESS_EVENT.json"
    EVIDENCE_ROOT.mkdir(parents=True, exist_ok=True)
    if not access_path.exists():
        atomic_json(access_path, {
            "schema_version": 1,
            "status": "AUTHORIZED_FIRST_N2_TEST_PAYLOAD_ACCESS_STARTED",
            "started_at": utc_now(),
            "campaign_id": unlock_value["campaign_id"],
            "pretest_freeze_sha256": sha256_file(PRETEST_ROOT / "PRETEST_FREEZE.json"),
            "combined_manifest_path": "data/manifests/image_manifest.csv",
            "selection_rule": "split_equals_test",
            "prior_identifier_display_incident": "docs/PROTOCOL_INCIDENT_001_TEST_IDENTIFIER_DISPLAY.md",
        })
    source_path = ROOT / "data" / "manifests" / "image_manifest.csv"
    lock, _ = verify_protocol()
    if sha256_file(source_path) != lock["frozen_inputs"]["derived_image_manifest"]["sha256"]:
        raise RuntimeError("Combined source manifest differs from protocol lock")
    values = [row for row in read_csv(source_path) if row["split"] == "test"]
    if len(values) != 1024 or len({row["specimen"] for row in values}) != 32:
        raise RuntimeError("N2 test population differs")
    if TEST_MANIFEST_PATH.exists():
        event = json.loads(access_path.read_text(encoding="utf-8"))
        digest = sha256_file(TEST_MANIFEST_PATH)
        if event.get("test_manifest_sha256") not in {None, digest}:
            raise RuntimeError("Existing materialized test manifest differs")
        if event.get("test_manifest_sha256") is None:
            recovered = read_csv(TEST_MANIFEST_PATH)
            if len(recovered) != 1024 or len({row["specimen"] for row in recovered}) != 32 or any(row["split"] != "test" for row in recovered):
                raise RuntimeError("Interrupted test-manifest transaction produced an invalid roster")
            event.update({
                "status": "AUTHORIZED_TEST_MANIFEST_MATERIALIZED",
                "materialized_at": utc_now(),
                "test_rows": len(recovered),
                "test_specimens": len({row["specimen"] for row in recovered}),
                "test_manifest_path": str(TEST_MANIFEST_PATH.relative_to(ROOT)).replace("\\", "/"),
                "test_manifest_sha256": digest,
                "transaction_recovered": True,
            })
            atomic_json(access_path, event)
    else:
        atomic_csv(TEST_MANIFEST_PATH, values, list(values[0]))
        event = json.loads(access_path.read_text(encoding="utf-8"))
        event.update({
            "status": "AUTHORIZED_TEST_MANIFEST_MATERIALIZED",
            "materialized_at": utc_now(),
            "test_rows": len(values),
            "test_specimens": len({row["specimen"] for row in values}),
            "test_manifest_path": str(TEST_MANIFEST_PATH.relative_to(ROOT)).replace("\\", "/"),
            "test_manifest_sha256": sha256_file(TEST_MANIFEST_PATH),
        })
        atomic_json(access_path, event)
    return values, sha256_file(TEST_MANIFEST_PATH)


def test_dataset(family: str, rows: list[dict[str, str]]) -> DetectorRows:
    relative = [w12.relative_real_row(row) for row in rows]
    hashes = {row["image_path"]: row["image_sha256"] for row in relative}
    return DetectorRows(relative, family, ROOT, augment_real=False, expected_sha256=hashes)


def run_one(spec: dict[str, Any], rows: list[dict[str, str]], test_manifest_sha256: str) -> dict[str, Any]:
    checkpoint_path = ROOT / spec["checkpoint_path"]
    completion_path = ROOT / spec["completion_path"]
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    identity = {
        "schema_version": 1,
        "campaign_id": load_config()["campaign_id"],
        "source_block": spec["source_block"],
        "run_id": spec["run_id"],
        "family": spec["family"],
        "arm": spec["arm"],
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "completion_sha256": sha256_file(completion_path),
        "test_manifest_sha256": test_manifest_sha256,
        "pretest_freeze_sha256": sha256_file(PRETEST_ROOT / "PRETEST_FREEZE.json"),
        "source_sha256": sha256_file(Path(__file__)),
    }
    if identity["checkpoint_sha256"] != completion["primary_checkpoint_sha256"]:
        raise RuntimeError("Test checkpoint differs from training completion")
    identity = {**identity, "identity_sha256": object_hash(identity)}
    directory = RUN_ROOT / spec["source_block"] / spec["run_id"]
    result_path = directory / "completion.json"
    if result_path.exists():
        prior = json.loads(result_path.read_text(encoding="utf-8"))
        if prior["identity_sha256"] != identity["identity_sha256"]:
            raise RuntimeError("Completed test evaluation identity differs")
        for relative, digest in prior["artifact_hashes"].items():
            if sha256_file(ROOT / relative) != digest:
                raise RuntimeError(f"Changed test artifact: {relative}")
        return prior
    directory.mkdir(parents=True, exist_ok=True)
    atomic_json(directory / "identity.json", identity)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" or torch.cuda.get_device_name(device) != "NVIDIA GeForce RTX 5080":
        raise RuntimeError("W17 test campaign requires the frozen local RTX 5080")
    detector_config = w12.load_config()
    weight_path = ROOT / detector_config["models"][spec["family"]]["weights_path"]
    torch.cuda.synchronize()
    started = time.perf_counter()
    model = build_model(spec["family"], detector_config, weight_path).to(device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    predictions = predict(model, test_dataset(spec["family"], rows), device, batch_size=4)
    metrics = official_coco_metrics(predictions)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    predictions_path = directory / "predictions.json.gz"
    metrics_path = directory / "metrics.json"
    specimen_path = directory / "per_specimen_metrics.parquet"
    write_gzip_json(predictions_path, predictions)
    atomic_json(metrics_path, {**identity, "split": "held_out_test", **metrics})
    pd.DataFrame(per_specimen_metrics(predictions)).to_parquet(specimen_path, index=False)
    del model
    torch.cuda.empty_cache()
    artifact_paths = [directory / "identity.json", predictions_path, metrics_path, specimen_path]
    result = {
        "schema_version": 1,
        "status": "COMPLETED",
        **{key: value for key, value in spec.items() if key not in {"checkpoint_path", "completion_path"}},
        "identity_sha256": identity["identity_sha256"],
        "checkpoint_sha256": identity["checkpoint_sha256"],
        "test_manifest_sha256": test_manifest_sha256,
        "metrics": metrics,
        "evaluation_gpu_hours": elapsed / 3600.0,
        "test_payload_accessed": True,
        "artifact_hashes": {str(path.relative_to(ROOT)).replace("\\", "/"): sha256_file(path) for path in artifact_paths},
    }
    atomic_json(result_path, result)
    return result


def test_access_occurred() -> bool:
    return (EVIDENCE_ROOT / "TEST_ACCESS_EVENT.json").exists()


def summarize() -> dict[str, Any]:
    completed, missing, failures = [], [], []
    for spec in test_roster():
        directory = RUN_ROOT / spec["source_block"] / spec["run_id"]
        path = directory / "completion.json"
        if path.exists():
            completed.append(json.loads(path.read_text(encoding="utf-8")))
        else:
            missing.append(f"{spec['source_block']}/{spec['run_id']}")
            failures.extend(str(path.relative_to(ROOT)).replace("\\", "/") for path in directory.glob("failure_*.json"))
    manifest_rows = [{
        "source_block": row["source_block"], "run_id": row["run_id"], "family": row["family"],
        "arm": row["arm"], "detector_seed": row["detector_seed"], "pipeline_seed": row["pipeline_seed"],
        "realization": row.get("realization"), "subset_size": row.get("subset_size"),
        "checkpoint_sha256": row["checkpoint_sha256"], "evaluation_gpu_hours": row["evaluation_gpu_hours"],
        **{f"test_{key}": value for key, value in row["metrics"].items()},
    } for row in completed]
    EVIDENCE_ROOT.mkdir(parents=True, exist_ok=True)
    manifest_path = EVIDENCE_ROOT / "test_metrics.parquet"
    if manifest_rows:
        pd.DataFrame(manifest_rows).to_parquet(manifest_path, index=False)
    receipt = {
        "schema_version": 1,
        "status": "COMPLETE" if len(completed) == 208 and not missing else "INCOMPLETE",
        "campaign_id": load_config()["campaign_id"],
        "planned_evaluations": 208,
        "completed_evaluations": len(completed),
        "missing": missing,
        "failure_records": failures,
        "evaluation_gpu_hours": sum(float(row["evaluation_gpu_hours"]) for row in completed),
        "test_payload_accessed": test_access_occurred(),
        "metrics_manifest": str(manifest_path.relative_to(ROOT)).replace("\\", "/") if manifest_rows else None,
        "metrics_manifest_sha256": sha256_file(manifest_path) if manifest_rows else None,
    }
    atomic_json(EVIDENCE_ROOT / "W17_TEST_COMPLETENESS.json", receipt)
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
        (ROOT / "evidence/grouped_internal/W16_GENERATOR_COMPLETENESS.json", ("cumulative_gpu_hours",)),
        (ROOT / "evidence/grouped_internal/W16_DETECTOR_COMPLETENESS.json", ("training_gpu_hours", "evaluation_gpu_hours")),
    )
    total = 0.0
    for path, keys in candidates:
        if path.exists():
            value = json.loads(path.read_text(encoding="utf-8"))
            total += sum(float(value.get(key, 0.0)) for key in keys)
    return total + float(summarize()["evaluation_gpu_hours"])


def run_all(session_hours: float) -> dict[str, Any]:
    if not 0 < session_hours < 12:
        raise ValueError("W17 session-hours must be positive and below 12")
    current = summarize()
    if current["status"] == "COMPLETE":
        return current
    ceiling = float(w12.load_config()["compute"]["cumulative_project_ceiling_gpu_hours"])
    used = accounted_gpu_hours()
    if used + 0.75 >= ceiling:
        current["status"] = "PAUSED_COMPUTE_CEILING_GUARD"
        current["accounted_gpu_hours"] = used
        atomic_json(EVIDENCE_ROOT / "W17_TEST_COMPLETENESS.json", current)
        return current
    freeze_pretest()
    unlock()
    rows, test_hash = materialize_test_manifest()
    started = time.perf_counter()
    for spec in test_roster():
        if (RUN_ROOT / spec["source_block"] / spec["run_id"] / "completion.json").exists():
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
            atomic_json(EVIDENCE_ROOT / "W17_TEST_COMPLETENESS.json", result)
            return result
        try:
            run_one(spec, rows, test_hash)
            summarize()
        except BaseException as error:
            directory = RUN_ROOT / spec["source_block"] / spec["run_id"]
            directory.mkdir(parents=True, exist_ok=True)
            atomic_json(directory / f"failure_{time.time_ns()}.json", {
                "schema_version": 1, "status": "FAILED_OR_INTERRUPTED", "source_block": spec["source_block"],
                "run_id": spec["run_id"], "error_type": type(error).__name__, "error": str(error),
                "seed_replaced": False, "test_payload_accessed": True,
            })
            raise
    return summarize()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--preflight", action="store_true")
    action.add_argument("--freeze-pretest", action="store_true")
    action.add_argument("--unlock", action="store_true")
    action.add_argument("--run-all", action="store_true")
    action.add_argument("--summarize", action="store_true")
    parser.add_argument("--session-hours", type=float, default=10.5)
    args = parser.parse_args()
    if args.preflight:
        receipt_checks = inspect_required_receipts()
        ready = all(item["accepted"] for item in receipt_checks.values())
        result = {
            "status": "READY_TO_FREEZE" if ready else "WAITING_FOR_PRETEST",
            "test_roster_count": len(test_roster()),
            "receipt_checks": receipt_checks,
            "test_payload_accessed": False,
        }
    elif args.freeze_pretest:
        result = freeze_pretest()
    elif args.unlock:
        result = unlock()
    elif args.run_all:
        result = run_all(args.session_hours)
    else:
        result = summarize()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
