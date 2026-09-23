"""Calculate the frozen validation-only W15 resolution and exposure audit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "vendor" / "parquet"))
import pandas as pd  # noqa: E402

sys.path.insert(0, str(ROOT / "src" / "generators"))
from run_w09_factorial import atomic_json, sha256_file, verify_inputs, verify_protocol  # noqa: E402

sys.path.insert(0, str(ROOT / "src" / "detectors"))
import run_w12_detectors as w12  # noqa: E402
from detector_common import build_model, official_coco_metrics, per_specimen_metrics, predict, write_gzip_json  # noqa: E402

CONFIG_PATH = ROOT / "configs" / "w15_resolution.json"
DETECTOR_CONFIG_PATH = ROOT / "configs" / "w12_detector_core.json"
CORE_ROOT = ROOT / "runs" / "new_study" / "W12_detector_core"
EVIDENCE_ROOT = ROOT / "evidence" / "resolution"


def load_config() -> dict[str, Any]:
    value = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if value["status"] != "FROZEN_BEFORE_W15_RESOLUTION_OUTCOMES" or value["held_out_test_access"] != "PROHIBITED":
        raise RuntimeError("W15 configuration is not frozen and test-locked")
    return value


class Common64Dataset:
    def __init__(self, source):
        self.source = source
        self.family = source.family

    def __len__(self) -> int:
        return len(self.source)

    def __getitem__(self, index: int):
        image, target, meta = self.source[index]
        reduced = F.interpolate(image.unsqueeze(0), size=(64, 64), mode="bilinear", align_corners=False)
        restored = F.interpolate(reduced, size=(300, 300), mode="bilinear", align_corners=False).squeeze(0)
        return restored, target, meta


def selected_specs() -> list[dict[str, Any]]:
    config = load_config()
    specs = [
        row for row in w12.trajectory_roster(w12.load_config())
        if row["block"] == "primary" and row["arm"] in config["analysis_arms"] and row["detector_seed"] in config["pipeline_seeds"]
    ]
    if len(specs) != config["resolution_evaluations"] or len(specs) != 40:
        raise RuntimeError("W15 requires 40 fixed primary checkpoints")
    return specs


def preflight() -> dict[str, Any]:
    config = load_config()
    lock, protocol_hash = verify_protocol()
    inputs = verify_inputs()
    core_path = ROOT / "evidence" / "detector_core" / "CORE_COMPLETENESS.json"
    core = json.loads(core_path.read_text(encoding="utf-8")) if core_path.exists() else None
    checks = {
        "core_complete": core is not None and core["status"] == "COMPLETE" and core["completed_trajectories"] == 100,
        "forty_fixed_checkpoints": len(selected_specs()) == 40,
        "zero_new_training": config["new_training_trajectories"] == 0,
        "higher_resolution_not_run": config["higher_resolution_branch"]["status"].startswith("NOT_RUN"),
        "input_receipt_test_free": inputs["test_rows_written"] == 0 and inputs["test_image_or_label_payload_accessed"] is False,
        "test_locked": lock["execution_gates"]["test_data_access_authorized_now"] is False,
    }
    status = "PASS" if all(checks.values()) else "WAITING_FOR_W12" if not checks["core_complete"] and all(value for key, value in checks.items() if key != "core_complete") else "FAIL"
    return {
        "schema_version": 1,
        "status": status,
        "checks": checks,
        "protocol_sha256": protocol_hash,
        "config_sha256": sha256_file(CONFIG_PATH),
        "test_payload_accessed": False,
    }


def exposure_table() -> pd.DataFrame:
    rows = []
    sources = (
        ("W12_core", ROOT / "evidence" / "detector_core" / "trajectory_manifest.parquet"),
        ("W14_low_data", ROOT / "evidence" / "low_data" / "low_data_detector_manifest.parquet"),
        ("W16_grouped_internal", ROOT / "evidence" / "grouped_internal" / "grouped_detector_manifest.parquet"),
    )
    for block, path in sources:
        if not path.exists():
            continue
        frame = pd.read_parquet(path)
        for record in frame.to_dict("records"):
            rows.append({
                "block": block,
                "run_id": record["run_id"],
                "arm": record["arm"],
                "optimizer_updates": int(record.get("optimizer_updates", 8960)),
                "base_real_roster_rows": int(record.get("base_rows", record.get("training_specimens", 112) * 32)),
                "additional_roster_rows": int(record.get("additional_rows", 0)),
                "real_seen": int(record["real_seen"]),
                "additional_real_seen": int(record["additional_real_seen"]),
                "synthetic_seen": int(record["synthetic_seen"]),
                "training_gpu_hours": float(record["training_gpu_hours"]),
                "evaluation_gpu_hours": float(record.get("validation_gpu_hours", record.get("evaluation_gpu_hours", 0.0))),
                "detector_tensor_shape": "300x300",
                "native_information_path": "real_native300; synthetic_64to300_when_present",
            })
    return pd.DataFrame(rows)


def run() -> dict[str, Any]:
    gate = preflight()
    if gate["status"] != "PASS":
        raise RuntimeError(f"W15 preflight is {gate['status']}")
    config = load_config()
    detector_config = w12.load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" or torch.cuda.get_device_name(device) != "NVIDIA GeForce RTX 5080":
        raise RuntimeError("W15 requires the frozen local RTX 5080")
    base_dataset, _ = w12.validation_dataset(config["detector_family"])
    audit_dataset = Common64Dataset(base_dataset)
    rows, specimen_rows, artifacts = [], [], {}
    evaluation_seconds = 0.0
    EVIDENCE_ROOT.mkdir(parents=True, exist_ok=True)
    for spec in selected_specs():
        directory = CORE_ROOT / spec["run_id"]
        completion = json.loads((directory / "completion.json").read_text(encoding="utf-8"))
        checkpoint_path = directory / "checkpoints" / f"milestone_{config['checkpoint']}.pt"
        if sha256_file(checkpoint_path) != completion["primary_checkpoint_sha256"]:
            raise RuntimeError("W15 checkpoint hash differs from W12 completion")
        native = completion["final_validation_metrics"]
        weight_path = ROOT / detector_config["models"][config["detector_family"]]["weights_path"]
        torch.cuda.synchronize()
        evaluation_started = time.perf_counter()
        model = build_model(config["detector_family"], detector_config, weight_path).to(device)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
        predictions = predict(model, audit_dataset, device, batch_size=4)
        common = official_coco_metrics(predictions)
        prediction_path = EVIDENCE_ROOT / "common64_predictions" / f"{spec['run_id']}.json.gz"
        write_gzip_json(prediction_path, predictions)
        artifacts[str(prediction_path.relative_to(ROOT)).replace("\\", "/")] = sha256_file(prediction_path)
        for path_name, metrics in (("native300", native), ("common64", common)):
            rows.append({
                "run_id": spec["run_id"], "arm": spec["arm"], "pipeline_seed": spec["detector_seed"],
                "resolution_path": path_name, **metrics,
            })
        for record in per_specimen_metrics(predictions):
            specimen_rows.append({
                "run_id": spec["run_id"], "arm": spec["arm"], "pipeline_seed": spec["detector_seed"],
                "resolution_path": "common64", **record,
            })
        torch.cuda.synchronize()
        evaluation_seconds += time.perf_counter() - evaluation_started
        del model
        torch.cuda.empty_cache()
    metrics_path = EVIDENCE_ROOT / "resolution_sensitivity.parquet"
    specimen_path = EVIDENCE_ROOT / "common64_per_specimen.parquet"
    pd.DataFrame(rows).to_parquet(metrics_path, index=False)
    pd.DataFrame(specimen_rows).to_parquet(specimen_path, index=False)
    exposure = exposure_table()
    exposure_path = EVIDENCE_ROOT / "exposure_compute_table.csv"
    exposure.to_csv(exposure_path, index=False)
    paths_path = EVIDENCE_ROOT / "resolution_paths.json"
    atomic_json(paths_path, {
        "schema_version": 1,
        "paths": config["paths"],
        "claims": config["claims"],
        "higher_resolution_branch": config["higher_resolution_branch"],
        "test_payload_accessed": False,
    })
    artifacts.update({
        str(path.relative_to(ROOT)).replace("\\", "/"): sha256_file(path)
        for path in (metrics_path, specimen_path, exposure_path, paths_path)
    })
    frame = pd.DataFrame(rows)
    pivot = frame.pivot(index=["run_id", "arm", "pipeline_seed"], columns="resolution_path", values="COCO_AP").reset_index()
    pivot["common64_minus_native300_COCO_AP"] = pivot["common64"] - pivot["native300"]
    summary = {
        "schema_version": 1,
        "status": "PASS",
        "runs": len(selected_specs()),
        "new_training_trajectories": 0,
        "evaluation_gpu_hours": evaluation_seconds / 3600.0,
        "common64_minus_native300_COCO_AP_mean_by_arm": pivot.groupby("arm")["common64_minus_native300_COCO_AP"].mean().to_dict(),
        "higher_resolution_branch": config["higher_resolution_branch"],
        "artifacts": artifacts,
        "test_payload_accessed": False,
    }
    atomic_json(EVIDENCE_ROOT / "W15_COMPLETENESS.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--preflight", action="store_true")
    action.add_argument("--run", action="store_true")
    args = parser.parse_args()
    result = preflight() if args.preflight else run()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] != "FAIL" else 1


if __name__ == "__main__":
    raise SystemExit(main())
