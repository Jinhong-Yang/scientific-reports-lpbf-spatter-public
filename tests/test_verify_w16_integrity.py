"""Unit coverage for the W16 full-trajectory integrity audit."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "verify_w16_integrity", ROOT / "scripts" / "verify_w16_integrity.py"
)
assert SPEC and SPEC.loader
VERIFY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFY)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_complete_campaign(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    generators = tmp_path / "runs" / "new_study" / "W16_grouped_generators"
    detectors = tmp_path / "runs" / "new_study" / "W16_grouped_detectors"
    evidence = tmp_path / "evidence" / "grouped_internal"
    evidence.mkdir(parents=True)
    for index in range(9):
        directory = generators / f"generator_{index:02d}"
        directory.mkdir(parents=True)
        model = directory / "best.pt"
        pool = directory / "NP_images_uint8.npy"
        model.write_text(f"model-{index}", encoding="utf-8")
        pool.write_text(f"pool-{index}", encoding="utf-8")
        (directory / "summary.json").write_text(json.dumps({
            "status": "COMPLETED", "epochs": 10,
            "checkpoint_policy": "fixed_final_epoch_10_no_oof_selection",
            "test_payload_accessed": False, "full_data_generator_reused": False,
            "artifact_hashes": {"best.pt": digest(model)},
        }), encoding="utf-8")
        (directory / "pool_summary.json").write_text(json.dumps({
            "status": "COMPLETED", "test_payload_accessed": False,
            "full_data_pool_reused": False,
            "artifact_hashes": {"NP_images_uint8.npy": digest(pool)},
        }), encoding="utf-8")
    for index in range(36):
        directory = detectors / f"detector_{index:02d}"
        directory.mkdir(parents=True)
        artifact = directory / "checkpoint.pt"
        artifact.write_text(f"detector-{index}", encoding="utf-8")
        (directory / "identity.json").write_text(json.dumps({
            "test_payload_accessed": False,
            "full_data_generator_or_pool_reused": False,
        }), encoding="utf-8")
        rel = artifact.relative_to(tmp_path).as_posix()
        (directory / "completion.json").write_text(json.dumps({
            "status": "COMPLETED", "optimizer_updates": 8960,
            "test_payload_accessed": False, "full_data_generator_or_pool_reused": False,
            "artifact_hashes": {rel: digest(artifact)},
        }), encoding="utf-8")
    manifest = evidence / "grouped_detector_manifest.parquet"
    manifest.write_text("synthetic manifest", encoding="utf-8")
    (evidence / "W16_GENERATOR_COMPLETENESS.json").write_text(json.dumps({
        "status": "COMPLETE", "planned_generator_fits": 9,
        "completed_generator_fits": 9, "completed_pools": 9,
        "missing_run_ids": [], "failure_records": [], "test_payload_accessed": False,
    }), encoding="utf-8")
    (evidence / "W16_DETECTOR_COMPLETENESS.json").write_text(json.dumps({
        "status": "COMPLETE", "planned_trajectories": 36,
        "completed_trajectories": 36, "missing_run_ids": [], "failure_records": [],
        "held_out_test_payload_accessed": False,
        "trajectory_manifest": manifest.relative_to(tmp_path).as_posix(),
        "trajectory_manifest_sha256": digest(manifest),
    }), encoding="utf-8")
    return generators, detectors, evidence, manifest


def configure(tmp_path: Path, monkeypatch) -> None:
    generators, detectors, evidence, _ = write_complete_campaign(tmp_path)
    monkeypatch.setattr(VERIFY, "ROOT", tmp_path)
    monkeypatch.setattr(VERIFY, "GENERATOR_ROOT", generators)
    monkeypatch.setattr(VERIFY, "DETECTOR_ROOT", detectors)
    monkeypatch.setattr(VERIFY, "EVIDENCE_ROOT", evidence)
    monkeypatch.setattr(VERIFY, "GENERATOR_RECEIPT", evidence / "W16_GENERATOR_COMPLETENESS.json")
    monkeypatch.setattr(VERIFY, "DETECTOR_RECEIPT", evidence / "W16_DETECTOR_COMPLETENESS.json")


def test_audit_requires_all_frozen_w16_runs_and_hashes(tmp_path, monkeypatch) -> None:
    configure(tmp_path, monkeypatch)
    result = VERIFY.audit()
    assert result["status"] == "PASS"
    assert result["summary"]["generator_summaries"] == 9
    assert result["summary"]["detector_receipts"] == 36
    assert result["summary"]["artifact_hashes_checked"] == 54


def test_audit_rejects_tampered_pool_artifact(tmp_path, monkeypatch) -> None:
    configure(tmp_path, monkeypatch)
    pool = tmp_path / "runs" / "new_study" / "W16_grouped_generators" / "generator_00" / "NP_images_uint8.npy"
    pool.write_text("tampered", encoding="utf-8")
    result = VERIFY.audit()
    assert result["status"] == "FAIL"
    assert result["conditions"]["every_generator_and_pool_valid"] is False
