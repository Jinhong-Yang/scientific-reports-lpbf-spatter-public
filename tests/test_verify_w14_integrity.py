"""Unit coverage for the W14 full-trajectory integrity audit."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "verify_w14_integrity", ROOT / "scripts" / "verify_w14_integrity.py"
)
assert SPEC and SPEC.loader
VERIFY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFY)


def test_audit_requires_all_108_hash_valid_trajectories(tmp_path, monkeypatch) -> None:
    campaign = tmp_path / "runs" / "new_study" / "W14_low_data_detectors"
    evidence = tmp_path / "evidence" / "low_data"
    evidence.mkdir(parents=True)
    for index in range(108):
        run_id = f"run_{index:03d}"
        run_dir = campaign / run_id
        run_dir.mkdir(parents=True)
        artifact = run_dir / "artifact.txt"
        artifact.write_text(run_id, encoding="utf-8")
        relative = artifact.relative_to(tmp_path).as_posix()
        digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
        (run_dir / "completion.json").write_text(
            json.dumps(
                {
                    "status": "COMPLETED",
                    "optimizer_updates": 8960,
                    "test_payload_accessed": False,
                    "artifact_hashes": {relative: digest},
                }
            ),
            encoding="utf-8",
        )
    completeness = evidence / "W14_DETECTOR_COMPLETENESS.json"
    completeness.write_text(
        json.dumps(
            {
                "status": "COMPLETE",
                "planned_trajectories": 108,
                "completed_trajectories": 108,
                "missing_run_ids": [],
                "failure_records": [],
                "test_payload_accessed": False,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(VERIFY, "ROOT", tmp_path)
    monkeypatch.setattr(VERIFY, "W14_ROOT", campaign)
    monkeypatch.setattr(VERIFY, "COMPLETENESS_RECEIPT", completeness)

    result = VERIFY.audit()

    assert result["status"] == "PASS"
    assert result["summary"]["trajectory_receipts"] == 108
    assert result["summary"]["artifact_hashes_checked"] == 108


def test_audit_rejects_a_missing_receipt(tmp_path, monkeypatch) -> None:
    campaign = tmp_path / "runs" / "new_study" / "W14_low_data_detectors"
    campaign.mkdir(parents=True)
    evidence = tmp_path / "evidence" / "low_data"
    evidence.mkdir(parents=True)
    completeness = evidence / "W14_DETECTOR_COMPLETENESS.json"
    completeness.write_text(
        json.dumps(
            {
                "status": "COMPLETE",
                "planned_trajectories": 108,
                "completed_trajectories": 108,
                "missing_run_ids": [],
                "failure_records": [],
                "test_payload_accessed": False,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(VERIFY, "ROOT", tmp_path)
    monkeypatch.setattr(VERIFY, "W14_ROOT", campaign)
    monkeypatch.setattr(VERIFY, "COMPLETENESS_RECEIPT", completeness)

    result = VERIFY.audit()

    assert result["status"] == "FAIL"
    assert result["conditions"]["exactly_108_receipts"] is False
