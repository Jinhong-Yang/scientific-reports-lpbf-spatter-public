"""Create a hash-bound, full-trajectory integrity receipt for W16.

W16 consists of nine training-fold-only generator refits and thirty-six
development-only grouped detector trajectories.  This audit is deliberately
separate from the runners: it verifies every receipt-listed artifact after the
campaign is complete, without opening the held-out test payload or changing a
study result.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
GENERATOR_ROOT = ROOT / "runs" / "new_study" / "W16_grouped_generators"
DETECTOR_ROOT = ROOT / "runs" / "new_study" / "W16_grouped_detectors"
EVIDENCE_ROOT = ROOT / "evidence" / "grouped_internal"
GENERATOR_RECEIPT = EVIDENCE_ROOT / "W16_GENERATOR_COMPLETENESS.json"
DETECTOR_RECEIPT = EVIDENCE_ROOT / "W16_DETECTOR_COMPLETENESS.json"
OUTPUT = EVIDENCE_ROOT / "W16_FULL_INTEGRITY_AUDIT.json"
EXPECTED_GENERATORS = 9
EXPECTED_DETECTORS = 36
EXPECTED_DETECTOR_UPDATES = 8960


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def artifact_mismatches(receipt: dict[str, Any], base: Path) -> list[dict[str, str]]:
    """Hash receipt-listed artifacts whose keys are relative to *base*."""
    problems: list[dict[str, str]] = []
    for relative_path, expected_hash in sorted(receipt.get("artifact_hashes", {}).items()):
        artifact = base / relative_path
        if not artifact.is_file():
            problems.append({"path": relative_path, "reason": "MISSING"})
        elif sha256(artifact).lower() != str(expected_hash).lower():
            problems.append({"path": relative_path, "reason": "SHA256_MISMATCH"})
    return problems


def audit() -> dict[str, Any]:
    generator_reports: list[dict[str, Any]] = []
    for summary_path in sorted(GENERATOR_ROOT.rglob("summary.json")):
        summary = read_json(summary_path)
        pool_path = summary_path.parent / "pool_summary.json"
        pool = read_json(pool_path) if pool_path.is_file() else {}
        summary_problems = artifact_mismatches(summary, summary_path.parent)
        pool_problems = artifact_mismatches(pool, summary_path.parent) if pool else [{"path": "pool_summary.json", "reason": "MISSING"}]
        generator_reports.append({
            "run_id": summary_path.parent.name,
            "summary_status": summary.get("status"),
            "epochs": summary.get("epochs"),
            "checkpoint_policy": summary.get("checkpoint_policy"),
            "test_payload_accessed": summary.get("test_payload_accessed"),
            "full_data_generator_reused": summary.get("full_data_generator_reused"),
            "summary_artifact_hash_count": len(summary.get("artifact_hashes", {})),
            "pool_status": pool.get("status"),
            "pool_test_payload_accessed": pool.get("test_payload_accessed"),
            "full_data_pool_reused": pool.get("full_data_pool_reused"),
            "pool_artifact_hash_count": len(pool.get("artifact_hashes", {})),
            "artifact_mismatches": summary_problems + pool_problems,
        })

    detector_reports: list[dict[str, Any]] = []
    for receipt_path in sorted(DETECTOR_ROOT.rglob("completion.json")):
        receipt = read_json(receipt_path)
        identity_path = receipt_path.parent / "identity.json"
        identity = read_json(identity_path) if identity_path.is_file() else {}
        problems: list[dict[str, str]] = []
        for relative_path, expected_hash in sorted(receipt.get("artifact_hashes", {}).items()):
            artifact = ROOT / relative_path
            if not artifact.is_file():
                problems.append({"path": relative_path, "reason": "MISSING"})
            elif sha256(artifact).lower() != str(expected_hash).lower():
                problems.append({"path": relative_path, "reason": "SHA256_MISMATCH"})
        detector_reports.append({
            "run_id": receipt_path.parent.name,
            "status": receipt.get("status"),
            "optimizer_updates": receipt.get("optimizer_updates"),
            "test_payload_accessed": receipt.get("test_payload_accessed"),
            "identity_present": identity_path.is_file(),
            "identity_test_payload_accessed": identity.get("test_payload_accessed"),
            "full_data_generator_or_pool_reused": identity.get("full_data_generator_or_pool_reused"),
            "artifact_hash_count": len(receipt.get("artifact_hashes", {})),
            "artifact_mismatches": problems,
        })

    generator_receipt = read_json(GENERATOR_RECEIPT)
    detector_receipt = read_json(DETECTOR_RECEIPT)
    bad_generators = [
        report for report in generator_reports
        if report["summary_status"] != "COMPLETED"
        or report["epochs"] != 10
        or report["checkpoint_policy"] != "fixed_final_epoch_10_no_oof_selection"
        or report["test_payload_accessed"] is not False
        or report["full_data_generator_reused"] is not False
        or not report["summary_artifact_hash_count"]
        or report["pool_status"] != "COMPLETED"
        or report["pool_test_payload_accessed"] is not False
        or report["full_data_pool_reused"] is not False
        or not report["pool_artifact_hash_count"]
        or report["artifact_mismatches"]
    ]
    bad_detectors = [
        report for report in detector_reports
        if report["status"] != "COMPLETED"
        or report["optimizer_updates"] != EXPECTED_DETECTOR_UPDATES
        or report["test_payload_accessed"] is not False
        or not report["identity_present"]
        or report["identity_test_payload_accessed"] is not False
        or report["full_data_generator_or_pool_reused"] is not False
        or not report["artifact_hash_count"]
        or report["artifact_mismatches"]
    ]
    manifest_path = ROOT / str(detector_receipt.get("trajectory_manifest") or "")
    manifest_hash_matches = (
        manifest_path.is_file()
        and sha256(manifest_path).lower() == str(detector_receipt.get("trajectory_manifest_sha256")).lower()
    )
    conditions = {
        "exactly_nine_generator_summaries": len(generator_reports) == EXPECTED_GENERATORS,
        "every_generator_and_pool_valid": not bad_generators,
        "generator_receipt_complete": generator_receipt.get("status") == "COMPLETE",
        "generator_receipt_exact_counts": (
            generator_receipt.get("planned_generator_fits") == EXPECTED_GENERATORS
            and generator_receipt.get("completed_generator_fits") == EXPECTED_GENERATORS
            and generator_receipt.get("completed_pools") == EXPECTED_GENERATORS
        ),
        "generator_receipt_no_missing_or_failures": (
            generator_receipt.get("missing_run_ids") == [] and generator_receipt.get("failure_records") == []
        ),
        "generator_receipt_no_test_access": generator_receipt.get("test_payload_accessed") is False,
        "exactly_36_detector_receipts": len(detector_reports) == EXPECTED_DETECTORS,
        "every_detector_valid": not bad_detectors,
        "detector_receipt_complete": detector_receipt.get("status") == "COMPLETE",
        "detector_receipt_exact_counts": (
            detector_receipt.get("planned_trajectories") == EXPECTED_DETECTORS
            and detector_receipt.get("completed_trajectories") == EXPECTED_DETECTORS
        ),
        "detector_receipt_no_missing_or_failures": (
            detector_receipt.get("missing_run_ids") == [] and detector_receipt.get("failure_records") == []
        ),
        "detector_receipt_no_test_access": detector_receipt.get("held_out_test_payload_accessed") is False,
        "detector_manifest_hash_matches": manifest_hash_matches,
    }
    return {
        "schema_version": 1,
        "audit_name": "W16_FULL_INTEGRITY_AUDIT",
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "all W16 generator/pool and detector completion artifacts",
        "status": "PASS" if all(conditions.values()) else "FAIL",
        "conditions": conditions,
        "summary": {
            "generator_summaries": len(generator_reports),
            "expected_generators": EXPECTED_GENERATORS,
            "detector_receipts": len(detector_reports),
            "expected_detectors": EXPECTED_DETECTORS,
            "expected_detector_optimizer_updates": EXPECTED_DETECTOR_UPDATES,
            "generators_with_problems": len(bad_generators),
            "detectors_with_problems": len(bad_detectors),
            "artifact_hashes_checked": sum(
                report["summary_artifact_hash_count"] + report["pool_artifact_hash_count"]
                for report in generator_reports
            ) + sum(report["artifact_hash_count"] for report in detector_reports),
        },
        "problems": {"generators": bad_generators, "detectors": bad_detectors},
    }


def main() -> int:
    result = audit()
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], sort_keys=True))
    print(f"W16_FULL_INTEGRITY_AUDIT={result['status']}")
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
