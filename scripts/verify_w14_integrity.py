"""Create a hash-bound, full-trajectory integrity receipt for W14.

The frozen W14 low-data detector campaign contains exactly 108 trajectories.
This standalone audit reads every completion receipt, hashes each
receipt-listed artifact relative to the managed project root, and writes one
machine-readable final receipt without changing a study artifact.  It must be
run only after the W14 detector completeness receipt reports COMPLETE.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
W14_ROOT = ROOT / "runs" / "new_study" / "W14_low_data_detectors"
COMPLETENESS_RECEIPT = ROOT / "evidence" / "low_data" / "W14_DETECTOR_COMPLETENESS.json"
OUTPUT = ROOT / "evidence" / "low_data" / "W14_FULL_INTEGRITY_AUDIT.json"
EXPECTED_TRAJECTORIES = 108
EXPECTED_UPDATES = 8960


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def audit() -> dict[str, Any]:
    trajectory_reports: list[dict[str, Any]] = []
    for receipt_path in sorted(W14_ROOT.rglob("completion.json")):
        receipt = read_json(receipt_path)
        artifact_hashes = receipt.get("artifact_hashes", {})
        mismatches: list[dict[str, str]] = []
        for relative_path, expected_hash in sorted(artifact_hashes.items()):
            artifact_path = ROOT / relative_path
            if not artifact_path.is_file():
                mismatches.append({"path": relative_path, "reason": "MISSING"})
                continue
            if sha256(artifact_path).lower() != str(expected_hash).lower():
                mismatches.append({"path": relative_path, "reason": "SHA256_MISMATCH"})
        trajectory_reports.append(
            {
                "run_id": receipt_path.parent.name,
                "status": receipt.get("status"),
                "optimizer_updates": receipt.get("optimizer_updates"),
                "test_payload_accessed": receipt.get("test_payload_accessed"),
                "artifact_hash_count": len(artifact_hashes),
                "artifact_mismatches": mismatches,
            }
        )

    completeness = read_json(COMPLETENESS_RECEIPT)
    bad_trajectories = [
        report
        for report in trajectory_reports
        if report["status"] != "COMPLETED"
        or report["optimizer_updates"] != EXPECTED_UPDATES
        or report["test_payload_accessed"] is not False
        or not report["artifact_hash_count"]
        or report["artifact_mismatches"]
    ]
    conditions = {
        "exactly_108_receipts": len(trajectory_reports) == EXPECTED_TRAJECTORIES,
        "every_trajectory_completed": not bad_trajectories,
        "completeness_receipt_complete": completeness.get("status") == "COMPLETE",
        "completeness_receipt_exact_count": (
            completeness.get("planned_trajectories") == EXPECTED_TRAJECTORIES
            and completeness.get("completed_trajectories") == EXPECTED_TRAJECTORIES
        ),
        "completeness_receipt_no_missing": completeness.get("missing_run_ids") == [],
        "completeness_receipt_no_failures": completeness.get("failure_records") == [],
        "completeness_receipt_no_test_access": completeness.get("test_payload_accessed") is False,
    }
    return {
        "schema_version": 1,
        "audit_name": "W14_FULL_INTEGRITY_AUDIT",
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "all W14 completion receipts and every receipt-listed artifact hash",
        "status": "PASS" if all(conditions.values()) else "FAIL",
        "conditions": conditions,
        "summary": {
            "trajectory_receipts": len(trajectory_reports),
            "expected_trajectories": EXPECTED_TRAJECTORIES,
            "expected_optimizer_updates": EXPECTED_UPDATES,
            "trajectories_with_problems": len(bad_trajectories),
            "artifact_hashes_checked": sum(report["artifact_hash_count"] for report in trajectory_reports),
        },
        "completeness_receipt": {
            "status": completeness.get("status"),
            "planned_trajectories": completeness.get("planned_trajectories"),
            "completed_trajectories": completeness.get("completed_trajectories"),
            "missing_run_ids": completeness.get("missing_run_ids"),
            "failure_records": completeness.get("failure_records"),
            "test_payload_accessed": completeness.get("test_payload_accessed"),
        },
        "problems": bad_trajectories,
    }


def main() -> int:
    result = audit()
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], sort_keys=True))
    print(f"W14_FULL_INTEGRITY_AUDIT={result['status']}")
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
