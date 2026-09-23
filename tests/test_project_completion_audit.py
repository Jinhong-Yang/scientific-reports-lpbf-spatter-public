from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("completion_audit", ROOT / "scripts/audit_project_completion.py")
assert SPEC and SPEC.loader
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


def test_receipt_check_accepts_only_declared_status(tmp_path: Path) -> None:
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps({"status": "PASS"}), encoding="utf-8")
    assert AUDIT.receipt_check(tmp_path, "test", "receipt.json", {"PASS"})["status"] == "PASS"
    assert AUDIT.receipt_check(tmp_path, "test", "receipt.json", {"COMPLETE"})["status"] == "WAITING_OR_FAIL"


def test_receipt_check_does_not_treat_missing_as_pass(tmp_path: Path) -> None:
    result = AUDIT.receipt_check(tmp_path, "test", "missing.json", {"PASS"})
    assert result["status"] == "WAITING_OR_FAIL"
    assert result["detail"] == "missing receipt"


def test_terminal_task_statuses_accept_scope_limited_pass() -> None:
    assert "PASS_WITH_SCOPE_LIMITATION" in AUDIT.TERMINAL_TASK_STATUSES
    assert "RUNNING" not in AUDIT.TERMINAL_TASK_STATUSES


def test_completion_audit_requires_both_pretest_integrity_audits() -> None:
    assert AUDIT.RECEIPTS["W14 full integrity audit"] == (
        "evidence/low_data/W14_FULL_INTEGRITY_AUDIT.json", {"PASS"},
    )
    assert AUDIT.RECEIPTS["W16 full integrity audit"] == (
        "evidence/grouped_internal/W16_FULL_INTEGRITY_AUDIT.json", {"PASS"},
    )


def test_render_requires_every_row_to_pass() -> None:
    result = {
        "status": "WAITING_OR_FAIL", "completion_classification": "NOT_COMPLETE",
        "checks_passed": 1, "checks_total": 2,
        "checks": [
            {"name": "one", "status": "PASS", "detail": "ok", "evidence": "one.json"},
            {"name": "two", "status": "WAITING_OR_FAIL", "detail": "missing", "evidence": "two.json"},
        ],
    }
    text = AUDIT.render(result)
    assert "Checks: 1/2" in text
    assert "goal may be marked complete only when every row reports `PASS`" in text


def test_public_asset_hashes_must_match_local_bytes(tmp_path: Path) -> None:
    path = tmp_path / "artifact.pdf"
    path.write_bytes(b"final bytes")
    expected = AUDIT.sha256_file(path)
    assert AUDIT.asset_hashes_match(
        {path.name}, {path.name: path}, [{"name": path.name, "sha256": expected}],
    )
    assert not AUDIT.asset_hashes_match(
        {path.name}, {path.name: path}, [{"name": path.name, "sha256": "0" * 64}],
    )
