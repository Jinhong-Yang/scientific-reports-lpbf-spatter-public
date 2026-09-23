from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "evaluation"))


def load_module():
    spec = importlib.util.spec_from_file_location("run_locked_test_campaign", ROOT / "src" / "evaluation" / "run_locked_test_campaign.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RUNNER = load_module()


def test_w17_roster_is_exactly_w12_plus_w14_final_checkpoints() -> None:
    roster = RUNNER.test_roster()
    assert len(roster) == 208
    assert sum(row["source_block"] == "W12_core" for row in roster) == 100
    assert sum(row["source_block"] == "W14_low_data" for row in roster) == 108
    assert all(row["checkpoint_path"].endswith("milestone_8960.pt") for row in roster)


def test_w17_config_prohibits_test_selection() -> None:
    config = RUNNER.load_config()
    assert config["model_or_threshold_selection_from_test"] == "PROHIBITED"
    assert config["seed_replacement_from_test"] == "PROHIBITED"
    assert config["endpoint_expansion_from_test"] == "PROHIBITED"


def test_preflight_does_not_materialize_test_manifest() -> None:
    existed = RUNNER.TEST_MANIFEST_PATH.exists()
    checks = RUNNER.inspect_required_receipts()
    assert len(checks) == 14
    assert checks["W14_integrity"]["status"] in {"MISSING", "PASS", "FAIL"}
    assert checks["W16_integrity"]["status"] in {"MISSING", "PASS", "FAIL"}
    assert RUNNER.TEST_MANIFEST_PATH.exists() is existed


def test_preflight_receipt_inspection_rejects_existing_incomplete_receipt(
    tmp_path: Path, monkeypatch,
) -> None:
    complete = tmp_path / "complete.json"
    incomplete = tmp_path / "incomplete.json"
    complete.write_text(json.dumps({"status": "COMPLETE", "test_payload_accessed": False}), encoding="utf-8")
    incomplete.write_text(json.dumps({"status": "INCOMPLETE", "test_payload_accessed": False}), encoding="utf-8")
    monkeypatch.setattr(RUNNER, "required_receipts", lambda: {
        "complete": (complete, {"COMPLETE"}),
        "incomplete": (incomplete, {"COMPLETE"}),
        "missing": (tmp_path / "missing.json", {"PASS"}),
    })
    checks = RUNNER.inspect_required_receipts()
    assert checks["complete"]["accepted"] is True
    assert checks["incomplete"]["accepted"] is False
    assert checks["missing"]["status"] == "MISSING"


def test_test_access_state_follows_access_event_not_completed_count(
    tmp_path: Path, monkeypatch,
) -> None:
    monkeypatch.setattr(RUNNER, "EVIDENCE_ROOT", tmp_path)
    assert RUNNER.test_access_occurred() is False
    (RUNNER.EVIDENCE_ROOT / "TEST_ACCESS_EVENT.json").write_text("{}", encoding="utf-8")
    assert RUNNER.test_access_occurred() is True


def test_w17_gpu_accounting_includes_every_prior_gpu_stage(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(RUNNER, "ROOT", tmp_path)
    receipts = {
        "evidence/generator_factorial/W09_COMPLETENESS.json": {"cumulative_generator_gpu_hours": 1.0},
        "evidence/prior_sweep/W10_SWEEP_COMPLETENESS.json": {"cumulative_gpu_hours": 2.0},
        "evidence/stronger_generator/W11_DIFFUSION_COMPLETENESS.json": {"cumulative_gpu_hours": 3.0},
        "evidence/detector_core/CORE_COMPLETENESS.json": {"training_gpu_hours": 4.0, "validation_gpu_hours": 5.0},
        "evidence/resolution/W15_COMPLETENESS.json": {"evaluation_gpu_hours": 6.0},
        "evidence/low_data/W14_GENERATOR_COMPLETENESS.json": {"cumulative_gpu_hours": 7.0},
        "evidence/low_data/W14_DETECTOR_COMPLETENESS.json": {"training_gpu_hours": 8.0, "validation_gpu_hours": 9.0},
        "evidence/grouped_internal/W16_GENERATOR_COMPLETENESS.json": {"cumulative_gpu_hours": 10.0},
        "evidence/grouped_internal/W16_DETECTOR_COMPLETENESS.json": {"training_gpu_hours": 11.0, "evaluation_gpu_hours": 12.0},
    }
    for relative, value in receipts.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")
    monkeypatch.setattr(RUNNER, "summarize", lambda: {"evaluation_gpu_hours": 13.0})
    assert RUNNER.accounted_gpu_hours() == 91.0
