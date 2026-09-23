from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "generators"))


def load_module():
    spec = importlib.util.spec_from_file_location("run_w16_grouped_generators", ROOT / "src" / "generators" / "run_w16_grouped_generators.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RUNNER = load_module()


def test_w16_generator_roster_is_nine_fold_specific_refits() -> None:
    roster = RUNNER.roster()
    assert len(roster) == 9
    assert len({row["run_id"] for row in roster}) == 9
    assert {row["partition_seed"] for row in roster} == {64001, 64002, 64003}
    assert {row["fold"] for row in roster} == {1, 2, 3}
    assert all(row["checkpoint_policy"] == "fixed_final_epoch_10_no_oof_selection" for row in roster)
    assert all(RUNNER.subset_path_for(row).is_file() for row in roster)


def test_w16_generator_preflight_waits_without_test_access() -> None:
    result = RUNNER.preflight()
    assert result["status"] in {"WAITING_FOR_PREREQUISITES", "PASS"}
    assert result["test_payload_accessed"] is False
    assert result["checks"]["test_locked"] is True


def test_generator_gpu_accounting_includes_all_prior_stages(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(RUNNER, "ROOT", tmp_path)
    receipts = {
        "evidence/generator_factorial/W09_COMPLETENESS.json": {"cumulative_generator_gpu_hours": 1.0},
        "evidence/prior_sweep/W10_SWEEP_COMPLETENESS.json": {"cumulative_gpu_hours": 2.0},
        "evidence/stronger_generator/W11_DIFFUSION_COMPLETENESS.json": {"cumulative_gpu_hours": 3.0},
        "evidence/detector_core/CORE_COMPLETENESS.json": {"training_gpu_hours": 4.0, "validation_gpu_hours": 5.0},
        "evidence/resolution/W15_COMPLETENESS.json": {"evaluation_gpu_hours": 6.0},
        "evidence/low_data/W14_GENERATOR_COMPLETENESS.json": {"cumulative_gpu_hours": 7.0},
        "evidence/low_data/W14_DETECTOR_COMPLETENESS.json": {"training_gpu_hours": 8.0, "validation_gpu_hours": 9.0},
    }
    for relative, value in receipts.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")
    monkeypatch.setattr(RUNNER, "summarize", lambda: {"cumulative_gpu_hours": 10.0})
    assert RUNNER.accounted_gpu_hours() == 55.0
