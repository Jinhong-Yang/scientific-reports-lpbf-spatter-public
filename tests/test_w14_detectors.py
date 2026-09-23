from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "detectors"))


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RUNNER = load("run_w14_lowdata_detectors", ROOT / "src" / "detectors" / "run_w14_lowdata_detectors.py")


def test_frozen_w14_roster_has_exact_crossing() -> None:
    roster = RUNNER.trajectory_roster()
    assert len(roster) == 108
    assert len({row["run_id"] for row in roster}) == 108
    assert {row["subset_size"] for row in roster} == {14, 28, 56}
    assert {row["realization"] for row in roster} == {"R1", "R2", "R3"}
    assert {row["arm"] for row in roster} == {"N1", "NR", "NB", "NP"}
    assert {row["pipeline_seed"] for row in roster} == {63001, 63002, 63003}


def test_w14_real_rosters_are_subset_only_and_ratio_exact() -> None:
    n1 = RUNNER.select_spec("R1", 14, "N1", 63001)
    nr = RUNNER.select_spec("R1", 14, "NR", 63001)
    n1_rows, _ = RUNNER.load_training_rows(n1, RUNNER.detector_config())
    nr_rows, _ = RUNNER.load_training_rows(nr, RUNNER.detector_config())
    assert len(n1_rows) == 448
    assert len(nr_rows) == 560
    assert len({row["specimen"] for row in n1_rows}) == 14
    assert sum(row["source_kind"] == "real" for row in nr_rows) == 448
    assert sum(row["source_kind"] == "additional_real" for row in nr_rows) == 112
    assert all(row["split"] == "train" for row in nr_rows)


def test_all_low_data_rosters_complete_integer_exposure_cycles() -> None:
    total_draws = RUNNER.detector_config()["optimization"]["optimizer_updates"] * RUNNER.detector_config()["optimization"]["micro_batch"]
    for spec in RUNNER.trajectory_roster():
        expected_rows = spec["base_rows"] if spec["arm"] == "N1" else spec["base_rows"] + spec["additional_rows"]
        assert total_draws % expected_rows == 0


def test_preflight_waits_without_test_access() -> None:
    result = RUNNER.preflight()
    assert result["status"] in {"WAITING_FOR_PREREQUISITES", "PASS"}
    assert result["test_payload_accessed"] is False
    assert result["checks"]["test_locked"] is True


def test_gpu_accounting_includes_w09_generator_field(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(RUNNER, "ROOT", tmp_path)
    monkeypatch.setattr(RUNNER, "EVIDENCE_ROOT", tmp_path / "evidence/low_data")
    receipts = {
        "evidence/generator_factorial/W09_COMPLETENESS.json": {"cumulative_generator_gpu_hours": 1.0},
        "evidence/prior_sweep/W10_SWEEP_COMPLETENESS.json": {"cumulative_gpu_hours": 2.0},
        "evidence/stronger_generator/W11_DIFFUSION_COMPLETENESS.json": {"cumulative_gpu_hours": 3.0},
        "evidence/detector_core/CORE_COMPLETENESS.json": {"training_gpu_hours": 4.0, "validation_gpu_hours": 5.0},
        "evidence/resolution/W15_COMPLETENESS.json": {"evaluation_gpu_hours": 0.5},
        "evidence/low_data/W14_GENERATOR_COMPLETENESS.json": {"cumulative_gpu_hours": 6.0},
    }
    for relative, value in receipts.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")
    monkeypatch.setattr(RUNNER, "summarize", lambda: {"training_gpu_hours": 7.0, "validation_gpu_hours": 8.0})
    assert RUNNER.accounted_gpu_hours() == 36.5
