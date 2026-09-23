from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "detectors"))


def load_module():
    spec = importlib.util.spec_from_file_location("run_w16_grouped_detectors", ROOT / "src" / "detectors" / "run_w16_grouped_detectors.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RUNNER = load_module()


def test_w16_detector_roster_has_exact_36_crossing() -> None:
    roster = RUNNER.trajectory_roster()
    assert len(roster) == 36
    assert len({row["run_id"] for row in roster}) == 36
    assert {row["partition_seed"] for row in roster} == {64001, 64002, 64003}
    assert {row["fold"] for row in roster} == {1, 2, 3}
    assert {row["arm"] for row in roster} == {"N1", "NR", "NB", "NP"}


def test_w16_n1_nr_rows_are_outer_training_only() -> None:
    n1 = RUNNER.select_spec(64001, 1, "N1")
    nr = RUNNER.select_spec(64001, 1, "NR")
    n1_rows, _ = RUNNER.load_training_rows(n1, RUNNER.detector_config())
    nr_rows, _ = RUNNER.load_training_rows(nr, RUNNER.detector_config())
    assert len(n1_rows) == n1["base_rows"]
    assert len(nr_rows) == nr["base_rows"] + nr["additional_rows"]
    assert all(row["split"] == "train" for row in nr_rows)
    oof_specimens = {row["specimen"] for row in RUNNER.read_csv(RUNNER.fold_path(n1, "oof"))}
    assert not ({row["specimen"] for row in nr_rows} & oof_specimens)


def test_w16_oof_sets_partition_development_specimens() -> None:
    for seed in (64001, 64002, 64003):
        folds = []
        for fold in (1, 2, 3):
            spec = RUNNER.select_spec(seed, fold, "N1")
            folds.append({row["specimen"] for row in RUNNER.read_csv(RUNNER.fold_path(spec, "oof"))})
        assert len(set.union(*folds)) == 144
        assert all(not left & right for index, left in enumerate(folds) for right in folds[index + 1:])


def test_w16_preflight_waits_without_test_access() -> None:
    result = RUNNER.preflight()
    assert result["status"] in {"WAITING_FOR_PREREQUISITES", "PASS"}
    assert result["test_payload_accessed"] is False
    assert result["checks"]["test_locked"] is True


def test_gpu_accounting_includes_w09_generator_field(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(RUNNER, "ROOT", tmp_path)
    monkeypatch.setattr(RUNNER, "EVIDENCE_ROOT", tmp_path / "evidence/grouped_internal")
    receipts = {
        "evidence/generator_factorial/W09_COMPLETENESS.json": {"cumulative_generator_gpu_hours": 1.0},
        "evidence/prior_sweep/W10_SWEEP_COMPLETENESS.json": {"cumulative_gpu_hours": 2.0},
        "evidence/stronger_generator/W11_DIFFUSION_COMPLETENESS.json": {"cumulative_gpu_hours": 3.0},
        "evidence/detector_core/CORE_COMPLETENESS.json": {"training_gpu_hours": 4.0, "validation_gpu_hours": 5.0},
        "evidence/resolution/W15_COMPLETENESS.json": {"evaluation_gpu_hours": 0.5},
        "evidence/low_data/W14_GENERATOR_COMPLETENESS.json": {"cumulative_gpu_hours": 6.0},
        "evidence/low_data/W14_DETECTOR_COMPLETENESS.json": {"training_gpu_hours": 7.0, "validation_gpu_hours": 8.0},
        "evidence/grouped_internal/W16_GENERATOR_COMPLETENESS.json": {"cumulative_gpu_hours": 9.0},
    }
    for relative, value in receipts.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")
    monkeypatch.setattr(RUNNER, "summarize", lambda: {"training_gpu_hours": 10.0, "evaluation_gpu_hours": 11.0})
    assert RUNNER.accounted_gpu_hours() == 66.5
