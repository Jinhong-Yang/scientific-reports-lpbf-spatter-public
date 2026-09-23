from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("run_w14_lowdata_generators", ROOT / "src" / "generators" / "run_w14_lowdata_generators.py")
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_generator_roster_has_27_crossed_training_only_refits() -> None:
    roster = MODULE.roster()
    assert len(roster) == 27
    assert len({row["run_id"] for row in roster}) == 27
    assert {row["realization"] for row in roster} == {"R1", "R2", "R3"}
    assert {row["subset_size"] for row in roster} == {14, 28, 56}
    assert {row["pipeline_seed"] for row in roster} == {63001, 63002, 63003}
    assert len({row["generator_seed"] for row in roster}) == 27


def test_generator_seed_is_deterministic_and_context_specific() -> None:
    first = MODULE.generator_seed("R1", 14, 63001)
    assert first == MODULE.generator_seed("R1", 14, 63001)
    assert first != MODULE.generator_seed("R2", 14, 63001)
    assert first != MODULE.generator_seed("R1", 28, 63001)
    assert first != MODULE.generator_seed("R1", 14, 63002)


def test_preflight_passes_without_test_access() -> None:
    config = MODULE.load_config()
    assert config["held_out_test_access"].startswith("PROHIBITED")
    assert config["generator_refits"]["NP"] == 27


def test_generator_gpu_accounting_includes_all_prior_stages(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(MODULE, "ROOT", tmp_path)
    receipts = {
        "evidence/generator_factorial/W09_COMPLETENESS.json": {"cumulative_generator_gpu_hours": 1.0},
        "evidence/prior_sweep/W10_SWEEP_COMPLETENESS.json": {"cumulative_gpu_hours": 2.0},
        "evidence/stronger_generator/W11_DIFFUSION_COMPLETENESS.json": {"cumulative_gpu_hours": 3.0},
        "evidence/detector_core/CORE_COMPLETENESS.json": {"training_gpu_hours": 4.0, "validation_gpu_hours": 5.0},
        "evidence/resolution/W15_COMPLETENESS.json": {"evaluation_gpu_hours": 6.0},
    }
    for relative, value in receipts.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")
    monkeypatch.setattr(MODULE, "summarize", lambda: {"cumulative_gpu_hours": 7.0})
    assert MODULE.accounted_gpu_hours() == 28.0
