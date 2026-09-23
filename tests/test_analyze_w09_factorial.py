from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "src" / "analysis" / "analyze_w09_factorial.py"
SPEC = importlib.util.spec_from_file_location("analyze_w09_factorial", PATH)
assert SPEC and SPEC.loader
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)


def test_factorial_effects_known_values():
    values = {"F00": 10.0, "F01": 12.0, "F10": 13.0, "F11": 19.0}
    result = analysis.factorial_effects(values)
    assert result["proxy_main_effect"] == pytest.approx(5.0)
    assert result["boundary_moment_main_effect"] == pytest.approx(4.0)
    assert result["interaction"] == pytest.approx(4.0)


def test_factorial_effects_require_all_four_cells():
    with pytest.raises(ValueError):
        analysis.factorial_effects({"F00": 1.0, "F01": 2.0, "F10": 3.0})


def test_build_effect_rows_keeps_seed_pairing():
    rows = [
        {"seed": seed, "arm": arm, "best_validation_mse": value + seed}
        for seed in (1, 2)
        for arm, value in {"F00": 0.0, "F01": 1.0, "F10": 2.0, "F11": 4.0}.items()
    ]
    output = analysis.build_effect_rows(rows)
    assert [row["seed"] for row in output] == [1, 2]
    assert all(row["interaction"] == pytest.approx(1.0) for row in output)


def test_exact_sign_flip_resolution_for_constant_positive_vector():
    assert analysis.exact_sign_flip_p([1.0, 1.0, 1.0, 1.0]) == pytest.approx(2 / 16)
