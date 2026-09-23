from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "analysis"))

from prevalidate_statistics import (  # noqa: E402
    bootstrap_mean_ci,
    exact_sign_flip,
    factorial_effects,
    holm_adjust,
    planning_grid,
    paired_cluster_coco_bootstrap,
)


def test_factorial_formulas():
    result = factorial_effects({"F00": 0, "F01": 2, "F10": 1, "F11": 5})
    assert result == {"optical_proxy_main": 2, "boundary_moment_main": 3, "interaction": 2}


def test_five_seed_sign_flip_resolution_and_zero_case():
    assert exact_sign_flip(np.ones(5))["two_sided_p"] == pytest.approx(0.0625)
    assert exact_sign_flip(np.zeros(5))["two_sided_p"] == 1.0


def test_holm_preserves_undefined_and_monotonicity():
    result = holm_adjust({"a": 0.01, "b": 0.03, "c": 0.02, "d": None})
    assert result == {"a": pytest.approx(0.03), "b": pytest.approx(0.04), "c": pytest.approx(0.04), "d": None}
    with pytest.raises(ValueError):
        holm_adjust({"a": 0.0})


def test_bootstrap_is_paired_deterministic_and_contains_constant_effect():
    matrix = np.full((8, 3), 0.02)
    first = bootstrap_mean_ci(matrix, seed=1, draws=200, confidence=0.95, resample_replicates=True)
    second = bootstrap_mean_ci(matrix, seed=1, draws=200, confidence=0.95, resample_replicates=True)
    assert first == second
    assert first["low"] == pytest.approx(0.02)
    assert first["high"] == pytest.approx(0.02)


def test_planning_grid_is_complete_and_explicitly_synthetic():
    grid = planning_grid()
    assert len(grid) == 36
    assert set(grid["evidence_layer"]) == {"SYNTHETIC_FIXTURE_ONLY"}
    assert not grid["pilot_calibrated"].any()
    assert set(grid["family_comparisons"]) == {4}


def test_paired_coco_bootstrap_duplicates_ids_safely_and_rejects_unpaired_cells():
    rows = []
    for replicate in (1, 2):
        for cluster in ("a", "b"):
            rows.append({
                "cluster_id": cluster,
                "replicate": replicate,
                "item_id": cluster,
                "gt_boxes": [[10, 10, 30, 30]],
                "boxes": [[10, 10, 30, 30]],
                "scores": [0.9],
                "labels": [1],
                "width": 300,
                "height": 300,
            })
    result = paired_cluster_coco_bootstrap(rows, [dict(row) for row in rows], draws=10, seed=8)
    assert result["mean_difference"] == 0
    assert result["new_evaluation_image_ids_assigned_by_coco_wrapper"] is True
    with pytest.raises(ValueError, match="identical"):
        paired_cluster_coco_bootstrap(rows, rows[:-1], draws=10, seed=8)
