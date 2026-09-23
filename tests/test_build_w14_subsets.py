from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("build_w14_subsets", ROOT / "src" / "data" / "build_w14_subsets.py")
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_farthest_point_order_is_deterministic_nested_and_material_initialized() -> None:
    ids = [f"s{index:03d}" for index in range(20)]
    matrix = np.arange(140, dtype=np.float64).reshape(20, 7)
    materials = {name: "A" if index < 10 else "B" for index, name in enumerate(ids)}
    first = MODULE.nested_order(ids, matrix, materials, 63001)
    second = MODULE.nested_order(ids, matrix, materials, 63001)
    assert first == second
    assert sorted(first) == list(range(20))
    assert {materials[ids[index]] for index in first[:2]} == {"A", "B"}
    assert set(first[:5]) < set(first[:10]) < set(first[:15])


def test_specimen_matrix_aggregates_rows_without_outcomes() -> None:
    rows = []
    for specimen, material, power in (("s1", "A", 100), ("s2", "B", 200)):
        for index in range(2):
            rows.append({
                "specimen": specimen,
                "material": material,
                "laser_power_w": str(power),
                "scan_speed_mm_s": "800",
                "line_energy_j_mm": str(power / 800),
                "lamination_direction_deg": "45",
                "bbox_x_px": str(100 + index),
                "bbox_y_px": str(120 + index),
                "bbox_width_px": "20",
                "bbox_height_px": "10",
            })
    ids, matrix, materials = MODULE.specimen_matrix(rows)
    assert ids == ["s1", "s2"]
    assert matrix.shape == (2, 7)
    assert np.isfinite(matrix).all()
    assert materials == {"s1": "A", "s2": "B"}


def test_config_crossing_is_exactly_108() -> None:
    config = MODULE.load_config()
    assert len(config["subset_sizes_specimens"]) * len(config["subset_realizations"]) * len(config["arms"]) * len(config["pipeline_replicates"]) == 108
    assert config["synthetic_only_branch"].startswith("NOT_INCLUDED")
