import importlib.util
from pathlib import Path
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "generators"))
SPEC = importlib.util.spec_from_file_location("build_w11_pools", ROOT / "src" / "generators" / "build_w11_pools.py")
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_ddim_schedule_is_exactly_fifty_descending_steps():
    values = MODULE.ddim_timesteps()
    assert len(values) == 50
    assert values[0] == 999 and values[-1] == 0
    assert np.all(np.diff(values) < 0)


def test_donor_plan_is_repeatable_and_same_domain():
    rows = [
        {"material": "A", "view": "x"}, {"material": "A", "view": "x"},
        {"material": "B", "view": "y"}, {"material": "B", "view": "y"},
    ]
    conditions = np.zeros((4, 13), dtype=np.float64)
    first = MODULE.donor_plan(rows, conditions, 7)
    second = MODULE.donor_plan(rows, conditions, 7)
    for key in first:
        assert np.array_equal(first[key], second[key])
    for index in range(4):
        assert rows[first["donor_a"][index]]["material"] == rows[index]["material"]
        assert rows[first["donor_b"][index]]["view"] == rows[index]["view"]


def test_blend_pool_shape_and_dtype():
    images = torch.tensor([[[[0.0]]], [[[1.0]]]])
    plan = {"donor_a": np.array([0, 1]), "donor_b": np.array([1, 0]),
            "alpha": np.array([0.5, 0.5], dtype=np.float32)}
    output = MODULE.blend_pool(images, plan)
    assert output.shape == (2, 1, 1, 1)
    assert output.dtype == np.uint8
    assert output[:, 0, 0, 0].tolist() == [128, 128]
