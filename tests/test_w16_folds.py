from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_module():
    spec = importlib.util.spec_from_file_location("build_w16_folds", ROOT / "src" / "data" / "build_w16_folds.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RUNNER = load_module()


def test_w16_assignment_keeps_specimens_together_and_is_deterministic() -> None:
    rows = []
    for stratum in ("A", "B"):
        for specimen_index in range(5):
            for image_index in range(32):
                rows.append({"specimen": f"{stratum}{specimen_index}", "stratum": stratum, "sample_id": str(image_index)})
    first = RUNNER.specimen_assignment(rows, 64001, 3)
    second = RUNNER.specimen_assignment(rows, 64001, 3)
    assert first == second
    assert set(first.values()) == {1, 2, 3}
    assert len(first) == 10


def test_w16_config_has_approved_36_trajectory_crossing() -> None:
    config = RUNNER.load_config()
    assert len(config["partition_seeds"]) * config["folds_per_partition"] * len(config["arms"]) == 36
    assert config["held_out_test_access"].startswith("PROHIBITED")
