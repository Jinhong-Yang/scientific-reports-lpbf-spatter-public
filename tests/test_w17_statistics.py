from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "analysis"))


def load_module():
    spec = importlib.util.spec_from_file_location("run_w17_statistics", ROOT / "src" / "analysis" / "run_w17_statistics.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RUNNER = load_module()


def record(specimen: str, box, scores, boxes):
    return {
        "sample_id": specimen,
        "specimen": specimen,
        "view": "v",
        "source_kind": "real",
        "target_sample_id": specimen,
        "gt_boxes": [box],
        "boxes": boxes,
        "scores": scores,
        "labels": [1] * len(scores),
    }


def test_fast_one_gt_coco_matches_official_fixture() -> None:
    records = [
        record("s1", [10, 10, 50, 50], [0.9, 0.2], [[10, 10, 50, 50], [100, 100, 120, 120]]),
        record("s2", [20, 20, 40, 40], [0.8], [[100, 100, 130, 130]]),
        record("s3", [5, 5, 15, 15], [], []),
    ]
    order, blocks = RUNNER.prepare_specimen_blocks(records)
    fast = RUNNER.fast_coco_ap(order, blocks, np.arange(len(order)))
    official = RUNNER.official_coco_metrics(records)["COCO_AP"]
    assert abs(fast - official) < 1e-12


def test_fast_coco_handles_duplicate_specimen_draws() -> None:
    records = [
        record("s1", [10, 10, 50, 50], [0.9], [[10, 10, 50, 50]]),
        record("s2", [20, 20, 40, 40], [], []),
    ]
    order, blocks = RUNNER.prepare_specimen_blocks(records)
    assert RUNNER.fast_coco_ap(order, blocks, np.asarray([0, 0])) > 0.99
    assert RUNNER.fast_coco_ap(order, blocks, np.asarray([1, 1])) == 0.0


def test_statistics_config_is_frozen_primary_family() -> None:
    config = RUNNER.load_config()
    assert config["primary_contrasts"] == ["NP-NB", "NP-NF", "NP-N1", "NP-NR"]
    assert config["bootstrap_draws"] == 20000
    assert config["simultaneous_individual_interval"] == 0.9875


def test_parallel_cell_decomposition_matches_direct_nested_resampling() -> None:
    records_a = [
        record("s1", [10, 10, 50, 50], [0.9], [[10, 10, 50, 50]]),
        record("s2", [20, 20, 40, 40], [], []),
    ]
    records_b = [
        record("s1", [10, 10, 50, 50], [0.9], [[100, 100, 130, 130]]),
        record("s2", [20, 20, 40, 40], [0.8], [[20, 20, 40, 40]]),
    ]
    order_a, blocks_a = RUNNER.prepare_specimen_blocks(records_a)
    order_b, blocks_b = RUNNER.prepare_specimen_blocks(records_b)
    specimen_draws = np.asarray([[0, 1], [0, 0], [1, 1]])
    replicate_draws = np.asarray([[0, 1], [0, 0], [1, 1]])
    cells = {
        "NP": {1: RUNNER.bootstrap_cell_values(order_a, blocks_a, specimen_draws),
               2: RUNNER.bootstrap_cell_values(order_b, blocks_b, specimen_draws)},
        "NB": {1: RUNNER.bootstrap_cell_values(order_b, blocks_b, specimen_draws),
               2: RUNNER.bootstrap_cell_values(order_a, blocks_a, specimen_draws)},
    }
    combined = RUNNER.combine_pipeline_bootstrap(cells, [1, 2], replicate_draws, ["NP-NB"])
    direct = []
    for draw_index, specimen_indices in enumerate(specimen_draws):
        selected = replicate_draws[draw_index]
        np_mean = np.mean([
            RUNNER.fast_coco_ap(order_a if index == 0 else order_b,
                                blocks_a if index == 0 else blocks_b, specimen_indices)
            for index in selected
        ])
        nb_mean = np.mean([
            RUNNER.fast_coco_ap(order_b if index == 0 else order_a,
                                blocks_b if index == 0 else blocks_a, specimen_indices)
            for index in selected
        ])
        direct.append(np_mean - nb_mean)
    assert np.allclose(combined[:, 0], direct)
