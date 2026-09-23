from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "detectors"))


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


COMMON = load("detector_common", ROOT / "src" / "detectors" / "detector_common.py")
RUNNER = load("run_w12_detectors", ROOT / "src" / "detectors" / "run_w12_detectors.py")


def test_frozen_roster_has_exact_primary_and_secondary_counts() -> None:
    config = RUNNER.load_config()
    roster = RUNNER.trajectory_roster(config)
    assert len(roster) == 100
    assert len({row["run_id"] for row in roster}) == 100
    assert sum(row["block"] == "primary" for row in roster) == 80
    assert sum(row["block"] == "secondary" for row in roster) == 20
    assert {row["pool_seed"] for row in roster if row["block"] == "secondary"} == set(range(61001, 61006))


def test_shuffle_stream_resume_and_complete_epoch_exposures() -> None:
    first = COMMON.ShuffledStream(4480, 17)
    head = first.take(137)
    state = first.state_dict()
    tail = first.take(401)
    second = COMMON.ShuffledStream(4480, 999)
    second.load_state_dict(state)
    assert second.take(401) == tail
    assert len(set(head)) == len(head)

    stream = COMMON.ShuffledStream(4480, 23)
    counts = np.zeros(4480, dtype=np.int64)
    for _ in range(8960):
        for index in stream.take(4):
            counts[index] += 1
    assert set(counts.tolist()) == {8}
    assert int(counts[:3584].sum()) == 28672
    assert int(counts[3584:].sum()) == 7168


def test_dataset_reads_native_real_and_upsampled_synthetic(tmp_path: Path) -> None:
    real_path = tmp_path / "real.bmp"
    real = np.zeros((300, 300), dtype=np.uint8)
    real[120:180, 130:170] = 255
    Image.fromarray(real).save(real_path)
    synthetic_path = tmp_path / "pool.npy"
    synthetic = np.zeros((1, 1, 64, 64), dtype=np.uint8)
    synthetic[0, 0, 20:40, 22:42] = 200
    np.save(synthetic_path, synthetic, allow_pickle=False)
    base = {
        "bbox_x_px": "120",
        "bbox_y_px": "110",
        "bbox_width_px": "60",
        "bbox_height_px": "70",
        "specimen": "S1",
        "view": "V1",
    }
    rows = [
        {**base, "sample_id": "real", "target_sample_id": "real", "source_kind": "real", "image_path": "real.bmp"},
        {**base, "sample_id": "synthetic", "target_sample_id": "real", "source_kind": "synthetic", "array_path": "pool.npy", "array_index": 0},
    ]
    dataset = COMMON.DetectorRows(rows, "fasterrcnn", tmp_path, augment_real=False)
    real_tensor, real_target, _ = dataset[0]
    synthetic_tensor, synthetic_target, _ = dataset[1]
    assert real_tensor.shape == synthetic_tensor.shape == (3, 300, 300)
    assert real_target["labels"].tolist() == [1]
    assert synthetic_target["boxes"].tolist() == [[120.0, 110.0, 180.0, 180.0]]


def test_official_coco_oracle_perfect_and_missed() -> None:
    record = {
        "sample_id": "x",
        "specimen": "s",
        "view": "v",
        "source_kind": "real",
        "target_sample_id": "x",
        "gt_boxes": [[10, 10, 50, 50]],
        "boxes": [[10, 10, 50, 50]],
        "scores": [0.99],
        "labels": [1],
    }
    perfect = COMMON.official_coco_metrics([record])
    missed = COMMON.official_coco_metrics([{**record, "boxes": [], "scores": [], "labels": []}])
    assert perfect["COCO_AP"] > 0.99
    assert missed["COCO_AP"] == 0.0


def test_preflight_waits_without_touching_test() -> None:
    result = RUNNER.preflight()
    assert result["status"] in {"WAITING_FOR_W11", "PASS"}
    assert result["test_payload_accessed"] is False
    assert result["checks"]["test_locked"] is True


def test_gpu_accounting_includes_w09_generator_field(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(RUNNER, "ROOT", tmp_path)
    receipts = {
        "evidence/generator_factorial/W09_COMPLETENESS.json": {"cumulative_generator_gpu_hours": 1.0},
        "evidence/prior_sweep/W10_SWEEP_COMPLETENESS.json": {"cumulative_gpu_hours": 2.0},
        "evidence/stronger_generator/W11_DIFFUSION_COMPLETENESS.json": {"cumulative_gpu_hours": 3.0},
    }
    for relative, value in receipts.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")
    monkeypatch.setattr(RUNNER, "summarize", lambda: {"training_gpu_hours": 4.0, "validation_gpu_hours": 5.0})
    assert RUNNER.completed_project_gpu_hours() == 15.0
