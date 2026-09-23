from __future__ import annotations

import importlib.util
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_recovery():
    detector_root = ROOT / "src" / "detectors"
    import sys
    sys.path.insert(0, str(detector_root))
    spec = importlib.util.spec_from_file_location("w12_retinanet_recovery", detector_root / "run_w12_retinanet_recovery.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_recovery_drops_only_invalid_coco_detections() -> None:
    recovery = load_recovery()
    records = [{
        "boxes": [[1, 2, 3, 4], [math.nan, 2, 3, 4], [4, 4, 4, 5]],
        "scores": [0.9, 0.8, 0.7],
        "labels": [1, 1, 1],
    }]
    cleaned = recovery.sanitize_records(records)
    assert cleaned[0]["boxes"] == [[1.0, 2.0, 3.0, 4.0]]
    assert cleaned[0]["scores"] == [0.9]
    assert cleaned[0]["labels"] == [1]
    assert cleaned[0]["prediction_sanitization"]["candidate_detections"] == 3
    assert cleaned[0]["prediction_sanitization"]["invalid_detections_discarded"] == 2
