import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("secondary_reporting", ROOT / "src/reporting/summarize_secondary_recovery.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_unsanitized_completed_cell_is_preserved():
    assert MODULE.summarize_records([{"boxes": [[0, 0, 1, 1]]}])["adapter_annotated_images"] == 0


def test_recovery_counts_must_reconcile():
    row = {"boxes": [[0, 0, 1, 1]], "prediction_sanitization": {
        "candidate_detections": 3, "invalid_detections_discarded": 2}}
    result = MODULE.summarize_records([row])
    assert result["candidate_detections"] == result["retained_detections"] + result["invalid_detections_discarded"]
    row["prediction_sanitization"]["candidate_detections"] = 4
    with pytest.raises(ValueError):
        MODULE.summarize_records([row])
