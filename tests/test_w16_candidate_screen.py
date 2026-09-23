from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ALLOWED = {
    "SAME_TASK_EXTERNAL", "NEW_TASK_REPLICATION", "REPRESENTATION_ONLY",
    "INCOMPATIBLE", "ACCESS_BLOCKED",
}


def test_w16_candidate_screen_is_complete_and_does_not_overclaim():
    with (ROOT / "docs" / "DATASET_COMPATIBILITY.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    receipt = json.loads(
        (ROOT / "evidence" / "external" / "W16_CANDIDATE_SCREEN.json").read_text(encoding="utf-8")
    )
    assert len(rows) == receipt["candidates"] == 7
    assert {row["classification"] for row in rows} <= ALLOWED
    assert all(row["direct_primary_use"] == "NO" for row in rows)
    assert all(row["official_url"].startswith("https://") for row in rows)
    counts = {label: sum(row["classification"] == label for row in rows) for label in ALLOWED}
    assert counts == receipt["classification_counts"]
    assert receipt["direct_primary_endpoint_candidates"] == 0
    assert receipt["downloaded_or_accessed_raw_candidate_data"] is False
    assert receipt["status"] == "NO_VERIFIED_SAME_TASK_EXTERNAL_COHORT"
