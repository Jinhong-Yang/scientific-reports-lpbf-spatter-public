from __future__ import annotations

import csv
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_json(relative: str):
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def test_w06_success_receipt_is_bounded_and_complete():
    summary = read_json("evidence/pilot/pilot_summary.json")
    assert summary["status"] == "PASS_WITH_LIMITATION"
    assert summary["updates"] == 200
    assert summary["test_split_used"] is False
    assert summary["amp_mode"] == "bf16"
    assert summary["training_rows"] == summary["validation_rows"] == 64
    assert summary["validation"]["milestones"] == [0, 50, 100, 150, 200]
    assert all(summary["acceptance"].values())
    assert summary["loss"]["all_finite"] is True


def test_w06_csv_has_all_training_and_validation_rows():
    with (ROOT / "evidence/pilot/pilot_results.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    training = [row for row in rows if row["row_type"] == "training"]
    validation = [row for row in rows if row["row_type"] == "validation"]
    assert [int(row["update"]) for row in training] == list(range(1, 201))
    assert [int(row["update"]) for row in validation] == [0, 50, 100, 150, 200]
    assert all(math.isfinite(float(row["loss_total"])) for row in training)


def test_w06_fp16_failure_is_preserved_and_w08_choices_are_frozen():
    failure = read_json("evidence/pilot/failure_amp_fp16_update1.json")
    budget = read_json("configs/HPO_BUDGET.json")
    assert failure["status"] == "STOPPED_AS_DESIGNED"
    assert failure["update"] == 1
    assert failure["gradient_norm_before_clip"] == "inf"
    assert failure["test_split_used"] is False
    assert budget["status"] == "W08_APPROVED_PRE_TEST"
    assert budget["non_pde_controls"]["numeric_values"]["blob_gaussian_sigma_px"] == [0, 0.5, 1.0, 2.0]
    assert budget["non_pde_controls"]["reconstruction_match_tolerance_relative_mse"] == 0.05
    assert budget["selection_rule"]
    assert "8960" in budget["stopping_rule"]
    assert budget["unresolved_before_w08"] == []
