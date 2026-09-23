from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_w08_draft_is_retained_as_preapproval_chronology():
    draft = yaml.safe_load((ROOT / "configs" / "protocol_draft.yaml").read_text(encoding="utf-8"))
    assert draft["status"] == "DRAFT_NOT_FROZEN"
    assert draft["test_outcomes_used_to_design_protocol"] is False
    assert draft["data_roles"]["test_access_before_final_evaluation"] == "prohibited"
    assert draft["label_policy"]["two_independent_rater_roles"] is None
    assert draft["detector"]["precision_policy_approved"] is None
    assert draft["primary_analysis"]["endpoint_approved"] is None
    assert draft["trajectory_budget"]["tier_approved"] is None


def test_w08_protocol_lock_is_hashed_and_outcome_blind():
    lock_path = ROOT / "configs" / "PROTOCOL_LOCK.yaml"
    receipt = (ROOT / "configs" / "PROTOCOL_LOCK.sha256").read_text(encoding="utf-8").strip()
    expected_hash, filename = receipt.split(maxsplit=1)
    assert filename == "PROTOCOL_LOCK.yaml"
    assert hashlib.sha256(lock_path.read_bytes()).hexdigest() == expected_hash

    lock = yaml.safe_load(lock_path.read_text(encoding="utf-8"))
    assert lock["status"] == "FROZEN_PRE_N2_OUTCOME"
    assert lock["test_outcomes_used_to_design_protocol"] is False
    assert lock["scope"]["human_label_review"] == "EXCLUDED_BY_USER_SCOPE"
    assert lock["scope"]["evaluation_route"] == "B_GROUPED_INTERNAL"
    assert lock["execution_gates"]["full_training_authorized"] is True
    assert lock["execution_gates"]["test_data_access_authorized_now"] is False
    assert lock["approved_trajectory_matrix"]["detector_total"] == 244
    assert lock["approved_trajectory_matrix"]["label_policy_interaction"] == 0
    assert len(lock["seed_rosters"]["primary_full_pipeline"]) == 10


def test_protocol_lock_frozen_input_hashes_match_files():
    lock = yaml.safe_load((ROOT / "configs" / "PROTOCOL_LOCK.yaml").read_text(encoding="utf-8"))
    for item in lock["frozen_inputs"].values():
        path = ROOT / item["path"]
        assert path.is_file(), item["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"], item["path"]

    hpo = json.loads((ROOT / "configs" / "HPO_BUDGET.json").read_text(encoding="utf-8"))
    assert hpo["status"] == "W08_APPROVED_PRE_TEST"
    assert hpo["unresolved_before_w08"] == []
    assert hpo["compact_diffusion"]["full_fit_updates"] == 100000
