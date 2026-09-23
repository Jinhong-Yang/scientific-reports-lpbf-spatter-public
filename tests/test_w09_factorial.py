from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "src" / "generators" / "run_w09_factorial.py"
SPEC = importlib.util.spec_from_file_location("run_w09_factorial", MODULE_PATH)
assert SPEC and SPEC.loader
w09 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(w09)


def test_w09_roster_is_exactly_frozen_2x2_by_10():
    rows = w09.roster()
    assert len(rows) == 40
    assert {arm for arm, _ in rows} == {"F00", "F01", "F10", "F11"}
    assert {seed for _, seed in rows} == set(range(61001, 61011))


def test_w09_paired_streams_match_across_arms_and_separate_phases():
    assert w09.stream_seed(61001, "posterior", 3) == w09.stream_seed(61001, "posterior", 3)
    assert w09.stream_seed(61001, "posterior", 3) != w09.stream_seed(61001, "pixels", 3)
    assert w09.stream_seed(61001, "posterior", 3) != w09.stream_seed(61002, "posterior", 3)


def test_w09_regularizer_fixture_is_finite_and_differentiable():
    config = w09.ModelConfig(condition_dim=13, latent_dim=4, hidden_dim=16, residual_blocks=1)
    model = w09.ConditionalVariationalField(config, 16)
    condition = torch.zeros(2, 13)
    condition[:, 9] = 0.2
    latent = torch.zeros(2, 4)
    coordinates = torch.rand(2, 5, 2) * 2 - 1
    boundary = torch.rand(2, 4, 2) * 2 - 1
    moment_grid = w09.make_coordinate_grid(4)[None].expand(2, -1, -1)
    terms = w09.regularizer_components(model.decoder, condition, latent, coordinates, boundary, moment_grid)
    assert set(terms) == {"proxy", "boundary", "moment"}
    assert all(torch.isfinite(value) for value in terms.values())
    sum(terms.values()).backward()
    assert any(parameter.grad is not None for parameter in model.decoder.parameters())


def test_w09_protocol_receipt_and_frozen_inputs_match():
    lock, protocol_hash = w09.verify_protocol()
    expected = hashlib.sha256((ROOT / "configs" / "PROTOCOL_LOCK.yaml").read_bytes()).hexdigest()
    assert protocol_hash == expected
    assert lock["execution_gates"]["test_data_access_authorized_now"] is False
    assert lock["scope"]["human_label_review"] == "EXCLUDED_BY_USER_SCOPE"


def test_w09_prepared_inputs_contain_no_test_rows_when_present():
    receipt_path = ROOT / "data" / "manifests" / "w09" / "input_receipt.json"
    if not receipt_path.exists():
        return
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["test_rows_written"] == 0
    for split in ("train", "validation"):
        rows = w09.read_csv(ROOT / receipt["outputs"][split]["path"])
        assert rows and all(row["split"] == split for row in rows)
        assert all(row["split"] != "test" for row in rows)


def test_w09_config_is_referenced_by_protocol_lock():
    lock = yaml.safe_load((ROOT / "configs" / "PROTOCOL_LOCK.yaml").read_text(encoding="utf-8"))
    item = lock["frozen_inputs"]["generator_factorial"]
    assert item["path"] == "configs/generator_factorial.json"
    assert hashlib.sha256((ROOT / item["path"]).read_bytes()).hexdigest() == item["sha256"]

