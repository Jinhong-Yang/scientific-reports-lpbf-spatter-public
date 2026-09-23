from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "pilot"))

from run_w06_diffusion_pilot import CompactConditionalUNet, cosine_schedule  # noqa: E402
from validate_diffusion_sampler import ddim_sample  # noqa: E402


def test_compact_conditional_unet_shape_and_schedule():
    model = CompactConditionalUNet(base=16, embedding_dim=32)
    output = model(torch.zeros(2, 1, 64, 64), torch.tensor([0, 999]), torch.zeros(2, 13))
    betas = cosine_schedule(1000)
    assert output.shape == (2, 1, 64, 64)
    assert betas.shape == (1000,)
    assert torch.isfinite(output).all()
    assert torch.all((betas > 0) & (betas < 1))


def test_diffusion_pilot_evidence_is_bounded_and_complete():
    summary = json.loads(
        (ROOT / "evidence" / "pilot" / "diffusion_pilot_summary.json").read_text(encoding="utf-8")
    )
    assert summary["status"] == "PASS_WITH_LIMITATION"
    assert summary["updates"] == 200
    assert summary["test_split_used"] is False
    assert summary["external_pretraining"] is False
    assert summary["resolution"] == [64, 64]
    assert summary["training_rows"] == summary["validation_rows"] == 64
    assert summary["validation"]["milestones"] == [0, 50, 100, 150, 200]
    assert all(summary["acceptance"].values())

    with (ROOT / "evidence" / "pilot" / "diffusion_pilot_results.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert sum(row["row_type"] == "training" for row in rows) == 200
    assert sum(row["row_type"] == "validation" for row in rows) == 5


def test_ddim_fixture_validation_and_input_rule():
    receipt = json.loads(
        (ROOT / "evidence" / "pilot" / "diffusion_sampler_validation.json").read_text(encoding="utf-8")
    )
    assert receipt["status"] == "PASS_MECHANICS_ONLY"
    assert receipt["study_images_used"] == 0
    assert all(receipt["acceptance"].values())
    model = CompactConditionalUNet(base=16, embedding_dim=32).eval()
    condition = torch.zeros(1, 13)
    cumulative = torch.cumprod(1 - cosine_schedule(20), dim=0)
    try:
        ddim_sample(model, condition, cumulative, seed=1, sampling_steps=1)
    except ValueError as error:
        assert "sampling_steps" in str(error)
    else:
        raise AssertionError("Invalid DDIM sampling step count was accepted")
