#!/usr/bin/env python3
"""Validate deterministic DDIM mechanics on an untrained compact model."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import torch

from run_w06_diffusion_pilot import CompactConditionalUNet, cosine_schedule


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "evidence" / "pilot" / "diffusion_sampler_validation.json"


@torch.no_grad()
def ddim_sample(
    model: torch.nn.Module,
    condition: torch.Tensor,
    cumulative: torch.Tensor,
    *,
    seed: int,
    sampling_steps: int,
) -> torch.Tensor:
    if sampling_steps < 2 or sampling_steps > len(cumulative):
        raise ValueError("sampling_steps must be between 2 and the training diffusion length")
    device = condition.device
    generator = torch.Generator(device=device).manual_seed(seed)
    sample = torch.randn((len(condition), 1, 64, 64), generator=generator, device=device)
    schedule = torch.linspace(len(cumulative) - 1, 0, sampling_steps, device=device).round().long()
    for index, timestep in enumerate(schedule):
        times = torch.full((len(condition),), int(timestep), dtype=torch.long, device=device)
        alpha = cumulative[times].view(-1, 1, 1, 1)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            epsilon = model(sample, times, condition).float()
        clean = ((sample - (1 - alpha).sqrt() * epsilon) / alpha.sqrt()).clamp(-1, 1)
        if index == len(schedule) - 1:
            sample = clean
        else:
            previous = cumulative[schedule[index + 1]].view(1, 1, 1, 1)
            sample = previous.sqrt() * clean + (1 - previous).sqrt() * epsilon
        if not torch.isfinite(sample).all():
            raise FloatingPointError(f"Nonfinite DDIM state at step {index}")
    return sample


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUTPUT}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.manual_seed(76003)
    torch.cuda.manual_seed_all(76003)
    device = torch.device("cuda")
    model = CompactConditionalUNet().to(device).eval()
    condition = torch.zeros(2, 13, device=device)
    condition[:, 3] = 1
    condition[:, 5] = 1
    cumulative = torch.cumprod(1 - cosine_schedule(1000).to(device), dim=0)
    first = ddim_sample(model, condition, cumulative, seed=76004, sampling_steps=10)
    second = ddim_sample(model, condition, cumulative, seed=76004, sampling_steps=10)
    different = ddim_sample(model, condition, cumulative, seed=76005, sampling_steps=10)
    result = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "PASS_MECHANICS_ONLY",
        "model_weights": "random_untrained",
        "study_images_used": 0,
        "test_split_used": False,
        "sampling_rule": "deterministic_DDIM_eta_0",
        "training_diffusion_steps": 1000,
        "sampling_steps_fixture": 10,
        "output_shape": list(first.shape),
        "output_min": float(first.min().cpu()),
        "output_max": float(first.max().cpu()),
        "acceptance": {
            "same_seed_bitwise_equal": torch.equal(first, second),
            "different_seed_changes_output": not torch.equal(first, different),
            "finite_output": bool(torch.isfinite(first).all()),
            "bounded_output": bool(first.min() >= -1 and first.max() <= 1),
            "expected_shape": list(first.shape) == [2, 1, 64, 64],
        },
        "limitations": [
            "Random untrained weights validate reverse-process mechanics only.",
            "Ten fixture steps do not choose the proposed production sampling count of 50.",
            "No image-quality, diversity, memorization, label, or detector-utility claim is supported."
        ]
    }
    if not all(result["acceptance"].values()):
        result["status"] = "FAILED"
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["status"] != "FAILED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
