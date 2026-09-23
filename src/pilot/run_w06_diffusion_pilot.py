#!/usr/bin/env python3
"""Run a bounded 200-update technical pilot for the proposed ND comparator."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn


ROOT = Path(__file__).resolve().parents[2]
HISTORICAL = ROOT / "historical" / "metal_spatter_pinn"
sys.path.insert(0, str(HISTORICAL / "src"))

from metal_spatter_pinn.data import condition_from_manifest  # noqa: E402


MANIFEST = ROOT / "data" / "manifests" / "image_manifest.csv"
RESULTS = ROOT / "evidence" / "pilot" / "diffusion_pilot_results.csv"
SUMMARY = ROOT / "evidence" / "pilot" / "diffusion_pilot_summary.json"
SEED = 76002
UPDATES = 200
BATCH_SIZE = 16
IMAGE_SIZE = 64
DIFFUSION_STEPS = 1000
MILESTONES = {0, 50, 100, 150, 200}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_values(values: list[str]) -> str:
    return hashlib.sha256("\n".join(sorted(values)).encode()).hexdigest()


def load_rows(split: str) -> list[dict[str, str]]:
    with MANIFEST.open(encoding="utf-8", newline="") as handle:
        return [row for row in csv.DictReader(handle) if row["split"] == split]


def select_roster(rows: list[dict[str, str]], seed: int) -> list[dict[str, str]]:
    buckets: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        buckets[(row["stratum"], row["view"])].append(row)
    if len(buckets) != 64:
        raise RuntimeError(f"Expected 64 stratum-view buckets, found {len(buckets)}")
    rng = random.Random(seed)
    return [
        sorted(buckets[key], key=lambda row: row["sample_id"])[rng.randrange(len(buckets[key]))]
        for key in sorted(buckets)
    ]


def load_tensors(rows: list[dict[str, str]]) -> tuple[torch.Tensor, torch.Tensor]:
    images = []
    conditions = []
    for row in rows:
        path = HISTORICAL / Path(row["image_path"])
        if sha256(path) != row["image_sha256"]:
            raise RuntimeError(f"Image hash mismatch: {row['sample_id']}")
        with Image.open(path) as handle:
            image = handle.convert("L").resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
        array = np.asarray(image, dtype=np.float32) / 127.5 - 1.0
        images.append(torch.from_numpy(array).unsqueeze(0))
        conditions.append(torch.from_numpy(condition_from_manifest(row).vector.astype(np.float32)))
    return torch.stack(images), torch.stack(conditions)


class TimeEmbedding(nn.Module):
    def __init__(self, dimension: int = 128):
        super().__init__()
        self.dimension = dimension
        self.mlp = nn.Sequential(nn.Linear(dimension, dimension * 2), nn.SiLU(), nn.Linear(dimension * 2, dimension))

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half = self.dimension // 2
        frequency = torch.exp(
            -math.log(10000) * torch.arange(half, device=timesteps.device, dtype=torch.float32) / (half - 1)
        )
        angles = timesteps.float().unsqueeze(1) * frequency.unsqueeze(0)
        return self.mlp(torch.cat([angles.sin(), angles.cos()], dim=1))


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, embedding_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.embedding = nn.Linear(embedding_dim, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, inputs: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
        hidden = self.conv1(torch.nn.functional.silu(self.norm1(inputs)))
        hidden = hidden + self.embedding(embedding).unsqueeze(-1).unsqueeze(-1)
        hidden = self.conv2(torch.nn.functional.silu(self.norm2(hidden)))
        return hidden + self.skip(inputs)


class CompactConditionalUNet(nn.Module):
    def __init__(self, condition_dim: int = 13, base: int = 64, embedding_dim: int = 128):
        super().__init__()
        self.time = TimeEmbedding(embedding_dim)
        self.condition = nn.Sequential(
            nn.Linear(condition_dim, embedding_dim), nn.SiLU(), nn.Linear(embedding_dim, embedding_dim)
        )
        self.input = nn.Conv2d(1, base, 3, padding=1)
        self.block1 = ResBlock(base, base, embedding_dim)
        self.down1 = nn.Conv2d(base, base * 2, 4, stride=2, padding=1)
        self.block2 = ResBlock(base * 2, base * 2, embedding_dim)
        self.down2 = nn.Conv2d(base * 2, base * 4, 4, stride=2, padding=1)
        self.middle = ResBlock(base * 4, base * 4, embedding_dim)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 4, stride=2, padding=1)
        self.block_up2 = ResBlock(base * 4, base * 2, embedding_dim)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 4, stride=2, padding=1)
        self.block_up1 = ResBlock(base * 2, base, embedding_dim)
        self.output = nn.Sequential(nn.GroupNorm(8, base), nn.SiLU(), nn.Conv2d(base, 1, 3, padding=1))

    def forward(self, noisy: torch.Tensor, timesteps: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        embedding = self.time(timesteps) + self.condition(condition)
        hidden1 = self.block1(self.input(noisy), embedding)
        hidden2 = self.block2(self.down1(hidden1), embedding)
        middle = self.middle(self.down2(hidden2), embedding)
        up2 = self.block_up2(torch.cat([self.up2(middle), hidden2], dim=1), embedding)
        up1 = self.block_up1(torch.cat([self.up1(up2), hidden1], dim=1), embedding)
        return self.output(up1)


def cosine_schedule(steps: int, s: float = 0.008) -> torch.Tensor:
    points = torch.linspace(0, steps, steps + 1, dtype=torch.float64)
    cumulative = torch.cos(((points / steps + s) / (1 + s)) * math.pi / 2).square()
    cumulative = cumulative / cumulative[0]
    betas = 1 - cumulative[1:] / cumulative[:-1]
    return betas.clamp(0.0001, 0.999).float()


def fixed_validation_loss(
    model: nn.Module,
    images: torch.Tensor,
    conditions: torch.Tensor,
    cumulative: torch.Tensor,
    device: torch.device,
) -> float:
    generator = torch.Generator(device=device).manual_seed(7600200)
    total = 0.0
    model.eval()
    with torch.no_grad():
        for start in range(0, len(images), BATCH_SIZE):
            clean = images[start:start + BATCH_SIZE].to(device)
            condition = conditions[start:start + BATCH_SIZE].to(device)
            timesteps = torch.full((len(clean),), 500, dtype=torch.long, device=device)
            noise = torch.randn(clean.shape, generator=generator, device=device)
            alpha = cumulative[timesteps].view(-1, 1, 1, 1)
            noisy = alpha.sqrt() * clean + (1 - alpha).sqrt() * noise
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                predicted = model(noisy, timesteps, condition)
                loss = torch.nn.functional.mse_loss(predicted.float(), noise.float(), reduction="sum")
            total += float(loss.cpu())
    torch.cuda.synchronize()
    return total / images.numel()


def write_results(rows: list[dict[str, object]]) -> None:
    fields = sorted({key for row in rows for key in row})
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    if RESULTS.exists() or SUMMARY.exists():
        raise FileExistsError("Refusing to overwrite diffusion pilot evidence")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    device = torch.device("cuda")

    train_rows = select_roster(load_rows("train"), SEED)
    validation_rows = select_roster(load_rows("validation"), SEED + 1)
    train_images, train_conditions = load_tensors(train_rows)
    validation_images, validation_conditions = load_tensors(validation_rows)
    model = CompactConditionalUNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    betas = cosine_schedule(DIFFUSION_STEPS).to(device)
    cumulative = torch.cumprod(1 - betas, dim=0)
    rng = torch.Generator(device=device).manual_seed(SEED)
    order_rng = np.random.default_rng(SEED)
    order: list[int] = []
    result_rows: list[dict[str, object]] = []
    validation_losses = []

    initial = fixed_validation_loss(model, validation_images, validation_conditions, cumulative, device)
    validation_losses.append(initial)
    result_rows.append({"row_type": "validation", "update": 0, "fixed_t500_noise_mse": initial})
    torch.cuda.reset_peak_memory_stats()
    step_times = []
    losses = []
    total_started = time.perf_counter()
    for update in range(1, UPDATES + 1):
        if len(order) < BATCH_SIZE:
            order.extend(order_rng.permutation(len(train_images)).tolist())
        indices, order = order[:BATCH_SIZE], order[BATCH_SIZE:]
        clean = train_images[indices].to(device)
        condition = train_conditions[indices].to(device)
        timesteps = torch.randint(0, DIFFUSION_STEPS, (BATCH_SIZE,), generator=rng, device=device)
        noise = torch.randn(clean.shape, generator=rng, device=device)
        alpha = cumulative[timesteps].view(-1, 1, 1, 1)
        noisy = alpha.sqrt() * clean + (1 - alpha).sqrt() * noise

        model.train()
        optimizer.zero_grad(set_to_none=True)
        started = time.perf_counter()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            predicted = model(noisy, timesteps, condition)
            loss = torch.nn.functional.mse_loss(predicted.float(), noise.float())
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite diffusion loss at update {update}")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(gradient_norm):
            raise FloatingPointError(f"Nonfinite diffusion gradient at update {update}")
        optimizer.step()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        value = float(loss.detach().cpu())
        step_times.append(elapsed)
        losses.append(value)
        result_rows.append(
            {
                "row_type": "training", "update": update, "elapsed_seconds": elapsed,
                "loss": value, "gradient_norm_before_clip": float(gradient_norm.detach().cpu()),
            }
        )
        if update in MILESTONES:
            metric = fixed_validation_loss(model, validation_images, validation_conditions, cumulative, device)
            validation_losses.append(metric)
            result_rows.append({"row_type": "validation", "update": update, "fixed_t500_noise_mse": metric})
        if update % 25 == 0:
            print(f"W06 diffusion pilot {update}/{UPDATES}; loss={value:.4f}; step={elapsed:.3f}s", flush=True)

    total_seconds = time.perf_counter() - total_started
    write_results(result_rows)
    summary = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "PASS_WITH_LIMITATION",
        "scope": "bounded ND technical pilot; no full fit, generation-quality comparison, or test access",
        "model_id": "ND_compact_conditional_ddpm_v0",
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "external_pretraining": False,
        "resolution": [IMAGE_SIZE, IMAGE_SIZE],
        "condition_dimension": 13,
        "diffusion_steps": DIFFUSION_STEPS,
        "prediction_target": "epsilon",
        "precision": "bfloat16_autocast",
        "batch_size": BATCH_SIZE,
        "updates": UPDATES,
        "test_split_used": False,
        "training_rows": len(train_rows),
        "validation_rows": len(validation_rows),
        "training_sample_ids_sha256": hash_values([row["sample_id"] for row in train_rows]),
        "validation_sample_ids_sha256": hash_values([row["sample_id"] for row in validation_rows]),
        "timing": {
            "loop_seconds_including_four_milestone_validations": total_seconds,
            "step_mean_seconds_after_5_warmup": float(np.mean(step_times[5:])),
            "step_median_seconds_all": float(np.median(step_times)),
            "step_p95_seconds_all": float(np.quantile(step_times, 0.95)),
        },
        "memory": {
            "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
            "peak_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2,
        },
        "loss": {
            "first_20_mean": float(np.mean(losses[:20])),
            "last_20_mean": float(np.mean(losses[-20:])),
            "all_finite": all(math.isfinite(value) for value in losses),
        },
        "validation": {
            "metric": "fixed_seed_t500_epsilon_MSE",
            "milestones": sorted(MILESTONES),
            "values": validation_losses,
            "all_finite": all(math.isfinite(value) for value in validation_losses),
        },
        "acceptance": {
            "exactly_200_updates": len(losses) == 200,
            "finite_losses_and_gradients": all(math.isfinite(value) for value in losses),
            "train_and_validation_only": True,
            "timing_and_memory_measured": True,
            "fixed_validation_path_executed": len(validation_losses) == 5,
        },
        "limitations": [
            "A 200-update epsilon-prediction pilot does not establish image quality or convergence.",
            "No DDIM sampling, FID/KID, label validity, memorization, or detector utility was evaluated.",
            "The 64x64 pilot does not estimate 128/256/300-resolution cost.",
            "One trajectory cannot estimate seed or pipeline variance."
        ]
    }
    SUMMARY.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
