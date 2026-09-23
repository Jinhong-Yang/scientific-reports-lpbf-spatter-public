"""Technical GPU smoke checks on toy tensors only, never study images or labels."""
from __future__ import annotations

import gc
import math
from pathlib import Path
import sys
import time

import torch

from rev02_common import REV_ROOT, seed_everything, utc_now, verify_protocols, write_json
from rev02_detector import build_model
from rev02_generator import (ConditionalVariationalField, ModelConfig, draw_training_inputs,
                             make_coordinate_grid, regularizer_components)


def main():
    verify_protocols()
    if not torch.cuda.is_available():
        raise RuntimeError("Frozen GPU environment unavailable")
    device = torch.device("cuda")
    rows = []
    for family in ("fasterrcnn", "retinanet"):
        seed_everything(71001)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        model = build_model(family, pretrained=True).to(device).train()
        images = []
        targets = []
        for index in range(12):
            image = torch.zeros(3, 300, 300, device=device)
            image[:, 135:152, 141:155] = 0.7 + 0.01 * index
            images.append(image)
            targets.append({"boxes": torch.tensor([[141., 135., 155., 152.]], device=device),
                            "labels": torch.tensor([0 if family == "retinanet" else 1], dtype=torch.int64, device=device)})
        optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=.003, momentum=.9, weight_decay=.0001)
        scaler = torch.amp.GradScaler("cuda")
        start = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.float16):
            losses = model(images, targets)
            loss = sum(losses.values())
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite toy detector loss")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10.)
        if not torch.isfinite(norm):
            raise RuntimeError("Nonfinite toy detector gradient")
        scaler.step(optimizer)
        scaler.update()
        torch.cuda.synchronize()
        result = {"component": family, "toy_batch": 12, "loss": float(loss.detach().cpu()),
                  "gradient_norm_before_clip": float(norm.cpu()), "elapsed_seconds": time.perf_counter()-start,
                  "peak_allocated_MiB": torch.cuda.max_memory_allocated()/1024**2,
                  "peak_reserved_MiB": torch.cuda.max_memory_reserved()/1024**2,
                  "parameters_trainable": sum(p.numel() for p in model.parameters() if p.requires_grad)}
        model.eval()
        with torch.no_grad():
            predictions = model(images[:1])
        expected_label = 0 if family == "retinanet" else 1
        if any(int(label) != expected_label for label in predictions[0]["labels"]):
            raise RuntimeError("Unexpected detector foreground label")
        result["prediction_label_check"] = "PASS"
        rows.append(result)
        print(result, flush=True)
        del model, images, targets, losses, loss, optimizer, scaler, predictions, norm
        gc.collect()
        torch.cuda.empty_cache()
    seed_everything(71002)
    model = ConditionalVariationalField(ModelConfig(decoder_type="pirate"), image_size=64).to(device)
    condition = torch.zeros(2, 13, device=device)
    condition[:, 3] = 1
    condition[:, 5] = 1
    condition[:, 9] = .35
    condition[:, 12] = 1
    latent, _, coordinates, boundary = draw_training_inputs(71002, 1, 2, 4096, 128, 16, 32, 32, device)
    grid = make_coordinate_grid(16, device=device).unsqueeze(0).expand(2, -1, -1)
    losses = regularizer_components(model.decoder, condition, latent, coordinates, boundary, grid)
    loss = sum(losses.values())
    loss.backward()
    if not all(torch.isfinite(value) for value in losses.values()):
        raise RuntimeError("Nonfinite toy generator regularizer")
    rows.append({"component": "generator_second_derivative_losses", "status": "PASS",
                 "toy_batch": 2, "losses": {key: float(value.detach().cpu()) for key, value in losses.items()}})
    output = REV_ROOT / "reports" / "GPU_TOY_SMOKE.json"
    write_json(output, {"status": "PASS", "finished_utc": utc_now(), "study_data_used": False,
                        "planned_seed_or_checkpoint_reused": False, "checks": rows})
    print(f"Toy-only GPU smoke passed: {output}", flush=True)


if __name__ == "__main__":
    main()
