#!/usr/bin/env python3
"""Run the bounded W06 primary-detector timing/stability pilot.

The pilot uses training and validation rows only, stops after 200 updates, and
does not retain a scientific checkpoint. Inherited boxes are used only to
measure mechanics while W04 semantics remain unresolved.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
REV02 = ROOT / "src" / "reproduction" / "rev02_implementation_7b20c457"
HISTORICAL = ROOT / "historical" / "metal_spatter_pinn"
COPIED_WEIGHTS = (
    ROOT / "historical" / "LPBF_REV02_20260902" / "torch_cache" / "hub" / "checkpoints"
    / "fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth"
)
MANIFEST = ROOT / "data" / "manifests" / "image_manifest.csv"
RESULTS = ROOT / "evidence" / "pilot" / "pilot_results.csv"
SUMMARY = ROOT / "evidence" / "pilot" / "pilot_summary.json"
FAILURE = ROOT / "evidence" / "pilot" / "failure_amp_fp16_update1.json"
sys.path.insert(0, str(REV02 / "scripts"))

import rev02_detector as detector  # noqa: E402


SEED = 76001
UPDATES = 200
BATCH_SIZE = 4
MILESTONES = {0, 50, 100, 150, 200}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_values(values: list[str]) -> str:
    return hashlib.sha256("\n".join(sorted(values)).encode()).hexdigest()


def gpu_snapshot() -> dict[str, str | None]:
    fields = "name,driver_version,memory.total,memory.free,utilization.gpu"
    try:
        line = subprocess.check_output(
            ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
            text=True,
        ).strip().splitlines()[0]
        values = [value.strip() for value in line.split(",")]
        return dict(zip(fields.split(","), values))
    except Exception as error:  # diagnostic metadata must not mask the pilot
        return {"error": f"{type(error).__name__}: {error}"}


def load_rows(split: str) -> list[dict[str, str]]:
    with MANIFEST.open(encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["split"] == split]
    if not rows or any(row["split"] != split for row in rows):
        raise RuntimeError(f"Invalid {split} roster")
    return rows


def stratified_one_per_stratum_view(rows: list[dict[str, str]], seed: int) -> list[dict[str, str]]:
    buckets: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        buckets[(row["stratum"], row["view"])].append(row)
    if len(buckets) != 64:
        raise RuntimeError(f"Expected 64 stratum-view buckets, found {len(buckets)}")
    rng = random.Random(seed)
    selected = []
    for key in sorted(buckets):
        bucket = sorted(buckets[key], key=lambda row: row["sample_id"])
        selected.append(bucket[rng.randrange(len(bucket))])
    return selected


def dataset_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [{**row, "source_kind": "real"} for row in rows]


def evaluate(model, dataset, device: torch.device) -> tuple[dict[str, float | int | None], float]:
    started = time.perf_counter()
    model.eval()
    records = []
    with torch.no_grad():
        for start in range(0, len(dataset), BATCH_SIZE):
            batch = [dataset[index] for index in range(start, min(start + BATCH_SIZE, len(dataset)))]
            outputs = model([image.to(device) for image, _, _ in batch])
            for output, (_, target, _) in zip(outputs, batch):
                keep = output["labels"].eq(1)
                records.append(
                    {
                        "gt_boxes": target["boxes"].tolist(),
                        "boxes": output["boxes"][keep].detach().cpu().tolist(),
                        "scores": output["scores"][keep].detach().cpu().tolist(),
                        "labels": [1] * int(keep.sum()),
                        "width": 300,
                        "height": 300,
                    }
                )
    torch.cuda.synchronize()
    return detector.official_coco_metrics(records), time.perf_counter() - started


def write_csv(rows: list[dict[str, object]]) -> None:
    fields = sorted({key for row in rows for key in row})
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--amp-mode", choices=("fp16", "bf16", "off"), default="fp16")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if RESULTS.exists() or SUMMARY.exists():
        raise FileExistsError("Refusing to overwrite the W06 pilot evidence")
    if not torch.cuda.is_available():
        raise RuntimeError("W06 timing pilot requires CUDA")
    if not COPIED_WEIGHTS.is_file():
        raise FileNotFoundError(COPIED_WEIGHTS)
    if not sha256(COPIED_WEIGHTS).startswith("dd69338a"):
        raise RuntimeError("Copied Faster R-CNN weights failed checksum-prefix verification")

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    device = torch.device("cuda")

    train_rows = stratified_one_per_stratum_view(load_rows("train"), SEED)
    validation_rows = stratified_one_per_stratum_view(load_rows("validation"), SEED + 1)
    train_data = detector.RowsDataset(
        dataset_rows(train_rows), "fasterrcnn", resolver=lambda value: HISTORICAL / Path(value)
    )
    validation_data = detector.RowsDataset(
        dataset_rows(validation_rows), "fasterrcnn", resolver=lambda value: HISTORICAL / Path(value)
    )

    # Force initialization to use the managed historical copy, never the source output store.
    detector.weights_path = lambda family: COPIED_WEIGHTS
    model = detector.build_model("fasterrcnn", pretrained=True).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.003, momentum=0.9, weight_decay=0.0001)
    autocast_enabled = args.amp_mode != "off"
    autocast_dtype = torch.float16 if args.amp_mode == "fp16" else torch.bfloat16
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp_mode == "fp16")
    before_gpu = gpu_snapshot()
    result_rows: list[dict[str, object]] = []
    validation_ap = []
    initial_metric, initial_seconds = evaluate(model, validation_data, device)
    validation_ap.append(initial_metric["COCO_AP"])
    result_rows.append({"row_type": "validation", "update": 0, "elapsed_seconds": initial_seconds, **initial_metric})

    torch.cuda.reset_peak_memory_stats()
    order_rng = np.random.default_rng(SEED)
    order: list[int] = []
    timed_step_seconds: list[float] = []
    losses: list[float] = []
    total_started = time.perf_counter()
    for update in range(1, UPDATES + 1):
        if len(order) < BATCH_SIZE:
            order.extend(order_rng.permutation(len(train_data)).tolist())
        indices, order = order[:BATCH_SIZE], order[BATCH_SIZE:]
        batch = [train_data[index] for index in indices]
        images = [image.to(device) for image, _, _ in batch]
        targets = [{key: value.to(device) for key, value in target.items()} for _, target, _ in batch]

        model.train()
        optimizer.zero_grad(set_to_none=True)
        started = time.perf_counter()
        with torch.amp.autocast("cuda", enabled=autocast_enabled, dtype=autocast_dtype):
            loss_parts = model(images, targets)
            loss = sum(loss_parts.values())
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite loss at update {update}")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        if not torch.isfinite(gradient_norm):
            if args.amp_mode == "fp16" and not FAILURE.exists():
                FAILURE.parent.mkdir(parents=True, exist_ok=True)
                FAILURE.write_text(
                    json.dumps(
                        {
                            "schema_version": 1,
                            "created_at": datetime.now(timezone.utc).isoformat(),
                            "status": "STOPPED_AS_DESIGNED",
                            "stage": "W06_bounded_detector_pilot",
                            "update": update,
                            "amp_mode": args.amp_mode,
                            "loss_total": float(loss.detach().cpu()),
                            "gradient_norm_before_clip": str(float(gradient_norm.detach().cpu())),
                            "grad_scaler_scale": float(scaler.get_scale()),
                            "test_split_used": False,
                            "reason": "Nonfinite gradient norm after fp16 AMP unscale; strict pilot STOP condition.",
                            "recovery_scope": "A separately identified bf16 or full-fp32 technical pilot may diagnose numerical precision without selecting a scientific outcome.",
                        },
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )
            raise FloatingPointError(f"Nonfinite gradient at update {update}")
        scaler.step(optimizer)
        scaler.update()
        torch.cuda.synchronize()
        step_seconds = time.perf_counter() - started
        loss_value = float(loss.detach().cpu())
        timed_step_seconds.append(step_seconds)
        losses.append(loss_value)
        result_rows.append(
            {
                "row_type": "training",
                "update": update,
                "elapsed_seconds": step_seconds,
                "loss_total": loss_value,
                "gradient_norm_before_clip": float(gradient_norm.detach().cpu()),
                **{name: float(value.detach().cpu()) for name, value in loss_parts.items()},
            }
        )
        if update in MILESTONES:
            metric, eval_seconds = evaluate(model, validation_data, device)
            validation_ap.append(metric["COCO_AP"])
            result_rows.append(
                {"row_type": "validation", "update": update, "elapsed_seconds": eval_seconds, **metric}
            )
        if update % 25 == 0:
            print(f"W06 pilot update {update}/{UPDATES}; loss={loss_value:.4f}; step={step_seconds:.3f}s", flush=True)

    total_seconds = time.perf_counter() - total_started
    write_csv(result_rows)
    finite_aps = [float(value) for value in validation_ap if value is not None and math.isfinite(float(value))]
    after_gpu = gpu_snapshot()
    summary = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "PASS_WITH_LIMITATION",
        "scope": "bounded technical dev pilot; not a scientific trajectory or W08 model selection",
        "test_split_used": False,
        "inherited_label_semantics_resolved": False,
        "model": "torchvision_fasterrcnn_resnet50_fpn_v2",
        "initialization": "FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1 from managed copied artifact",
        "weights_sha256": sha256(COPIED_WEIGHTS),
        "resolution": [300, 300],
        "batch_size": BATCH_SIZE,
        "amp_mode": args.amp_mode,
        "autocast_enabled": autocast_enabled,
        "grad_scaler_enabled": scaler.is_enabled(),
        "optimizer": {"name": "SGD", "learning_rate": 0.003, "momentum": 0.9, "weight_decay": 0.0001},
        "updates": UPDATES,
        "training_rows": len(train_rows),
        "validation_rows": len(validation_rows),
        "training_specimens": len({row["specimen"] for row in train_rows}),
        "validation_specimens": len({row["specimen"] for row in validation_rows}),
        "training_sample_ids_sha256": hash_values([row["sample_id"] for row in train_rows]),
        "validation_sample_ids_sha256": hash_values([row["sample_id"] for row in validation_rows]),
        "training_view_counts": Counter(row["view"] for row in train_rows),
        "validation_view_counts": Counter(row["view"] for row in validation_rows),
        "timing": {
            "training_loop_seconds_including_milestone_validation": total_seconds,
            "step_mean_seconds_all": float(np.mean(timed_step_seconds)),
            "step_median_seconds_all": float(np.median(timed_step_seconds)),
            "step_p95_seconds_all": float(np.quantile(timed_step_seconds, 0.95)),
            "step_mean_seconds_after_5_warmup": float(np.mean(timed_step_seconds[5:])),
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
            "milestones": sorted(MILESTONES),
            "pooled_coco_ap": validation_ap,
            "all_defined_and_finite": len(finite_aps) == len(validation_ap),
        },
        "gpu_before": before_gpu,
        "gpu_after": after_gpu,
        "acceptance": {
            "exactly_200_updates": len(losses) == 200,
            "finite_losses_and_gradients": all(math.isfinite(value) for value in losses),
            "train_and_validation_only": True,
            "managed_copied_weight_used": True,
            "timing_and_memory_measured": True,
        },
        "limitations": [
            "W04 box semantics are unresolved; validation AP is a technical stability signal only.",
            "One short real-only pilot cannot select augmentation ratio, generator lambda, stopping rule, or architecture.",
            "Wall time may include background GPU/host contention and is not a vendor benchmark.",
            "No checkpoint is retained for scientific reuse and no held-out test image was accessed.",
        ],
    }
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=dict) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, default=dict))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
