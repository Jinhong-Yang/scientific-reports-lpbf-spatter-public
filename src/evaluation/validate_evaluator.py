#!/usr/bin/env python3
"""Run bounded W05 evaluator and forward/backward validation.

Only one validation image is decoded to verify the data interface. Model
optimization uses toy tensors and random initialization; no study trajectory,
checkpoint selection, or test-split access occurs.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
REV02 = ROOT / "src" / "reproduction" / "rev02_implementation_7b20c457"
HISTORICAL = ROOT / "historical" / "metal_spatter_pinn"
sys.path.insert(0, str(REV02 / "scripts"))
sys.path.insert(0, str(HISTORICAL / "src"))

import rev02_detector as detector  # noqa: E402
import rev02_generator as generator  # noqa: E402
from metal_spatter_pinn.data import condition_from_manifest  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def central_laplacian(function, x: float, y: float, step: float = 1e-4) -> float:
    center = float(function(x, y))
    dxx = (float(function(x + step, y)) - 2 * center + float(function(x - step, y))) / step**2
    dyy = (float(function(x, y + step)) - 2 * center + float(function(x, y - step))) / step**2
    return dxx + dyy


def load_one_validation_row() -> dict[str, str]:
    manifest = ROOT / "data" / "manifests" / "image_manifest.csv"
    with manifest.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["split"] == "validation":
                return row
    raise RuntimeError("No validation row found")


def data_interface_check() -> dict[str, object]:
    row = load_one_validation_row()
    source = HISTORICAL / Path(row["image_path"])
    if sha256(source) != row["image_sha256"]:
        raise RuntimeError("Validation fixture image hash mismatch")
    dataset_row = dict(row)
    dataset_row["source_kind"] = "real"
    dataset = detector.RowsDataset([dataset_row], "fasterrcnn", resolver=lambda value: HISTORICAL / Path(value))
    image, target, metadata = dataset[0]
    condition = condition_from_manifest(row)
    expected_box = [
        float(row["bbox_x_px"]),
        float(row["bbox_y_px"]),
        float(row["bbox_x_px"]) + float(row["bbox_width_px"]),
        float(row["bbox_y_px"]) + float(row["bbox_height_px"]),
    ]
    if image.shape != (3, 300, 300):
        raise RuntimeError(f"Unexpected detector tensor shape: {image.shape}")
    if not torch.allclose(target["boxes"][0], torch.tensor(expected_box, dtype=torch.float32)):
        raise RuntimeError("xywh to xyxy conversion mismatch")
    if condition.vector.shape != (13,) or not np.isfinite(condition.vector).all():
        raise RuntimeError("Condition vector invalid")
    if row["split"] != "validation":
        raise RuntimeError("Non-validation row entered W05 interface check")
    return {
        "status": "PASS",
        "split": "validation",
        "sample_id_sha256": hashlib.sha256(row["sample_id"].encode()).hexdigest(),
        "source_image_sha256": row["image_sha256"],
        "tensor_shape": list(image.shape),
        "tensor_dtype": str(image.dtype),
        "pixel_min": float(image.min()),
        "pixel_max": float(image.max()),
        "box_xyxy": expected_box,
        "condition_vector_length": int(condition.vector.size),
        "condition_vector_finite": True,
        "dataset_metadata_fields": sorted(metadata),
        "split_verified_from_manifest_row": True,
        "dataset_metadata_carries_split": "split" in metadata,
        "label_semantics_validated": False,
    }


def coco_oracle_check() -> dict[str, object]:
    def record(boxes, scores, ground_truth=True):
        return {
            "gt_boxes": [[10, 10, 30, 30]] if ground_truth else [],
            "boxes": boxes,
            "scores": scores,
            "labels": [1] * len(boxes),
            "width": 300,
            "height": 300,
        }

    perfect = detector.official_coco_metrics([record([[10, 10, 30, 30]], [0.9])])
    missed = detector.official_coco_metrics([record([], [])])
    empty_gt = detector.official_coco_metrics([record([[10, 10, 30, 30]], [0.9], False)])
    if not math.isclose(perfect["COCO_AP"], 1.0, abs_tol=1e-12):
        raise RuntimeError("Perfect COCO oracle did not produce AP=1")
    if missed["COCO_AP"] != 0.0:
        raise RuntimeError("Empty prediction oracle did not produce AP=0")
    if empty_gt["COCO_AP"] is not None:
        raise RuntimeError("Empty ground-truth oracle must be undefined")
    invalid_rejected = False
    invalid = record([[10, 10, 30, 30]], [0.9])
    invalid["labels"] = [0]
    try:
        detector.official_coco_metrics([invalid])
    except ValueError:
        invalid_rejected = True
    if not invalid_rejected:
        raise RuntimeError("Canonical foreground mapping error was not rejected")
    return {
        "status": "PASS",
        "perfect_prediction_AP": perfect["COCO_AP"],
        "empty_prediction_AP": missed["COCO_AP"],
        "empty_ground_truth_AP": empty_gt["COCO_AP"],
        "invalid_foreground_mapping_rejected": invalid_rejected,
        "official_evaluator": "pycocotools COCOeval bbox",
    }


def manufactured_solution_check() -> dict[str, object]:
    function = lambda x, y: x * x + y * y
    points = [(-0.7, -0.2), (0.0, 0.0), (0.4, 0.8)]
    finite = [central_laplacian(function, x, y) for x, y in points]
    coordinates = torch.tensor(points, dtype=torch.float64, requires_grad=True)
    values = coordinates[:, 0].square() + coordinates[:, 1].square()
    first = torch.autograd.grad(values.sum(), coordinates, create_graph=True)[0]
    dxx = torch.autograd.grad(first[:, 0].sum(), coordinates, retain_graph=True)[0][:, 0]
    dyy = torch.autograd.grad(first[:, 1].sum(), coordinates)[0][:, 1]
    autograd = (dxx + dyy).detach().cpu().numpy()
    finite_error = float(np.max(np.abs(np.asarray(finite) - 4.0)))
    autograd_error = float(np.max(np.abs(autograd - 4.0)))
    if finite_error > 1e-6 or autograd_error > 1e-12:
        raise RuntimeError("Manufactured Laplacian check failed")
    return {
        "status": "PASS",
        "manufactured_field": "u(x,y)=x^2+y^2",
        "expected_laplacian": 4.0,
        "finite_difference_max_abs_error": finite_error,
        "autograd_max_abs_error": autograd_error,
        "physical_calibration_claim": False,
    }


def deterministic_order_check() -> dict[str, object]:
    seed = 73001
    first = torch.randperm(1024, generator=torch.Generator().manual_seed(seed)).tolist()
    second = torch.randperm(1024, generator=torch.Generator().manual_seed(seed)).tolist()
    if first != second or sorted(first) != list(range(1024)):
        raise RuntimeError("Deterministic order fixture failed")
    digest = hashlib.sha256(",".join(map(str, first)).encode()).hexdigest()
    return {"status": "PASS", "seed": seed, "items": 1024, "permutation_sha256": digest}


def generator_toy_check(device: torch.device) -> dict[str, object]:
    torch.manual_seed(73002)
    model = generator.ConditionalVariationalField(
        generator.ModelConfig(decoder_type="pirate"), image_size=64
    ).to(device)
    condition = torch.zeros(2, 13, device=device)
    condition[:, 3] = 1
    condition[:, 5] = 1
    condition[:, 9] = 0.35
    condition[:, 12] = 1
    latent, _, coordinates, boundary = generator.draw_training_inputs(
        73002, 1, 2, 4096, 128, 16, 32, 32, device
    )
    grid = generator.make_coordinate_grid(16, device=device).unsqueeze(0).expand(2, -1, -1)
    losses = generator.regularizer_components(
        model.decoder, condition, latent, coordinates, boundary, grid
    )
    loss = sum(losses.values())
    loss.backward()
    if not torch.isfinite(loss):
        raise RuntimeError("Generator toy loss is nonfinite")
    return {
        "status": "PASS",
        "device": str(device),
        "loss": float(loss.detach().cpu()),
        "components": {name: float(value.detach().cpu()) for name, value in losses.items()},
        "study_images_used": 0,
    }


def detector_toy_check(family: str, device: torch.device) -> dict[str, object]:
    torch.manual_seed(73003 if family == "fasterrcnn" else 73004)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    model = detector.build_model(family, pretrained=False).to(device).train()
    expected_label = 0 if family == "retinanet" else 1
    images = []
    targets = []
    for index in range(2):
        image = torch.zeros(3, 300, 300, device=device)
        image[:, 135:155, 140:160] = 0.7 + 0.01 * index
        images.append(image)
        targets.append(
            {
                "boxes": torch.tensor([[140.0, 135.0, 160.0, 155.0]], device=device),
                "labels": torch.tensor([expected_label], dtype=torch.int64, device=device),
            }
        )
    optimizer = torch.optim.SGD(
        [parameter for parameter in model.parameters() if parameter.requires_grad], lr=0.003, momentum=0.9
    )
    start = time.perf_counter()
    losses = model(images, targets)
    loss = sum(losses.values())
    if not torch.isfinite(loss):
        raise RuntimeError(f"{family} toy loss is nonfinite")
    loss.backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
    if not torch.isfinite(gradient_norm):
        raise RuntimeError(f"{family} toy gradient is nonfinite")
    optimizer.step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    model.eval()
    with torch.no_grad():
        prediction = model(images[:1])[0]
    keep = prediction["labels"].eq(expected_label)
    canonical_record = {
        "gt_boxes": [[140.0, 135.0, 160.0, 155.0]],
        "boxes": prediction["boxes"][keep].detach().cpu().tolist(),
        "scores": prediction["scores"][keep].detach().cpu().tolist(),
        "labels": [1] * int(keep.sum()),
        "width": 300,
        "height": 300,
    }
    metric = detector.official_coco_metrics([canonical_record])
    result = {
        "status": "PASS",
        "family": family,
        "device": str(device),
        "toy_batch": 2,
        "loss": float(loss.detach().cpu()),
        "gradient_norm_before_clip": float(gradient_norm.detach().cpu()),
        "elapsed_seconds": time.perf_counter() - start,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2 if device.type == "cuda" else None,
        "foreground_label": expected_label,
        "prediction_count": int(keep.sum()),
        "canonical_coco_AP_after_random_toy_step": metric["COCO_AP"],
        "prediction_to_metric_path_executed": True,
        "pretrained_weights_downloaded": False,
        "study_images_used": 0,
    }
    del model, images, targets, optimizer, losses, loss, gradient_norm, prediction, canonical_record
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def main() -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("W05 CUDA validation requires an available GPU")
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda")
    checks = {
        "data_interface": data_interface_check(),
        "coco_oracle": coco_oracle_check(),
        "manufactured_solution": manufactured_solution_check(),
        "deterministic_order": deterministic_order_check(),
        "generator_toy": generator_toy_check(device),
        "fasterrcnn_toy": detector_toy_check("fasterrcnn", device),
        "retinanet_toy": detector_toy_check("retinanet", device),
    }
    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "PASS_WITH_LIMITATION",
        "scope": "W05 mechanics validation; no full training and no test-split access",
        "implementation_commit": "7b20c45788b2020142cd2de7ed47b866253f851d",
        "checks": checks,
        "label_semantics_validated": False,
        "full_training_authorized": False,
        "limitations": [
            "PyTorch reports that roi_align backward lacks a deterministic implementation; deterministic data order is verified, but detector backward is not claimed bitwise deterministic.",
            "Detector metrics from random toy weights validate the pipeline only and are not scientific outcomes.",
            "Human label semantics remain unresolved at W04."
        ],
    }
    output = ROOT / "evidence" / "evaluator" / "evaluator_validation.json"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing validation receipt: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
