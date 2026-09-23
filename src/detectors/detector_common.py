"""Shared deterministic data, model, and COCO evaluation support for N2 detectors."""
from __future__ import annotations

from contextlib import redirect_stdout
import gzip
import io
import json
import math
from pathlib import Path
import random
from typing import Any

import numpy as np
from PIL import Image, ImageEnhance, ImageOps
import torch


def numeric_box(row: dict[str, Any]) -> list[float]:
    x = float(row["bbox_x_px"])
    y = float(row["bbox_y_px"])
    width = float(row["bbox_width_px"])
    height = float(row["bbox_height_px"])
    values = (x, y, width, height)
    if not all(math.isfinite(value) for value in values) or x < 0 or y < 0 or width <= 0 or height <= 0:
        raise ValueError("Invalid inherited bounding box")
    box = [x, y, x + width, y + height]
    if box[2] > 300 + 1e-6 or box[3] > 300 + 1e-6:
        raise ValueError("Inherited bounding box lies outside 300x300 image")
    return box


class ShuffledStream:
    """Serializable without-replacement stream with an isolated RNG."""

    def __init__(self, size: int, seed: int):
        if size <= 0:
            raise ValueError("Stream size must be positive")
        self.size = int(size)
        self.rng = random.Random(int(seed))
        self.order = list(range(self.size))
        self.rng.shuffle(self.order)
        self.cursor = 0
        self.epoch = 0

    def take(self, count: int) -> list[int]:
        if count < 0:
            raise ValueError("Cannot draw a negative count")
        output: list[int] = []
        while len(output) < count:
            if self.cursor == self.size:
                self.order = list(range(self.size))
                self.rng.shuffle(self.order)
                self.cursor = 0
                self.epoch += 1
            current = min(count - len(output), self.size - self.cursor)
            output.extend(self.order[self.cursor:self.cursor + current])
            self.cursor += current
        return output

    def state_dict(self) -> dict[str, Any]:
        return {
            "size": self.size,
            "rng": self.rng.getstate(),
            "order": self.order.copy(),
            "cursor": self.cursor,
            "epoch": self.epoch,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if int(state["size"]) != self.size or sorted(state["order"]) != list(range(self.size)):
            raise ValueError("Sampler state does not match roster")
        if not 0 <= int(state["cursor"]) <= self.size:
            raise ValueError("Sampler cursor is invalid")
        self.order = list(state["order"])
        self.cursor = int(state["cursor"])
        self.epoch = int(state["epoch"])
        self.rng.setstate(state["rng"])


class DetectorRows:
    def __init__(
        self,
        rows: list[dict[str, Any]],
        family: str,
        source_root: Path,
        augment_real: bool,
        expected_sha256: dict[str, str] | None = None,
    ):
        if family not in {"fasterrcnn", "retinanet"}:
            raise ValueError("Unknown detector family")
        self.rows = rows
        self.family = family
        self.source_root = source_root
        self.augment_real = bool(augment_real)
        self.expected_sha256 = expected_sha256 or {}
        self.array_cache: dict[str, np.ndarray] = {}
        self.verified_paths: set[str] = set()
        for row in rows:
            if row.get("source_kind") not in {"real", "additional_real", "synthetic"}:
                raise ValueError("Explicit real/additional_real/synthetic lineage is required")
            numeric_box(row)
            if row["source_kind"] == "synthetic" and not {"array_path", "array_index"} <= set(row):
                raise ValueError("Synthetic row lacks array address")

    def __len__(self) -> int:
        return len(self.rows)

    def _synthetic_image(self, row: dict[str, Any]) -> Image.Image:
        relative = str(row["array_path"]).replace("\\", "/")
        if relative not in self.array_cache:
            path = self.source_root / relative
            self.array_cache[relative] = np.load(path, mmap_mode="r", allow_pickle=False)
        array = self.array_cache[relative]
        index = int(row["array_index"])
        if array.shape[1:] != (1, 64, 64) or array.dtype != np.uint8 or not 0 <= index < len(array):
            raise RuntimeError("Synthetic pool array contract changed")
        pixels = np.asarray(array[index, 0])
        return Image.fromarray(pixels, mode="L").resize((300, 300), Image.Resampling.BILINEAR)

    def _real_image(self, row: dict[str, Any]) -> Image.Image:
        relative = str(row["image_path"]).replace("\\", "/")
        path = self.source_root / relative
        if relative not in self.verified_paths and relative in self.expected_sha256:
            from hashlib import sha256

            digest = sha256(path.read_bytes()).hexdigest()
            if digest != self.expected_sha256[relative]:
                raise RuntimeError(f"Real image hash mismatch: {relative}")
            self.verified_paths.add(relative)
        with Image.open(path) as handle:
            image = handle.convert("L").copy()
        if image.size != (300, 300):
            raise ValueError("Real detector input must be native 300x300")
        return image

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, Any]]:
        row = self.rows[index]
        box = numeric_box(row)
        image = self._synthetic_image(row) if row["source_kind"] == "synthetic" else self._real_image(row)
        if self.augment_real and row["source_kind"] in {"real", "additional_real"}:
            if random.random() < 0.5:
                image = ImageOps.mirror(image)
                box = [300 - box[2], box[1], 300 - box[0], box[3]]
            image = ImageEnhance.Brightness(image).enhance(random.uniform(0.8, 1.2))
            image = ImageEnhance.Contrast(image).enhance(random.uniform(0.8, 1.2))
        pixels = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(pixels).unsqueeze(0).repeat(3, 1, 1)
        area = (box[2] - box[0]) * (box[3] - box[1])
        label_value = 0 if self.family == "retinanet" else 1
        target = {
            "boxes": torch.tensor([box], dtype=torch.float32),
            "labels": torch.tensor([label_value], dtype=torch.int64),
            "image_id": torch.tensor([index], dtype=torch.int64),
            "area": torch.tensor([area], dtype=torch.float32),
            "iscrowd": torch.zeros(1, dtype=torch.int64),
        }
        meta = {
            "sample_id": row["sample_id"],
            "specimen": row["specimen"],
            "view": row["view"],
            "source_kind": row["source_kind"],
            "target_sample_id": row.get("target_sample_id", row["sample_id"]),
        }
        return tensor, target, meta


def build_model(family: str, specification: dict[str, Any], weight_path: Path) -> torch.nn.Module:
    from torchvision.models.detection import (
        FasterRCNN_ResNet50_FPN_V2_Weights,
        RetinaNet_ResNet50_FPN_V2_Weights,
        fasterrcnn_resnet50_fpn_v2,
        retinanet_resnet50_fpn_v2,
    )
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

    common = specification["model_common"]
    model_spec = specification["models"][family]
    old_hub = torch.hub.get_dir()
    torch.hub.set_dir(str(weight_path.parent.parent))
    try:
        if family == "fasterrcnn":
            model = fasterrcnn_resnet50_fpn_v2(
                weights=FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1,
                weights_backbone=None,
                min_size=common["input_min_size"],
                max_size=common["input_max_size"],
                trainable_backbone_layers=model_spec["trainable_backbone_layers"],
                box_score_thresh=common["score_threshold"],
                box_nms_thresh=common["nms_threshold"],
                box_detections_per_img=common["detections_per_image"],
            )
            features = model.roi_heads.box_predictor.cls_score.in_features
            model.roi_heads.box_predictor = FastRCNNPredictor(features, 2)
        elif family == "retinanet":
            model = retinanet_resnet50_fpn_v2(
                weights=RetinaNet_ResNet50_FPN_V2_Weights.COCO_V1,
                weights_backbone=None,
                min_size=common["input_min_size"],
                max_size=common["input_max_size"],
                trainable_backbone_layers=model_spec["trainable_backbone_layers"],
                score_thresh=common["score_threshold"],
                nms_thresh=common["nms_threshold"],
                detections_per_img=common["detections_per_image"],
                topk_candidates=1000,
            )
            head = model.head.classification_head
            old = head.cls_logits
            head.cls_logits = torch.nn.Conv2d(old.in_channels, head.num_anchors, kernel_size=3, stride=1, padding=1)
            torch.nn.init.normal_(head.cls_logits.weight, std=0.01)
            torch.nn.init.constant_(head.cls_logits.bias, -math.log((1 - 0.01) / 0.01))
            head.num_classes = 1
            if model.head.regression_head._loss_type != "giou":
                raise RuntimeError("RetinaNet-v2 GIoU head contract changed")
        else:
            raise ValueError(f"Unknown detector family {family}")
    finally:
        torch.hub.set_dir(old_hub)
    return model


def official_coco_metrics(records: list[dict[str, Any]]) -> dict[str, float | int | None]:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    images, annotations, detections = [], [], []
    for image_id, record in enumerate(records, 1):
        images.append({"id": image_id, "width": 300, "height": 300})
        for box in record.get("gt_boxes", []):
            x1, y1, x2, y2 = map(float, box)
            if x2 <= x1 or y2 <= y1:
                raise ValueError("Invalid COCO ground truth")
            annotations.append({
                "id": len(annotations) + 1,
                "image_id": image_id,
                "category_id": 1,
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "area": (x2 - x1) * (y2 - y1),
                "iscrowd": 0,
            })
        if not (len(record["boxes"]) == len(record["scores"]) == len(record["labels"])):
            raise ValueError("Prediction columns have unequal lengths")
        for box, score, label_value in zip(record["boxes"], record["scores"], record["labels"]):
            if int(label_value) != 1:
                raise ValueError("Only canonical reporting label 1 may enter COCO")
            x1, y1, x2, y2 = map(float, box)
            if not all(math.isfinite(value) for value in (x1, y1, x2, y2, float(score))) or x2 <= x1 or y2 <= y1:
                raise ValueError("Invalid COCO detection")
            detections.append({
                "image_id": image_id,
                "category_id": 1,
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "score": float(score),
            })
    names = ("COCO_AP", "COCO_AP50", "COCO_AP75", "COCO_AP_small", "COCO_AP_medium", "COCO_AP_large")
    if not images:
        return {**{name: None for name in names}, "COCO_AR100": None, "COCO_images": 0}
    with redirect_stdout(io.StringIO()):
        ground_truth = COCO()
        ground_truth.dataset = {
            "images": images,
            "annotations": annotations,
            "categories": [{"id": 1, "name": "inherited_bbox_region"}],
            "info": {},
        }
        ground_truth.createIndex()
        if detections:
            detected = ground_truth.loadRes(detections)
        else:
            detected = COCO()
            detected.dataset = {
                "images": images,
                "annotations": [],
                "categories": ground_truth.dataset["categories"],
                "info": {},
            }
            detected.createIndex()
        evaluator = COCOeval(ground_truth, detected, "bbox")
        evaluator.params.imgIds = [item["id"] for item in images]
        evaluator.params.catIds = [1]
        evaluator.params.maxDets = [1, 10, 100]
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()

    def clean(value: float) -> float | None:
        return float(value) if value >= 0 and math.isfinite(float(value)) else None

    return {
        **{name: clean(value) for name, value in zip(names, evaluator.stats[:6])},
        "COCO_AR100": clean(evaluator.stats[8]),
        "COCO_images": len(images),
    }


@torch.inference_mode()
def predict(model: torch.nn.Module, dataset: DetectorRows, device: torch.device, batch_size: int = 4) -> list[dict[str, Any]]:
    model.eval()
    expected_label = 0 if dataset.family == "retinanet" else 1
    output: list[dict[str, Any]] = []
    for start in range(0, len(dataset), batch_size):
        batch = [dataset[index] for index in range(start, min(start + batch_size, len(dataset)))]
        predictions = model([image.to(device) for image, _, _ in batch])
        for prediction, (_, target, meta) in zip(predictions, batch):
            keep = prediction["labels"] == expected_label
            output.append({
                **meta,
                "gt_boxes": target["boxes"].tolist(),
                "boxes": prediction["boxes"][keep].float().cpu().tolist(),
                "scores": prediction["scores"][keep].float().cpu().tolist(),
                "labels": [1] * int(keep.sum()),
            })
    return output


def per_specimen_metrics(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(str(record["specimen"]), []).append(record)
    rows = []
    for specimen, values in sorted(grouped.items()):
        rows.append({"specimen": specimen, "images": len(values), **official_coco_metrics(values)})
    return rows


def write_gzip_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as raw:
        with gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as handle:
            handle.write(json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":")).encode())
    temporary.replace(path)
