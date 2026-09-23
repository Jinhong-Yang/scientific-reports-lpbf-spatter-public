"""REV02 detectors: validation selection, frozen trajectories, and gated T7.

Importing this module does not load any dataset or protocol. All data entry points
use rev02_common's split-specific loader. Toy tests exercise the pure contracts.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import redirect_stdout
from fractions import Fraction
import gzip
import hashlib
import importlib.metadata
import io
import json
import math
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
from typing import Any

import numpy as np
from PIL import Image, ImageEnhance, ImageOps
import torch


def common():
    import rev02_common
    return rev02_common


def ratio_tag(ratio: float) -> str:
    return format(float(ratio), "g").replace(".", "p")


def budget_run_id(family: str, arm: str, axis: str, ratio: float, seed: int) -> str:
    canonical = "shared" if family == "fasterrcnn" and arm in {"D0", "D1"} else axis
    return f"budget_{family}_{arm}_r{ratio_tag(ratio)}_{canonical}_s{int(seed)}"


def ratio_run_id(arm: str, ratio: float, seed: int) -> str:
    return f"ratio_fasterrcnn_{arm}_r{ratio_tag(ratio)}_s{int(seed)}"


def synthetic_count(ratio: float, step: int) -> int:
    if step < 1 or ratio < 0:
        raise ValueError("A positive step and nonnegative ratio are required")
    multiplier = 4 * Fraction(str(ratio))
    return (multiplier * step).__floor__() - (multiplier * (step - 1)).__floor__()


class ShuffledStream:
    """Serializable without-replacement stream with a private RNG and cursor."""
    def __init__(self, size: int, seed: int):
        if size <= 0:
            raise ValueError("Stream size must be positive")
        self.size = int(size)
        self.rng = random.Random(seed)
        self.order = list(range(self.size))
        self.rng.shuffle(self.order)
        self.cursor = 0
        self.epoch = 0

    def take(self, count: int) -> list[int]:
        if count < 0:
            raise ValueError("Negative stream draw")
        out = []
        while len(out) < count:
            if self.cursor == self.size:
                self.order = list(range(self.size))
                self.rng.shuffle(self.order)
                self.cursor = 0
                self.epoch += 1
            n = min(count - len(out), self.size - self.cursor)
            out.extend(self.order[self.cursor:self.cursor + n])
            self.cursor += n
        return out

    def state_dict(self) -> dict:
        return {"size": self.size, "order": self.order.copy(), "cursor": self.cursor,
                "epoch": self.epoch, "rng": self.rng.getstate()}

    def load_state_dict(self, state: dict) -> None:
        if state["size"] != self.size or sorted(state["order"]) != list(range(self.size)):
            raise ValueError("Sampler state does not match input roster")
        if not 0 <= state["cursor"] <= self.size:
            raise ValueError("Invalid sampler cursor")
        self.order = state["order"].copy()
        self.cursor, self.epoch = state["cursor"], state["epoch"]
        self.rng.setstate(state["rng"])


class ExposureSampler:
    def __init__(self, nreal: int, nsyn: int, ratio: float, axis: str, seed: int):
        self.nreal, self.nsyn, self.ratio, self.axis = nreal, nsyn, ratio, axis
        if nreal % 4 or (nreal + nsyn) % 4:
            raise ValueError("Frozen full-batch roster must be divisible by four")
        if Fraction(str(ratio)) * nreal != nsyn:
            raise ValueError("Pool count does not equal frozen synthetic/real ratio")
        self.mixed = ShuffledStream(nreal + nsyn, seed)
        self.real = ShuffledStream(nreal, seed)
        self.synthetic = ShuffledStream(nsyn, seed + 1000003) if nsyn else None

    def batch(self, step: int) -> list[int]:
        if self.axis == "B4":
            real = self.real.take(4)
            count = synthetic_count(self.ratio, step)
            syn = self.synthetic.take(count) if self.synthetic else []
            if len(syn) != count:
                raise ValueError("Missing synthetic stream")
            return real + [self.nreal + i for i in syn]
        return self.mixed.take(4)

    def state_dict(self) -> dict:
        return {"contract": [self.nreal, self.nsyn, self.ratio, self.axis],
                "mixed": self.mixed.state_dict(), "real": self.real.state_dict(),
                "synthetic": self.synthetic.state_dict() if self.synthetic else None}

    def load_state_dict(self, state: dict) -> None:
        if state["contract"] != [self.nreal, self.nsyn, self.ratio, self.axis]:
            raise ValueError("Exposure sampler contract mismatch")
        self.mixed.load_state_dict(state["mixed"])
        self.real.load_state_dict(state["real"])
        if self.synthetic:
            self.synthetic.load_state_dict(state["synthetic"])


def _number(row: dict, key: str) -> float:
    result = float(row[key])
    if not math.isfinite(result):
        raise ValueError(f"Nonfinite {key}")
    return result


def row_box(row: dict) -> list[float]:
    x, y = _number(row, "bbox_x_px"), _number(row, "bbox_y_px")
    w, h = _number(row, "bbox_width_px"), _number(row, "bbox_height_px")
    if w <= 0 or h <= 0 or x < 0 or y < 0:
        raise ValueError("Invalid canonical bounding box")
    return [x, y, x + w, y + h]


class RowsDataset:
    def __init__(self, rows: list[dict], family: str, augment: bool = False, resolver=None):
        self.rows, self.family, self.augment = rows, family, augment
        self.resolver = resolver if resolver else common().resolve_path
        self.verified_images = set()
        for row in rows:
            if row.get("source_kind") not in {"real", "synthetic"}:
                raise ValueError("Every row requires explicit source_kind")
            row_box(row)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        path = self.resolver(row["image_path"])
        expected = row.get("image_sha256")
        if expected and index not in self.verified_images:
            if common().sha256_file(path).lower() != expected.lower():
                raise RuntimeError("Image content differs from frozen row hash")
            self.verified_images.add(index)
        with Image.open(path) as handle:
            im = handle.convert("L")
        if im.size != (300, 300):
            raise ValueError("Detector source image must already be canonical 300x300")
        box = row_box(row)
        if box[2] > 300 + 1e-6 or box[3] > 300 + 1e-6:
            raise ValueError("Bounding box outside image")
        if self.augment:
            if row["source_kind"] != "real":
                raise ValueError("Conventional augmentation must use real-only rows")
            if random.random() < .5:
                im = ImageOps.mirror(im)
                box = [300 - box[2], box[1], 300 - box[0], box[3]]
            im = ImageEnhance.Brightness(im).enhance(random.uniform(.8, 1.2))
            im = ImageEnhance.Contrast(im).enhance(random.uniform(.8, 1.2))
        image = torch.from_numpy(np.array(im, dtype=np.float32) / 255).unsqueeze(0).repeat(3, 1, 1)
        area = (box[2] - box[0]) * (box[3] - box[1])
        target = {"boxes": torch.tensor([box], dtype=torch.float32),
                  "labels": torch.tensor([0 if self.family == "retinanet" else 1], dtype=torch.int64),
                  "image_id": torch.tensor([index]), "area": torch.tensor([area]),
                  "iscrowd": torch.zeros(1, dtype=torch.int64)}
        return image, target, {k: row.get(k, "") for k in
                               ("sample_id", "specimen", "view", "image_path", "source_kind")}


def _historical_metrics(records: list[dict]) -> dict:
    c = common()
    if str(c.PROJECT_ROOT / "src") not in sys.path:
        sys.path.insert(0, str(c.PROJECT_ROOT / "src"))
    from metal_spatter_pinn.detector import metrics_from_records
    raw = metrics_from_records(records)
    mapping = {"coco_map_50_95": "study_mAP_50_95", "ap50": "study_AP_50",
               "ap75": "study_AP_75", "ap_small": "study_AP_small_50"}
    return {mapping.get(k, k): float(v) if np.isfinite(v) else None for k, v in raw.items()}


def official_coco_metrics(records: list[dict]) -> dict:
    """Official COCO bbox AP, including empty detections/ground-truth datasets."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    images, annotations, detections = [], [], []
    for image_id, r in enumerate(records, 1):
        images.append({"id": image_id, "width": int(r.get("width", 300)), "height": int(r.get("height", 300))})
        boxes = r.get("gt_boxes", [r["gt_box"]] if r.get("gt_box") is not None else [])
        for box in boxes:
            x, y, x2, y2 = map(float, box)
            if x2 <= x or y2 <= y:
                raise ValueError("Invalid ground truth for COCO")
            annotations.append({"id": len(annotations) + 1, "image_id": image_id,
                                "category_id": 1, "bbox": [x, y, x2-x, y2-y],
                                "area": (x2-x)*(y2-y), "iscrowd": 0})
        labels = r.get("labels", [1] * len(r["boxes"]))
        if not (len(labels) == len(r["boxes"]) == len(r["scores"])):
            raise ValueError("Prediction column lengths differ")
        for box, score, label in zip(r["boxes"], r["scores"], labels):
            if label != 1:
                raise ValueError("Only canonical foreground label 1 may enter COCO")
            x, y, x2, y2 = map(float, box)
            if not all(math.isfinite(v) for v in [x, y, x2, y2, float(score)]) or x2 <= x or y2 <= y:
                raise ValueError("Invalid detection for COCO")
            detections.append({"image_id": image_id, "category_id": 1,
                               "bbox": [x, y, x2-x, y2-y], "score": float(score)})
    names = ["COCO_AP", "COCO_AP50", "COCO_AP75", "COCO_AP_small", "COCO_AP_medium", "COCO_AP_large"]
    if not images:
        return {**{key: None for key in names}, "COCO_AR100": None, "COCO_images": 0}
    with redirect_stdout(io.StringIO()):
        gt = COCO()
        gt.dataset = {"images": images, "annotations": annotations, "categories": [{"id": 1, "name": "spatter"}], "info": {}}
        gt.createIndex()
        if detections:
            dt = gt.loadRes(detections)
        else:
            dt = COCO()
            dt.dataset = {"images": images, "annotations": [], "categories": gt.dataset["categories"], "info": {}}
            dt.createIndex()
        evaluator = COCOeval(gt, dt, "bbox")
        evaluator.params.imgIds = [r["id"] for r in images]
        evaluator.params.catIds = [1]
        evaluator.params.maxDets = [1, 10, 100]
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    clean = lambda x: float(x) if x >= 0 and math.isfinite(float(x)) else None
    return {**{key: clean(v) for key, v in zip(names, evaluator.stats[:6])},
            "COCO_AR100": clean(evaluator.stats[8]), "COCO_images": len(images)}


@torch.no_grad()
def evaluate_model(model, dataset: RowsDataset, device) -> tuple[dict, list[dict], list[dict]]:
    model.eval()
    records = []
    expected_label = 0 if dataset.family == "retinanet" else 1
    for start in range(0, len(dataset), 4):
        batch = [dataset[i] for i in range(start, min(start + 4, len(dataset)))]
        outputs = model([image.to(device) for image, _, _ in batch])
        for out, (_, target, meta) in zip(outputs, batch):
            keep = out["labels"] == expected_label
            gt = target["boxes"][0].tolist()
            records.append({**meta, "width": 300, "height": 300, "gt_box": gt,
                            "boxes": out["boxes"][keep].detach().cpu().tolist(),
                            "scores": out["scores"][keep].detach().cpu().tolist(),
                            "labels": [1] * int(keep.sum()),
                            "area_group": "small" if float(target["area"][0]) < 1024 else "medium_large"})
    pooled = {**_historical_metrics(records), **official_coco_metrics(records)}
    groups = defaultdict(list)
    for r in records:
        if not r["specimen"]:
            raise ValueError("Missing specimen lineage")
        groups[r["specimen"]].append(r)
    specimens = []
    for specimen, current in sorted(groups.items()):
        metric = _historical_metrics(current)
        specimens.append({"specimen": specimen, "views": len(current), **metric,
                          "specimen_level_study_mAP_50_95": metric["study_mAP_50_95"]})
    return pooled, records, specimens


def weights_path(family: str) -> Path:
    c = common()
    filename = ("fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth" if family == "fasterrcnn"
                else "retinanet_resnet50_fpn_v2_coco-5905b1c5.pth")
    roots = [c.ARTIFACT_ROOT / "torch_cache" / "hub" / "checkpoints", Path(torch.hub.get_dir()) / "checkpoints",
             c.ARTIFACT_ROOT / "torch_cache" / "checkpoints", c.ARTIFACT_ROOT / "torch" / "hub" / "checkpoints"]
    for root in roots:
        if (root / filename).is_file():
            path = root / filename
            expected_prefix = filename.rsplit("-", 1)[1].split(".")[0]
            if not c.sha256_file(path).lower().startswith(expected_prefix):
                raise RuntimeError(f"Pretrained weights checksum prefix mismatch: {path}")
            return path
    raise FileNotFoundError(f"Pretrained {family} weights not cached; trainer never downloads weights: {filename}")


def build_model(family: str, pretrained: bool = True):
    from torchvision.models.detection import (fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights,
                                               retinanet_resnet50_fpn_v2, RetinaNet_ResNet50_FPN_V2_Weights)
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    p = common().load_protocol("T2" if family == "fasterrcnn" else "T6B")["model"]
    old_hub = torch.hub.get_dir()
    if pretrained:
        cached = weights_path(family)
        torch.hub.set_dir(str(cached.parent.parent))
    try:
        if family == "fasterrcnn":
            model = fasterrcnn_resnet50_fpn_v2(
                weights=FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1 if pretrained else None,
                weights_backbone=None, min_size=p["input_min_size"], max_size=p["input_max_size"],
                trainable_backbone_layers=p["trainable_backbone_layers"], box_score_thresh=p["score_threshold"],
                box_nms_thresh=p["nms_threshold"], box_detections_per_img=p["detections_per_image"])
            model.roi_heads.box_predictor = FastRCNNPredictor(model.roi_heads.box_predictor.cls_score.in_features, 2)
        elif family == "retinanet":
            model = retinanet_resnet50_fpn_v2(
                weights=RetinaNet_ResNet50_FPN_V2_Weights.COCO_V1 if pretrained else None,
                weights_backbone=None, min_size=p["input_min_size"], max_size=p["input_max_size"],
                trainable_backbone_layers=p["trainable_backbone_layers"], score_thresh=p["score_threshold"],
                nms_thresh=p["nms_threshold"], detections_per_img=p["detections_per_image"], topk_candidates=p["topk_candidates"])
            head = model.head.classification_head
            old = head.cls_logits
            head.cls_logits = torch.nn.Conv2d(old.in_channels, head.num_anchors, kernel_size=3, stride=1, padding=1)
            torch.nn.init.normal_(head.cls_logits.weight, std=.01)
            torch.nn.init.constant_(head.cls_logits.bias, -math.log((1-.01)/.01))
            head.num_classes = 1
            if model.head.regression_head._loss_type != "giou":
                raise RuntimeError("RetinaNet v2 GIoU contract broken")
        else:
            raise ValueError(f"Unknown detector family {family}")
    finally:
        torch.hub.set_dir(old_hub)
    return model


def _hash_json(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _rng_state() -> dict:
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def _restore_rng(state) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        torch.cuda.set_rng_state_all([v.cpu() for v in state["cuda"]])


def _atomic_torch(path: Path, payload, overwrite=False) -> str:
    c = common()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)
    digest = c.sha256_file(path)
    c.write_json(path.with_suffix(path.suffix + ".sha256.json"), {"sha256": digest}, overwrite=overwrite)
    return digest


def _save_predictions(path: Path, rows: list[dict]) -> str:
    c = common()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    # mtime=0 keeps identical prediction payloads byte-reproducible.
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as raw:
        with gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as z:
            z.write(json.dumps(rows, sort_keys=True, allow_nan=False).encode())
    tmp.replace(path)
    return c.sha256_file(path)


def _strict_provenance(task: str, config: dict, inputs: dict[str, Path]) -> dict:
    c = common()
    for task_id in ("T2", "T3B", "T5", "T6A", "T6B"):
        c.load_protocol(task_id)
    freeze = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    frozen_environment = c.REV_ROOT / "env" / "pip_freeze.txt"
    if not frozen_environment.is_file() or sorted(freeze.splitlines()) != sorted(frozen_environment.read_text(encoding="utf-8-sig").splitlines()):
        raise RuntimeError("Installed packages differ from frozen environment")
    sources = {"detector": Path(__file__), "common": Path(c.__file__),
               "historical_metrics": c.PROJECT_ROOT / "src" / "metal_spatter_pinn" / "detector.py"}
    strict = {"config": config, "inputs": {k: c.sha256_file(v) for k, v in sorted(inputs.items())},
              "source": {k: c.sha256_file(v) for k, v in sources.items()},
              "protocols": {t: c.sha256_file(c.REV_ROOT / "protocol" / f"REV02_{t}.yaml") for t in ("T2", "T3B", "T5", "T6A", "T6B")},
              "environment_sha256": hashlib.sha256(freeze.encode()).hexdigest(),
              "torch": torch.__version__, "torchvision": importlib.metadata.version("torchvision"),
              "pycocotools": importlib.metadata.version("pycocotools")}
    if strict["pycocotools"] != "2.0.11":
        raise RuntimeError("Frozen official evaluator requires pycocotools 2.0.11")
    return {"strict": strict, "strict_sha256": _hash_json(strict), "pip_freeze": freeze,
            "common_provenance": c.run_provenance(task, config, inputs)}


def _verify_files(entries: dict[str, str]) -> None:
    c = common()
    if not isinstance(entries, dict) or not entries:
        raise RuntimeError("An immutable receipt must contain a nonempty file-hash ledger")
    for value, digest in entries.items():
        if not isinstance(value, str) or not value or not isinstance(digest, str) or len(digest) != 64 or any(ch not in "0123456789abcdefABCDEF" for ch in digest):
            raise RuntimeError("Malformed immutable file-hash ledger")
        if not Path(value).is_file() or c.sha256_file(Path(value)).lower() != digest.lower():
            raise RuntimeError(f"Retained artifact missing or changed: {value}")


def _read_selections() -> dict[str, float]:
    c = common()
    root = c.task_dir("T6A")
    freeze = c.load_json(root / "PRETEST_SELECTION_FREEZE.json")
    _verify_files(freeze["files"])
    out = {arm: float(c.load_json(root / f"{arm}_ratio_selection.json")["selected_ratio"]) for arm in ("D2", "D4")}
    return out


def _load_training_rows(arm, ratio) -> tuple[list[dict], dict[str, Path]]:
    c = common()
    real = c.load_rows("train")
    expected = c.load_protocol("T2")["data_contract"]["expected_real_training_images"]
    if len(real) != expected or any(r["source_kind"] != "real" for r in real):
        raise RuntimeError("Real training roster does not match frozen contract")
    _validate_specimen_roster(real, 112)
    inputs = {"train_manifest": c.manifest_path("train"), "validation_manifest": c.manifest_path("validation")}
    if arm in {"D0", "D1"}:
        return real, inputs
    pool_id = {"D2": "G1", "D4": "G2", "D5": "D5"}[arm]
    path = c.pool_dir(pool_id) / "manifest.csv"
    pool_summary_path = c.pool_dir(pool_id) / "summary.json"
    summary = c.load_json(pool_summary_path)
    if summary.get("status") != "completed" or summary.get("box_mismatches") != 0:
        raise RuntimeError("Pool is incomplete or box reconciliation failed")
    if not summary.get("artifact_hashes") or not summary.get("ordered_triples_sha256"):
        raise RuntimeError("Pool integrity receipt is missing")
    _verify_files({str(c.pool_dir(pool_id)/relative): digest for relative, digest in summary["artifact_hashes"].items()})
    pool = c.read_csv(path)
    if len(pool) != 7168:
        raise RuntimeError("Incomplete synthetic pool")
    stride = Fraction(2, 1) / Fraction(str(ratio))
    if stride.denominator != 1:
        raise ValueError("Ratio is outside nested frozen grid")
    subset = pool[::int(stride)]
    if len(subset) != int(expected * ratio):
        raise RuntimeError("Synthetic subset has wrong size")
    seeds = {int(r["generator_seed"]) for r in subset}
    if seeds != set(c.load_protocol("T3B")["pool"]["generator_seeds"]):
        raise RuntimeError("Synthetic subset does not represent all frozen generator seeds")
    if any(r.get("source_kind") != "synthetic" for r in subset):
        raise RuntimeError("Synthetic provenance missing")
    inputs["pool_manifest"] = path
    inputs["pool_summary"] = pool_summary_path
    return real + subset, inputs


def _validate_specimen_roster(rows: list[dict], specimens: int) -> None:
    counts = defaultdict(list)
    for row in rows:
        counts[row["specimen"]].append(row["sample_id"])
    if len(counts) != specimens or any(len(ids) != 32 or len(set(ids)) != 32 or any(not value for value in ids) for ids in counts.values()):
        raise RuntimeError("Frozen real roster requires exactly 32 unique sample IDs per specimen; camera view categories may repeat")


def _validate_request(phase, family, arm, axis, ratio, seed):
    c = common()
    if (c.REV_ROOT / "T7_UNLOCK.json").exists():
        raise PermissionError("Training and ratio selection are permanently closed after T7 unlock")
    p = c.load_protocol("T6A" if phase == "ratio" else ("T2" if family == "fasterrcnn" else "T6B"))
    if seed not in p["detector_seeds"]:
        raise ValueError("Seed outside frozen roster")
    if phase == "ratio":
        if family != "fasterrcnn" or arm not in p["arms"] or ratio not in p["ratios"]:
            raise ValueError("Ratio cell outside frozen grid")
    else:
        allowed = p["arms"]
        if arm not in allowed or axis not in {"B1B3", "B2", "B4"}:
            raise ValueError("Unplanned budget cell")
        if family == "retinanet" and axis != "B4":
            raise ValueError("RetinaNet is B4-only")
        selected = _read_selections()
        allowed_ratios = [0.] if arm in {"D0", "D1"} else ([selected[arm]] if arm != "D5" else sorted(set(selected.values())))
        if ratio not in allowed_ratios:
            raise ValueError("Requested ratio differs from frozen validation selection")


def _schedule(phase, axis, nreal, nsyn, p2, p6a):
    if phase == "ratio":
        return {e: e * ((nreal + nsyn) // 4) for e in p6a["selection"]["epochs"]}
    labels = p2["budgets"]["B2"]["optimizer_step_milestones"]
    if axis == "B1B3":
        return {label: e * ((nreal + nsyn) // 4) for label, e in zip(labels, p2["budgets"]["B1_B3"]["epoch_milestones"])}
    return {label: label for label in labels}


def _output_evaluation(model, dataset, device, run_dir, label, split, metadata):
    c = common()
    destination = run_dir / split
    receipt = destination / f"milestone_{label}_receipt.json"
    if receipt.exists():
        prior = c.load_json(receipt)
        _verify_files(prior["files"])
        if prior["checkpoint_sha256"] != metadata["checkpoint_sha256"]:
            raise RuntimeError("Evaluation checkpoint identity changed")
        return prior
    # A partial test evaluation cannot silently rerun; coordinator must document recovery.
    started = destination / f"milestone_{label}_started.json"
    if started.exists():
        raise RuntimeError(f"Incomplete prior {split} evaluation; documented recovery required: {started}")
    c.write_json(started, {"started_utc": c.utc_now(), **metadata})
    rng = _rng_state()
    try:
        metrics, records, specimens = evaluate_model(model, dataset, device)
    finally:
        _restore_rng(rng)
    pred = destination / f"milestone_{label}_predictions.json.gz"
    pred_hash = _save_predictions(pred, records)
    rows = [{**metadata, "split": split, **r, "prediction_sha256": pred_hash} for r in specimens]
    spec_path = destination / f"milestone_{label}_specimens.csv"
    metric_path = destination / f"milestone_{label}_metrics.json"
    c.write_csv(spec_path, rows)
    c.write_json(metric_path, {**metadata, "split": split, **metrics, "prediction_sha256": pred_hash,
                               "specimen_count": len(specimens), "image_count": len(records)})
    result = {"completed_utc": c.utc_now(), "checkpoint_sha256": metadata["checkpoint_sha256"],
              "files": {str(pred): pred_hash, str(spec_path): c.sha256_file(spec_path), str(metric_path): c.sha256_file(metric_path)},
              "metrics": metrics, "specimens_file": str(spec_path), "metrics_file": str(metric_path)}
    c.write_json(receipt, result)
    return result


def _authorize_recovery(run_dir, old, new, receipt_path):
    c = common()
    if not receipt_path:
        raise RuntimeError("Incomplete run requires explicit --recovery-receipt; no automatic retraining/resume")
    record = c.load_json(Path(receipt_path))
    if record.get("run_id") != run_dir.name or record.get("old_provenance_sha256") != old["strict_sha256"]:
        raise RuntimeError("Recovery receipt does not identify prior run and provenance")
    a, b = old["strict"], new["strict"]
    for key in ("config", "inputs", "protocols", "environment_sha256"):
        if a[key] != b[key]:
            raise RuntimeError(f"Recovery cannot silently change {key}")
    if a["source"] != b["source"] and (not record.get("source_change_authorized") or record.get("new_provenance_sha256") != new["strict_sha256"]):
        raise RuntimeError("Changed code needs explicit documented source recovery")
    if not record.get("reason") or not record.get("pretest_amendment_sha256"):
        raise RuntimeError("Recovery requires a reason and pretest amendment hash")
    c.record_event("detector_recovery_authorized", {"run_id": run_dir.name, "receipt_sha256": c.sha256_file(Path(receipt_path))})


def train(phase, family, arm, axis, ratio, seed, recovery_receipt=None):
    c = common()
    _validate_request(phase, family, arm, axis, ratio, seed)
    p2, p6a = c.load_protocol("T2"), c.load_protocol("T6A")
    task = "T6A" if phase == "ratio" else ("T2" if family == "fasterrcnn" else "T6B")
    c.task_dir(task)
    if phase != "ratio" and family == "fasterrcnn" and arm in {"D0", "D1"}:
        axis = "B2"  # The unique shared trajectory has the step-clock schedule.
    run_id = ratio_run_id(arm, ratio, seed) if phase == "ratio" else budget_run_id(family, arm, axis, ratio, seed)
    directory = c.detector_run_dir(run_id)
    rows, inputs = _load_training_rows(arm, ratio)
    validation = c.load_rows("validation")
    if len(validation) != 1024 or len({r["specimen"] for r in validation}) != 32:
        raise RuntimeError("Validation roster differs from frozen 32-specimen contract")
    _validate_specimen_roster(validation, 32)
    inputs["pretrained_weights"] = weights_path(family)
    if phase != "ratio":
        inputs["selection_freeze"] = c.task_dir("T6A") / "PRETEST_SELECTION_FREEZE.json"
    config = {"phase": phase, "family": family, "arm": arm, "axis": axis, "ratio": ratio,
              "seed": seed, "run_id": run_id, "task": task}
    provenance = _strict_provenance(task, config, inputs)
    complete_path = directory / "completion.json"
    if complete_path.exists():
        completed = c.load_json(complete_path)
        if completed["provenance_sha256"] != provenance["strict_sha256"]:
            raise RuntimeError("Completed run differs from current code/environment/inputs; do not replace it")
        _verify_files(completed["files"])
        return completed
    old_path = directory / "provenance.json"
    resuming = old_path.exists()
    if resuming:
        _authorize_recovery(directory, c.load_json(old_path), provenance, recovery_receipt)
    if shutil.disk_usage(c.ARTIFACT_ROOT).free < 4_000_000_000:
        raise RuntimeError("Less than 4 GB free at retained checkpoint root")
    if not resuming:
        c.write_json(directory / "run_config.json", config)
        c.write_json(old_path, provenance)
        c.record_event("detector_run_started", {**config, "provenance_sha256": provenance["strict_sha256"]})
        c.append_run_log(task, f"Started {run_id}; validation-only, retained milestones, no test access.")
    c.seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Frozen GPU execution requires CUDA; no silent CPU substitution")
    torch.cuda.reset_peak_memory_stats()
    data = RowsDataset(rows, family, augment=arm == "D1")
    val_data = RowsDataset(validation, family)
    nreal = sum(r["source_kind"] == "real" for r in rows)
    nsyn = len(rows) - nreal
    sampler_axis = "B2" if phase == "ratio" else axis
    sampler = ExposureSampler(nreal, nsyn, ratio, sampler_axis, seed)
    schedule = _schedule(phase, axis, nreal, nsyn, p2, p6a)
    inverse = {v: k for k, v in schedule.items()}
    try:
        model = build_model(family).to(device)
    except BaseException as error:
        failure = {"run_id": run_id, "time_utc": c.utc_now(), "stage": "model_initialization", "error_type": type(error).__name__, "error": str(error)}
        c.write_json(directory / f"failure_{time.time_ns()}.json", failure)
        c.record_event("detector_run_failed", failure)
        torch.cuda.empty_cache()
        raise
    op = p2["optimization"]
    optimizer = torch.optim.SGD(model.parameters(), lr=op["learning_rate"], momentum=op["momentum"], weight_decay=op["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", enabled=op["amp"])
    lr_interval = 3 * ((nreal + nsyn) // 4) if phase == "ratio" or axis == "B1B3" else 2688
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_interval, gamma=.2)
    progress = {"step": 0, "real_seen": 0, "synthetic_seen": 0, "loss_sum": 0., "loss_count": 0,
                "training_elapsed_seconds": 0., "milestones": [], "evaluations": {}}
    if resuming:
        resume_path = directory / "resume.pt"
        _verify_files({str(resume_path): c.load_json(resume_path.with_suffix(".pt.sha256.json"))["sha256"]})
        state = torch.load(resume_path, map_location="cpu", weights_only=False)
        if state["provenance_sha256"] not in {c.load_json(old_path)["strict_sha256"], provenance["strict_sha256"]}:
            raise RuntimeError("Resume provenance mismatch")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        scheduler.load_state_dict(state["scheduler"])
        sampler.load_state_dict(state["sampler"])
        progress = state["progress"]
        _restore_rng(state["rng"])
        for milestone in progress["milestones"]:
            label = milestone["milestone"]
            evaluation_dir = directory / "validation"
            if (evaluation_dir / f"milestone_{label}_started.json").exists() and not (evaluation_dir / f"milestone_{label}_receipt.json").exists():
                raise RuntimeError("Prior validation evaluation was interrupted; train recovery alone cannot authorize re-evaluation. Preserve partial outputs and obtain a separate documented pretest evaluation recovery.")
    else:
        if nsyn:
            counts = defaultdict(int)
            for row in rows[nreal:]:
                counts[int(row["generator_seed"])] += 1
            c.write_csv(directory / "pool_subset_composition.csv", [{"generator_seed": s, "images": n} for s, n in sorted(counts.items())])
    def state_payload():
        return {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
                "scheduler": scheduler.state_dict(), "sampler": sampler.state_dict(), "rng": _rng_state(),
                "progress": progress, "epoch": progress["step"] // ((nreal + nsyn)//4),
                "provenance_sha256": provenance["strict_sha256"], "config": config}
    def save_resume():
        _atomic_torch(directory / "resume.pt", state_payload(), overwrite=True)
    if not resuming:
        save_resume()
    try:
        for step in range(progress["step"] + 1, max(schedule.values()) + 1):
            batch_started = time.perf_counter()
            indices = sampler.batch(step)
            batch = [data[index] for index in indices]
            real_count = sum(meta["source_kind"] == "real" for _, _, meta in batch)
            syn_count = len(batch) - real_count
            if axis == "B4" and (real_count != 4 or syn_count != synthetic_count(ratio, step)):
                raise RuntimeError("B4 exact per-step exposure check failed")
            model.train()
            images = [im.to(device) for im, _, _ in batch]
            targets = [{k: v.to(device) for k, v in target.items()} for _, target, _ in batch]
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=op["amp"]):
                losses = model(images, targets)
                loss = sum(losses.values())
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite detector loss")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), op["gradient_clip_norm"])
            if not torch.isfinite(norm):
                raise FloatingPointError("Nonfinite detector gradient norm; no skipped-update exposure drift")
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            progress.update(step=step, real_seen=progress["real_seen"]+real_count,
                            synthetic_seen=progress["synthetic_seen"]+syn_count,
                            loss_sum=progress["loss_sum"]+float(loss.detach()), loss_count=progress["loss_count"]+1,
                            training_elapsed_seconds=progress["training_elapsed_seconds"]+time.perf_counter()-batch_started)
            if step in inverse:
                label = inverse[step]
                if axis == "B4" and (progress["real_seen"] != 4*step or progress["synthetic_seen"] != (4*Fraction(str(ratio))*step).__floor__()):
                    raise RuntimeError("B4 milestone exposure check failed")
                checkpoint = directory / "checkpoints" / f"milestone_{label}.pt"
                metadata = {**config, "milestone": label, "optimizer_steps": step,
                            "real_images_seen": progress["real_seen"], "synthetic_images_seen": progress["synthetic_seen"],
                            "epoch_milestone": label if phase == "ratio" else [1, 2, 5, 10][list(schedule).index(label)],
                            "mean_loss": progress["loss_sum"]/progress["loss_count"], "learning_rate": optimizer.param_groups[0]["lr"],
                            "training_elapsed_seconds": progress["training_elapsed_seconds"],
                            "peak_cuda_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
                            "peak_cuda_memory_reserved_bytes": torch.cuda.max_memory_reserved()}
                digest = _atomic_torch(checkpoint, state_payload())
                entry = {**metadata, "checkpoint_path": str(checkpoint), "checkpoint_sha256": digest}
                progress["milestones"].append(entry)
                save_resume()
                result = _output_evaluation(model, val_data, device, directory, label, "validation", entry)
                progress["evaluations"][str(label)] = result
                progress["loss_sum"], progress["loss_count"] = 0., 0
                save_resume()
                print(f"{run_id}: milestone {label}, step {step}, real {progress['real_seen']}, synthetic {progress['synthetic_seen']}", flush=True)
            elif step % 256 == 0:
                save_resume()
        # A resumed checkpoint may end exactly at the final step before evaluation.
        for entry in progress["milestones"]:
            key = str(entry["milestone"])
            if key not in progress["evaluations"]:
                state = torch.load(entry["checkpoint_path"], map_location="cpu", weights_only=False)
                model.load_state_dict(state["model"])
                progress["evaluations"][key] = _output_evaluation(model, val_data, device, directory, entry["milestone"], "validation", entry)
        if len(progress["milestones"]) != len(schedule) or len(progress["evaluations"]) != len(schedule):
            raise RuntimeError("Incomplete fixed trajectory")
        files = {entry["checkpoint_path"]: entry["checkpoint_sha256"] for entry in progress["milestones"]}
        for entry in progress["evaluations"].values():
            files.update(entry["files"])
        files.update({str(directory / name): c.sha256_file(directory / name) for name in ("run_config.json", "provenance.json")})
        for path in (directory / "resume.pt", directory / "pool_subset_composition.csv"):
            if path.exists():
                files[str(path)] = c.sha256_file(path)
        complete = {**config, "status": "complete", "completed_utc": c.utc_now(), "test_loaded": False,
                    "finished_utc": c.utc_now(), "provenance_sha256": provenance["strict_sha256"], "milestones": progress["milestones"],
                    "evaluations": progress["evaluations"], "files": files}
        if phase == "ratio":
            candidates = [(int(k), float(v["metrics"]["study_mAP_50_95"])) for k, v in progress["evaluations"].items()]
            best = min(candidates, key=lambda v: (-v[1], v[0]))
            complete.update(best_epoch=best[0], validation_study_map_50_95=best[1])
        c.write_json(complete_path, complete)
        c.record_event("detector_run_completed", {"run_id": run_id, "completion_sha256": c.sha256_file(complete_path)})
        c.append_run_log(task, f"Completed {run_id}; all {len(schedule)} fixed milestones and validation outputs retained.")
        return complete
    except BaseException as error:
        failure = {"run_id": run_id, "time_utc": c.utc_now(), "step": progress["step"],
                   "error_type": type(error).__name__, "error": str(error), "completed_seed_replaced": False}
        c.write_json(directory / f"failure_{time.time_ns()}.json", failure)
        c.record_event("detector_run_failed", failure)
        c.append_run_log(task, f"FAILED {run_id} at step {progress['step']}: {type(error).__name__}; outputs retained and no seed replacement.")
        raise
    finally:
        del model, optimizer, scaler
        torch.cuda.empty_cache()


def select_grid(rows: list[dict], ratios: list[float], seeds: list[int], tolerance=.002) -> dict:
    indexed = {}
    for row in rows:
        key = (float(row["ratio"]), int(row["seed"]))
        if key in indexed:
            raise ValueError("Duplicate ratio-seed cell")
        value = float(row["validation_study_map_50_95"])
        if not math.isfinite(value):
            raise ValueError("Nonfinite ratio metric")
        indexed[key] = value
    if set(indexed) != {(float(r), int(s)) for r in ratios for s in seeds}:
        raise ValueError("Incomplete or extra ratio grid")
    means = {r: float(np.mean([indexed[(r, s)] for s in seeds])) for r in ratios}
    best = max(means.values())
    ties = [r for r in ratios if best - means[r] <= tolerance]
    return {"selected_ratio": min(ties), "best_mean": best, "tie_set": ties,
            "tolerance": tolerance, "candidates": [{"ratio": r, "mean": means[r], "per_seed": [indexed[(r, s)] for s in seeds]} for r in ratios]}


def select_ratios():
    c = common()
    if (c.REV_ROOT / "T7_UNLOCK.json").exists():
        raise PermissionError("Ratio selection is closed after global T7 unlock")
    p = c.load_protocol("T6A")
    root = c.task_dir("T6A")
    freeze_path = root / "PRETEST_SELECTION_FREEZE.json"
    if freeze_path.exists():
        frozen = c.load_json(freeze_path)
        _verify_files(frozen["files"])
        return frozen
    all_rows, epoch_rows, composition_rows, inputs = [], [], [], {}
    outputs = {}
    for arm in p["arms"]:
        rows = []
        for ratio in p["ratios"]:
            for seed in p["detector_seeds"]:
                path = c.detector_run_dir(ratio_run_id(arm, ratio, seed)) / "completion.json"
                value = c.load_json(path)
                if value.get("status") != "complete" or value.get("test_loaded") is not False:
                    raise RuntimeError("Invalid validation candidate receipt")
                _verify_files(value["files"])
                for epoch, evaluation in value["evaluations"].items():
                    epoch_rows.append({"arm": arm, "ratio": ratio, "seed": seed, "epoch": int(epoch),
                                       **c.load_json(Path(evaluation["metrics_file"]))})
                composition_rows.extend([{**r, "arm": arm, "ratio": ratio, "seed": seed} for r in c.read_csv(path.parent / "pool_subset_composition.csv")])
                row = {"arm": arm, "ratio": ratio, "seed": seed, "best_epoch": value["best_epoch"],
                       "validation_study_map_50_95": value["validation_study_map_50_95"]}
                rows.append(row)
                all_rows.append(row)
                inputs[str(path)] = c.sha256_file(path)
        selection = select_grid(rows, p["ratios"], p["detector_seeds"], p["selection"]["ratio_tie_tolerance_absolute"])
        selection.update(arm=arm, selection_split="validation", test_used=False, seed_rows=rows,
                         lower_boundary_selected=selection["selected_ratio"] == min(p["ratios"]), frozen_utc=c.utc_now())
        output = root / f"{arm}_ratio_selection.json"
        c.write_json(output, selection)
        outputs[str(output)] = c.sha256_file(output)
    grid = root / "raw" / "ratio_grid.csv"
    c.write_csv(grid, all_rows)
    outputs[str(grid)] = c.sha256_file(grid)
    for filename, rows in (("epoch_metrics.csv", epoch_rows), ("pool_subset_composition.csv", composition_rows)):
        path = root / "raw" / filename
        c.write_csv(path, rows)
        outputs[str(path)] = c.sha256_file(path)
    freeze = {"status": "complete", "cells": len(all_rows), "frozen_utc": c.utc_now(), "test_used": False,
              "files": {**inputs, **outputs}}
    c.write_json(freeze_path, freeze)
    c.write_task_summary("T6A", freeze)
    c.record_event("detector_ratios_frozen", {"sha256": c.sha256_file(freeze_path), "cells": len(all_rows)})
    c.append_run_log("T6A", "Frozen all 30 validation-only ratio cells with the three-seed mean and absolute 0.002 smaller-ratio tie rule.")
    return freeze


def evaluate_run(run_id: str, split: str):
    c = common()
    if split != "test":
        raise ValueError("Only the global T7 test evaluation is exposed; training already records validation")
    c.require_test_unlocked()
    if not run_id.startswith("budget_"):
        raise ValueError("Ratio-grid checkpoints must not be evaluated on test")
    directory = c.detector_run_dir(run_id)
    frozen = c.load_json(c.REV_ROOT / "PRETEST_FREEZE.json")
    planned = frozen["test_roster"]["detectors"]
    if not isinstance(planned, list) or any(not isinstance(r, dict) or not r.get("run_id") or not r.get("completion_sha256") for r in planned):
        raise PermissionError("T7 detector roster must bind each run ID to its immutable completion SHA-256")
    matching = [r for r in planned if r["run_id"] == run_id]
    if len(matching) != 1:
        raise PermissionError("Detector is absent from frozen T7 checkpoint roster")
    if c.sha256_file(directory / "completion.json") != matching[0]["completion_sha256"]:
        raise PermissionError("Detector completion differs from the frozen T7 roster")
    completed = c.load_json(directory / "completion.json")
    _verify_files(completed["files"])
    final = directory / "test_completion.json"
    if final.exists():
        prior = c.load_json(final)
        _verify_files(prior["files"])
        return prior
    rows = c.load_rows("test")
    if len(rows) != 1024 or len({r["specimen"] for r in rows}) != 32:
        raise RuntimeError("Test roster contract mismatch")
    _validate_specimen_roster(rows, 32)
    dataset = RowsDataset(rows, completed["family"])
    device = torch.device("cuda")
    model = build_model(completed["family"], pretrained=False).to(device)
    files, evaluations = {}, {}
    try:
        for entry in completed["milestones"]:
            state = torch.load(entry["checkpoint_path"], map_location="cpu", weights_only=False)
            model.load_state_dict(state["model"])
            receipt = _output_evaluation(model, dataset, device, directory, entry["milestone"], "test", entry)
            files.update(receipt["files"])
            evaluations[str(entry["milestone"])] = receipt
        result = {"status": "complete", "run_id": run_id, "split": "test", "completed_utc": c.utc_now(), "finished_utc": c.utc_now(),
                  "files": files, "evaluations": evaluations, "training_completion_sha256": c.sha256_file(directory / "completion.json")}
        c.write_json(final, result)
        c.record_event("detector_test_completed", {"run_id": run_id, "sha256": c.sha256_file(final)})
        return result
    finally:
        del model
        torch.cuda.empty_cache()


def expected_budget_roster(selections: dict, seeds: list[int]) -> list[dict]:
    rows = []
    union = sorted(set(selections.values()))
    for family in ("fasterrcnn", "retinanet"):
        axes = ("B1B3", "B2", "B4") if family == "fasterrcnn" else ("B4",)
        arms = ("D0", "D1", "D2", "D4", "D5") if family == "fasterrcnn" else ("D0", "D2", "D4", "D5")
        for axis in axes:
            for arm in arms:
                ratios = [0.] if arm in {"D0", "D1"} else (union if arm == "D5" else [selections[arm]])
                for ratio in ratios:
                    for seed in seeds:
                        rows.append({"family": family, "arm": arm, "axis": axis, "ratio": ratio, "seed": seed,
                                     "run_id": budget_run_id(family, arm, axis, ratio, seed)})
    return rows


def summarize():
    c = common()
    selections = _read_selections()
    roster = expected_budget_roster(selections, c.load_protocol("T2")["detector_seeds"])
    results = {}
    # Pretest summarization never probes the sealed manifest or attempts unlock.
    for family, task in (("fasterrcnn", "T2"), ("retinanet", "T6B")):
        destination = c.task_dir(task)
        current = [r for r in roster if r["family"] == family]
        all_specs, all_metrics, exposure, files = [], [], [], {}
        for row in current:
            directory = c.detector_run_dir(row["run_id"])
            path = directory / "completion.json"
            done = c.load_json(path)
            _verify_files(done["files"])
            files[str(path)] = c.sha256_file(path)
            if len(done["milestones"]) != 4:
                raise RuntimeError("Incomplete milestone roster")
            for milestone in done["milestones"]:
                label = milestone["milestone"]
                receipt = done["evaluations"][str(label)]
                spec = c.read_csv(Path(receipt["specimens_file"]))
                if len(spec) != 32 or len({r["specimen"] for r in spec}) != 32:
                    raise RuntimeError("Incomplete specimen matrix")
                all_specs.extend([{**r, **row, "split": "validation"} for r in spec])
                metric = c.load_json(Path(receipt["metrics_file"]))
                all_metrics.append({**metric, **row, "split": "validation"})
                exposure.append({**milestone, **row})
        c.write_csv(destination / "raw" / "trajectory_manifest.csv", current, overwrite=True)
        c.write_csv(destination / "raw" / "exposure_grid.csv", exposure, overwrite=True)
        c.write_csv(destination / "raw" / "specimen_metrics_validation.csv", all_specs, overwrite=True)
        c.write_csv(destination / "raw" / "validation_metrics.csv", all_metrics, overwrite=True)
        # Only read test prediction files if a verified global unlock exists.
        test_receipts = [c.detector_run_dir(r["run_id"]) / "test_completion.json" for r in current]
        test_complete = all(path.exists() for path in test_receipts)
        if test_complete:
            c.require_test_unlocked()
            test_specs, test_metrics = [], []
            for row, path in zip(current, test_receipts):
                receipt = c.load_json(path)
                _verify_files(receipt["files"])
                for evaluation in receipt["evaluations"].values():
                    test_specs.extend([{**r, **row, "split": "test"} for r in c.read_csv(Path(evaluation["specimens_file"]))])
                    test_metrics.append({**c.load_json(Path(evaluation["metrics_file"])), **row, "split": "test"})
            c.write_csv(destination / "raw" / "specimen_metrics_test.csv", test_specs, overwrite=True)
            c.write_csv(destination / "raw" / "test_metrics.csv", test_metrics, overwrite=True)
            aggregate_files = {str(destination / "raw" / name): c.sha256_file(destination / "raw" / name)
                               for name in ("specimen_metrics_test.csv", "test_metrics.csv")}
            source_files = {str(path): c.sha256_file(path) for path in test_receipts}
            aggregate_path = destination / "TEST_AGGREGATE_COMPLETE.json"
            aggregate = {"status": "completed", "task_id": task, "split": "test", "files": {**aggregate_files, **source_files}}
            if aggregate_path.exists():
                if c.load_json(aggregate_path) != aggregate:
                    raise RuntimeError("Immutable test aggregate receipt changed")
            else:
                c.write_json(aggregate_path, aggregate)
        summary = {"status": "pretest_complete" if not test_complete else "complete", "family": family,
                   "unique_trajectories": len({r["run_id"] for r in current}), "display_trajectory_rows": len(current),
                   "milestone_metric_rows": len(all_metrics), "specimen_metric_rows": len(all_specs),
                   "selected_ratios": selections, "test_status": "complete" if test_complete else "locked_or_pending",
                   "files": files, "completed_utc": c.utc_now(), "alias_replication": False}
        summary["validation_aggregate_files"] = {str(destination / "raw" / name): c.sha256_file(destination / "raw" / name)
                                                for name in ("trajectory_manifest.csv", "exposure_grid.csv", "specimen_metrics_validation.csv", "validation_metrics.csv")}
        validation_receipt = {"status": "completed", "task_id": task, "split": "validation",
                              "files": {**files, **summary["validation_aggregate_files"]}}
        validation_path = destination / "VALIDATION_COMPLETE.json"
        if validation_path.exists():
            if c.load_json(validation_path) != validation_receipt:
                raise RuntimeError("Immutable validation aggregate receipt changed")
        else:
            c.write_json(validation_path, validation_receipt)
        c.write_task_summary(task, summary)
        c.append_run_log(task, f"Reconciled {len(current)} display trajectories; {summary['unique_trajectories']} unique fits, four milestones each. Alias rows are not independent replications.")
        results[task] = summary
    return results


def pretest_evidence() -> dict:
    """Verify all detector rosters without probing any test manifest or outputs."""
    c = common()
    p6a, p2 = c.load_protocol("T6A"), c.load_protocol("T2")
    selection_root = c.task_dir("T6A")
    freeze_path = selection_root / "PRETEST_SELECTION_FREEZE.json"
    frozen = c.load_json(freeze_path)
    _verify_files(frozen["files"])
    if frozen.get("cells") != 30 or frozen.get("test_used") is not False:
        raise RuntimeError("Ratio freeze does not certify the complete validation-only grid")
    evidence = {}
    ratio_paths, ratio_ends, count = [freeze_path, *map(Path, frozen["files"])], [], 0
    for arm in p6a["arms"]:
        grid = []
        for ratio in p6a["ratios"]:
            for seed in p6a["detector_seeds"]:
                path = c.detector_run_dir(ratio_run_id(arm, ratio, seed)) / "completion.json"
                completed = c.load_json(path)
                _verify_files(completed["files"])
                if completed.get("status") != "complete" or completed.get("test_loaded") is not False or len(completed["milestones"]) != 2:
                    raise RuntimeError("Ratio candidate completion is invalid")
                expected = {"arm": arm, "ratio": ratio, "seed": seed, "phase": "ratio", "family": "fasterrcnn"}
                if any(completed.get(k) != v for k, v in expected.items()):
                    raise RuntimeError("Ratio receipt cell identity mismatch")
                grid.append(completed)
                count += 1
                ratio_paths.extend([path, *map(Path, completed["files"])])
                ratio_ends.append(completed["finished_utc"])
        recomputed = select_grid(grid, p6a["ratios"], p6a["detector_seeds"], p6a["selection"]["ratio_tie_tolerance_absolute"])
        locked_path = selection_root / f"{arm}_ratio_selection.json"
        if c.load_json(locked_path)["selected_ratio"] != recomputed["selected_ratio"]:
            raise RuntimeError("Ratio selection does not reproduce from every frozen seed")
        ratio_paths.append(locked_path)
    evidence["T6A"] = {"task_id": "T6A", "status": "pretest_complete", "expected_units": 30,
                        "completed_units": count, "missing_units": [], "integrity_checks_passed": True,
                        "last_training_validation_end_utc": max(ratio_ends), "output_paths": sorted({str(p) for p in ratio_paths})}
    roster = expected_budget_roster(_read_selections(), p2["detector_seeds"])
    for family, task in (("fasterrcnn", "T2"), ("retinanet", "T6B")):
        unique = {row["run_id"]: row for row in roster if row["family"] == family}
        paths, endings = [], []
        for run_id, row in unique.items():
            path = c.detector_run_dir(run_id) / "completion.json"
            completed = c.load_json(path)
            _verify_files(completed["files"])
            if completed.get("status") != "complete" or completed.get("test_loaded") is not False:
                raise RuntimeError("Budget receipt is not validation-only complete")
            if {m["milestone"] for m in completed["milestones"]} != {896,1792,4480,8960}:
                raise RuntimeError("Budget milestone roster is not exact")
            for key in ("family", "arm", "ratio", "seed", "run_id"):
                if completed.get(key) != row[key]:
                    raise RuntimeError("Budget receipt identity mismatch")
            for milestone in completed["milestones"]:
                evaluation = completed["evaluations"][str(milestone["milestone"])]
                specs = c.read_csv(Path(evaluation["specimens_file"]))
                if len(specs) != 32 or len({s["specimen"] for s in specs}) != 32:
                    raise RuntimeError("Missing specimen metrics")
                if any(int(s["views"]) != 32 or not math.isfinite(float(s["specimen_level_study_mAP_50_95"])) for s in specs):
                    raise RuntimeError("Nonfinite primary specimen metric")
                if completed["axis"] == "B4":
                    s = milestone["optimizer_steps"]
                    if milestone["real_images_seen"] != 4*s or milestone["synthetic_images_seen"] != int(4*Fraction(str(row["ratio"]))*s):
                        raise RuntimeError("Exact B4 exposure receipt failure")
            paths.extend([path, *map(Path, completed["files"])])
            endings.append(completed["finished_utc"])
        local = c.task_dir(task)
        validation_receipt_path = local / "VALIDATION_COMPLETE.json"
        validation_receipt = c.load_json(validation_receipt_path)
        if validation_receipt.get("status") != "completed" or validation_receipt.get("split") != "validation":
            raise RuntimeError("Immutable validation aggregate receipt missing")
        _verify_files(validation_receipt["files"])
        paths.append(validation_receipt_path)
        for name in ("trajectory_manifest.csv", "exposure_grid.csv", "specimen_metrics_validation.csv", "validation_metrics.csv"):
            path = local / "raw" / name
            if not path.is_file():
                raise RuntimeError("Run detector summarize before pretest evidence")
            paths.append(path)
        evidence[task] = {"task_id": task, "status": "pretest_complete", "expected_units": len(unique)*4,
                          "completed_units": len(unique)*4, "missing_units": [], "integrity_checks_passed": True,
                          "last_training_validation_end_utc": max(endings), "output_paths": sorted({str(p) for p in paths})}
    return evidence


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    ratio = commands.add_parser("ratio")
    ratio.add_argument("--arm", choices=["D2", "D4"], required=True)
    ratio.add_argument("--ratio", type=float, required=True)
    ratio.add_argument("--seed", type=int, required=True)
    ratio.add_argument("--recovery-receipt", type=Path)
    commands.add_parser("select-ratios")
    budget = commands.add_parser("budget")
    budget.add_argument("--family", choices=["fasterrcnn", "retinanet"], required=True)
    budget.add_argument("--arm", choices=["D0", "D1", "D2", "D4", "D5"], required=True)
    budget.add_argument("--axis", choices=["B1B3", "B2", "B4"], required=True)
    budget.add_argument("--ratio", type=float, required=True)
    budget.add_argument("--seed", type=int, required=True)
    budget.add_argument("--recovery-receipt", type=Path)
    evaluation = commands.add_parser("evaluate")
    evaluation.add_argument("--run-id", required=True)
    evaluation.add_argument("--split", choices=["test"], required=True)
    commands.add_parser("summarize")
    args = parser.parse_args(argv)
    if args.command == "ratio":
        return train("ratio", "fasterrcnn", args.arm, "B1B3", args.ratio, args.seed, args.recovery_receipt)
    if args.command == "budget":
        return train("budget", args.family, args.arm, args.axis, args.ratio, args.seed, args.recovery_receipt)
    if args.command == "select-ratios":
        return select_ratios()
    if args.command == "evaluate":
        return evaluate_run(args.run_id, args.split)
    return summarize()


if __name__ == "__main__":
    main()
