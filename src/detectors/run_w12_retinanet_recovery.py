"""Resume W12 secondary RetinaNet cells after invalid-output COCO recovery.

This is intentionally a wrapper, rather than a modification to the frozen W12
runner.  That keeps existing W12 run identities and checkpoints immutable while
making invalid detector outputs explicit, auditable non-detections.
"""
from __future__ import annotations

import argparse
import math
from typing import Any

import torch

import run_w12_detectors as w12


RECOVERY_ID = "PROTOCOL_AMENDMENT_003_W12_RETINANET_INVALID_PREDICTIONS"
ORIGINAL_PREDICT = w12.predict


def sanitize_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove only detections that cannot be expressed as valid COCO boxes."""
    cleaned: list[dict[str, Any]] = []
    for record in records:
        boxes = record["boxes"]
        scores = record["scores"]
        labels = record["labels"]
        if not (len(boxes) == len(scores) == len(labels)):
            raise ValueError("Prediction columns have unequal lengths before recovery")
        valid_boxes: list[list[float]] = []
        valid_scores: list[float] = []
        invalid = 0
        for box, score, label in zip(boxes, scores, labels):
            if int(label) != 1:
                raise ValueError("Noncanonical foreground mapping in RetinaNet recovery")
            x1, y1, x2, y2 = map(float, box)
            numeric = (x1, y1, x2, y2, float(score))
            if not all(math.isfinite(value) for value in numeric) or x2 <= x1 or y2 <= y1:
                invalid += 1
                continue
            valid_boxes.append([x1, y1, x2, y2])
            valid_scores.append(float(score))
        cleaned.append({
            **record,
            "boxes": valid_boxes,
            "scores": valid_scores,
            "labels": [1] * len(valid_boxes),
            "prediction_sanitization": {
                "recovery_id": RECOVERY_ID,
                "candidate_detections": len(boxes),
                "invalid_detections_discarded": invalid,
            },
        })
    return cleaned


@torch.inference_mode()
def recovered_predict(
    model: torch.nn.Module,
    dataset: w12.DetectorRows,
    device: torch.device,
    batch_size: int = 4,
) -> list[dict[str, Any]]:
    if dataset.family != "retinanet":
        return ORIGINAL_PREDICT(model, dataset, device, batch_size)
    return sanitize_records(ORIGINAL_PREDICT(model, dataset, device, batch_size))


def verify_recovery_binding() -> None:
    config = w12.load_config()
    for spec in w12.trajectory_roster(config):
        if spec["block"] != "secondary" or spec["family"] != "retinanet":
            continue
        identity_path = w12.RUN_ROOT / spec["run_id"] / "identity.json"
        if not identity_path.exists():
            continue
        identity = w12.json.loads(identity_path.read_text(encoding="utf-8"))
        if identity["source_sha256"] != w12.sha256_file(w12.Path(w12.__file__)):
            raise RuntimeError("Frozen W12 source differs from resumable RetinaNet identity")
        common_path = w12.ROOT / "src" / "detectors" / "detector_common.py"
        if identity["common_source_sha256"] != w12.sha256_file(common_path):
            raise RuntimeError("Frozen common detector source differs from resumable RetinaNet identity")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-all", action="store_true")
    parser.add_argument("--session-hours", type=float, default=10.5)
    args = parser.parse_args()
    if not args.run_all:
        parser.error("--run-all is required")
    verify_recovery_binding()
    w12.predict = recovered_predict
    result = w12.run_all(args.session_hours)
    print(w12.json.dumps({"recovery_id": RECOVERY_ID, "W12": result}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
