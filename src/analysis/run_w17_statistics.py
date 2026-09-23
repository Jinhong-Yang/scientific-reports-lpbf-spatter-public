"""Run the outcome-frozen W17 primary and secondary statistical analyses."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import gzip
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "vendor" / "parquet"))
import pandas as pd  # noqa: E402

sys.path.insert(0, str(ROOT / "src" / "generators"))
from run_w09_factorial import atomic_json, sha256_file, verify_protocol  # noqa: E402

sys.path.insert(0, str(ROOT / "src" / "detectors"))
from detector_common import official_coco_metrics  # noqa: E402

CONFIG_PATH = ROOT / "configs" / "w17_statistics.json"
TEST_ROOT = ROOT / "runs" / "new_study" / "W17_test_campaign"
GROUPED_ROOT = ROOT / "runs" / "new_study" / "W16_grouped_detectors"
EVIDENCE_ROOT = ROOT / "evidence" / "final_statistics"

IOU_THRESHOLDS = np.linspace(0.50, 0.95, 10)
RECALL_THRESHOLDS = np.linspace(0.0, 1.0, 101)
PRIMARY_BOOTSTRAP_WORKERS = 10


def load_config() -> dict[str, Any]:
    value = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if value["status"] != "FROZEN_BEFORE_FIRST_N2_TEST_OUTCOME":
        raise RuntimeError("W17 statistics configuration is not outcome-frozen")
    if value["bootstrap_draws"] != 20000 or value["simultaneous_individual_interval"] != 0.9875:
        raise RuntimeError("W17 primary interval contract differs")
    return value


def read_gzip_json(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def box_iou(boxes: np.ndarray, target: np.ndarray) -> np.ndarray:
    if len(boxes) == 0:
        return np.empty(0, dtype=np.float64)
    intersection_x1 = np.maximum(boxes[:, 0], target[0])
    intersection_y1 = np.maximum(boxes[:, 1], target[1])
    intersection_x2 = np.minimum(boxes[:, 2], target[2])
    intersection_y2 = np.minimum(boxes[:, 3], target[3])
    intersection = np.maximum(intersection_x2 - intersection_x1, 0) * np.maximum(intersection_y2 - intersection_y1, 0)
    box_area = np.maximum(boxes[:, 2] - boxes[:, 0], 0) * np.maximum(boxes[:, 3] - boxes[:, 1], 0)
    target_area = max(float((target[2] - target[0]) * (target[3] - target[1])), 0.0)
    return intersection / np.maximum(box_area + target_area - intersection, np.spacing(1))


def prepare_specimen_blocks(records: list[dict[str, Any]]) -> tuple[list[str], dict[str, dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        if len(record.get("gt_boxes", [])) != 1:
            raise RuntimeError("Fast COCO engine requires exactly one inherited box per image")
        grouped.setdefault(str(record["specimen"]), []).append(record)
    order = sorted(grouped)
    blocks: dict[str, dict[str, Any]] = {}
    for specimen in order:
        scores_parts, tp_parts = [], []
        values = grouped[specimen]
        for record in values:
            scores = np.asarray(record["scores"], dtype=np.float64)
            boxes = np.asarray(record["boxes"], dtype=np.float64).reshape(-1, 4)
            if len(scores) != len(boxes) or len(scores) > 100:
                raise RuntimeError("Prediction count differs from COCO maxDet contract")
            index = np.argsort(-scores, kind="mergesort")
            scores = scores[index]
            overlaps = box_iou(boxes[index], np.asarray(record["gt_boxes"][0], dtype=np.float64))
            tp = np.zeros((len(scores), len(IOU_THRESHOLDS)), dtype=np.uint8)
            for threshold_index, threshold in enumerate(IOU_THRESHOLDS):
                eligible = np.flatnonzero(overlaps >= threshold)
                if len(eligible):
                    tp[eligible[0], threshold_index] = 1
            scores_parts.append(scores)
            tp_parts.append(tp)
        blocks[specimen] = {
            "scores": np.concatenate(scores_parts) if scores_parts else np.empty(0, dtype=np.float64),
            "tp": np.concatenate(tp_parts) if tp_parts else np.empty((0, len(IOU_THRESHOLDS)), dtype=np.uint8),
            "ground_truths": len(values),
        }
    return order, blocks


def fast_coco_ap(specimen_order: list[str], blocks: dict[str, dict[str, Any]], sampled_indices: np.ndarray) -> float:
    selected = [blocks[specimen_order[int(index)]] for index in sampled_indices]
    ground_truths = sum(int(block["ground_truths"]) for block in selected)
    scores = np.concatenate([block["scores"] for block in selected])
    if not len(scores):
        return 0.0
    tp = np.concatenate([block["tp"] for block in selected], axis=0)
    index = np.argsort(-scores, kind="mergesort")
    tp = tp[index].astype(np.float64)
    values = []
    for threshold_index in range(len(IOU_THRESHOLDS)):
        true_positive = np.cumsum(tp[:, threshold_index])
        false_positive = np.cumsum(1.0 - tp[:, threshold_index])
        recall = true_positive / ground_truths
        precision = true_positive / (true_positive + false_positive + np.spacing(1))
        for position in range(len(precision) - 1, 0, -1):
            if precision[position] > precision[position - 1]:
                precision[position - 1] = precision[position]
        sampled = np.zeros(len(RECALL_THRESHOLDS), dtype=np.float64)
        locations = np.searchsorted(recall, RECALL_THRESHOLDS, side="left")
        valid = locations < len(precision)
        sampled[valid] = precision[locations[valid]]
        values.append(float(sampled.mean()))
    return float(np.mean(values))


def bootstrap_cell_values(specimen_order: list[str], blocks: dict[str, dict[str, Any]],
                          specimen_draws: np.ndarray) -> np.ndarray:
    """Evaluate one arm/seed cell for every frozen specimen draw."""
    return np.asarray([fast_coco_ap(specimen_order, blocks, draw) for draw in specimen_draws], dtype=np.float64)


def bootstrap_cell_worker(task: tuple[str, int, str, np.ndarray, list[str]]) -> tuple[str, int, np.ndarray]:
    """Load one prediction cell in a worker and preserve the shared draw order."""
    arm, seed, relative_path, specimen_draws, reference_order = task
    order, blocks = prepare_specimen_blocks(read_gzip_json(ROOT / relative_path))
    if order != reference_order:
        raise RuntimeError(f"Primary prediction specimen roster differs for {arm} seed {seed}")
    return arm, seed, bootstrap_cell_values(order, blocks, specimen_draws)


def combine_pipeline_bootstrap(cell_samples: dict[str, dict[int, np.ndarray]], seeds: list[int],
                               replicate_draws: np.ndarray, contrast_names: list[str]) -> np.ndarray:
    """Apply paired pipeline-resample counts to precomputed specimen-resample AP."""
    counts = np.column_stack([(replicate_draws == index).sum(axis=1) for index in range(len(seeds))])
    means: dict[str, np.ndarray] = {}
    for arm, values in cell_samples.items():
        matrix = np.column_stack([values[seed] for seed in seeds])
        means[arm] = (matrix * counts).sum(axis=1) / len(seeds)
    output = np.empty((len(replicate_draws), len(contrast_names)), dtype=np.float64)
    for index, name in enumerate(contrast_names):
        left, right = name.split("-")
        output[:, index] = means[left] - means[right]
    return output


def primary_records(config: dict[str, Any]) -> tuple[dict[str, dict[int, tuple[list[str], dict[str, dict[str, Any]]]]], pd.DataFrame]:
    manifest_path = ROOT / "evidence" / "test_campaign" / "test_metrics.parquet"
    manifest = pd.read_parquet(manifest_path)
    primary = manifest[(manifest["source_block"] == "W12_core") & (manifest["family"] == "fasterrcnn")]
    cells: dict[str, dict[int, tuple[list[str], dict[str, dict[str, Any]]]]] = {}
    for arm in config["primary_arms"]:
        cells[arm] = {}
        arm_frame = primary[primary["arm"] == arm]
        if len(arm_frame) != 10:
            raise RuntimeError(f"Primary arm {arm} does not contain ten test cells")
        for row in arm_frame.to_dict("records"):
            path = TEST_ROOT / "W12_core" / row["run_id"] / "predictions.json.gz"
            cells[arm][int(row["detector_seed"])] = prepare_specimen_blocks(read_gzip_json(path))
    return cells, primary


def primary_bootstrap(config: dict[str, Any]) -> tuple[pd.DataFrame, np.ndarray, dict[str, Any]]:
    cells, primary = primary_records(config)
    seeds = sorted(cells["NP"])
    reference_order = cells["NP"][seeds[0]][0]
    if len(reference_order) != 32:
        raise RuntimeError("Primary bootstrap requires 32 test specimens")
    maximum_engine_error = 0.0
    for arm in config["primary_arms"]:
        for seed in seeds:
            order, blocks = cells[arm][seed]
            if order != reference_order:
                raise RuntimeError("Primary prediction specimen rosters are not paired")
            fast = fast_coco_ap(order, blocks, np.arange(len(order)))
            official = float(primary[(primary["arm"] == arm) & (primary["detector_seed"] == seed)].iloc[0]["test_COCO_AP"])
            maximum_engine_error = max(maximum_engine_error, abs(fast - official))
    if maximum_engine_error > float(config["fast_engine_absolute_tolerance"]):
        raise RuntimeError(f"Fast pooled AP engine differs from pycocotools by {maximum_engine_error}")
    rng = np.random.default_rng(int(config["bootstrap_seed"]))
    draws = int(config["bootstrap_draws"])
    specimen_draws = rng.integers(0, len(reference_order), size=(draws, len(reference_order)), dtype=np.int16)
    replicate_draws = rng.integers(0, len(seeds), size=(draws, len(seeds)), dtype=np.int8)
    contrast_names = config["primary_contrasts"]
    tasks = []
    for arm in config["primary_arms"]:
        for seed in seeds:
            run_id = primary[(primary["arm"] == arm) & (primary["detector_seed"] == seed)].iloc[0]["run_id"]
            prediction_path = TEST_ROOT / "W12_core" / run_id / "predictions.json.gz"
            relative_path = str(prediction_path.relative_to(ROOT)).replace("\\", "/")
            tasks.append((arm, seed, relative_path, specimen_draws, reference_order))
    del cells
    cell_samples: dict[str, dict[int, np.ndarray]] = {arm: {} for arm in config["primary_arms"]}
    workers = min(PRIMARY_BOOTSTRAP_WORKERS, len(tasks))
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(bootstrap_cell_worker, task) for task in tasks]
        for completed, future in enumerate(as_completed(futures), start=1):
            arm, seed, values = future.result()
            cell_samples[arm][seed] = values
            print(json.dumps({"W17_primary_bootstrap_cells_complete": completed,
                              "cells_total": len(tasks), "arm": arm, "seed": seed}), flush=True)
    samples = combine_pipeline_bootstrap(cell_samples, seeds, replicate_draws, contrast_names)
    point = primary.groupby("arm")["test_COCO_AP"].mean().to_dict()
    rows = []
    for index, name in enumerate(contrast_names):
        left, right = name.split("-")
        values = samples[:, index]
        ordinary_alpha = 1 - float(config["ordinary_interval"])
        simultaneous_alpha = 1 - float(config["simultaneous_individual_interval"])
        rows.append({
            "analysis": "primary_held_out_test",
            "contrast": name,
            "estimate": float(point[left] - point[right]),
            "ordinary_low": float(np.quantile(values, ordinary_alpha / 2)),
            "ordinary_high": float(np.quantile(values, 1 - ordinary_alpha / 2)),
            "ordinary_confidence": config["ordinary_interval"],
            "simultaneous_low": float(np.quantile(values, simultaneous_alpha / 2)),
            "simultaneous_high": float(np.quantile(values, 1 - simultaneous_alpha / 2)),
            "simultaneous_individual_confidence": config["simultaneous_individual_interval"],
            "draws": draws,
            "specimens": len(reference_order),
            "pipeline_replicates": len(seeds),
            "resampling": config["resampling"],
            "equivalence_or_noninferiority_claim": "PROHIBITED",
        })
    diagnostics = {
        "maximum_fast_engine_absolute_error_vs_pycocotools": maximum_engine_error,
        "verified_primary_cells": 50,
        "bootstrap_seed": config["bootstrap_seed"],
        "bootstrap_draws": draws,
        "parallel_cells": len(tasks),
        "parallel_workers": workers,
        "parallelization_changes_estimator": False,
    }
    return pd.DataFrame(rows), samples, diagnostics


def paired_value_bootstrap(values: np.ndarray, draws: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(draws, len(values)))
    means = values[indices].mean(axis=1)
    return tuple(float(value) for value in np.quantile(means, [0.025, 0.975]))


def low_data_statistics(config: dict[str, Any]) -> pd.DataFrame:
    frame = pd.read_parquet(ROOT / "evidence" / "test_campaign" / "test_metrics.parquet")
    frame = frame[frame["source_block"] == "W14_low_data"]
    rows = []
    for size in sorted(frame["subset_size"].dropna().unique()):
        subset = frame[frame["subset_size"] == size]
        indexed = subset.set_index(["realization", "pipeline_seed", "arm"])["test_COCO_AP"]
        for contrast_index, right in enumerate(("NB", "N1", "NR")):
            differences = []
            for realization in sorted(subset["realization"].unique()):
                for seed in sorted(subset["pipeline_seed"].unique()):
                    differences.append(float(indexed[(realization, seed, "NP")] - indexed[(realization, seed, right)]))
            values = np.asarray(differences)
            low, high = paired_value_bootstrap(values, int(config["low_data_draws"]), int(config["low_data_bootstrap_seed"]) + int(size) * 10 + contrast_index)
            rows.append({
                "analysis": "low_data_held_out_test",
                "subset_size": int(size),
                "contrast": f"NP-{right}",
                "estimate": float(values.mean()),
                "ordinary_low": low,
                "ordinary_high": high,
                "ordinary_confidence": 0.95,
                "paired_realization_pipeline_cells": len(values),
                "draws": config["low_data_draws"],
                "status": "secondary_descriptive",
            })
    return pd.DataFrame(rows)


def grouped_oof_statistics() -> pd.DataFrame:
    rows = []
    for seed in (64001, 64002, 64003):
        for arm in ("N1", "NR", "NB", "NP"):
            records = []
            for fold in (1, 2, 3):
                path = GROUPED_ROOT / f"seed{seed}_fold{fold}_{arm}" / "validation" / "milestone_8960" / "predictions.json.gz"
                records.extend(read_gzip_json(path))
            if len(records) != 4608 or len({row["specimen"] for row in records}) != 144:
                raise RuntimeError("W16 concatenated OOF roster differs")
            metrics = official_coco_metrics(records)
            rows.append({"partition_seed": seed, "arm": arm, **metrics})
    frame = pd.DataFrame(rows)
    output = []
    indexed = frame.set_index(["partition_seed", "arm"])["COCO_AP"]
    for right in ("NB", "N1", "NR"):
        values = np.asarray([indexed[(seed, "NP")] - indexed[(seed, right)] for seed in (64001, 64002, 64003)], dtype=float)
        output.append({
            "analysis": "development_grouped_oof",
            "contrast": f"NP-{right}",
            "estimate": float(values.mean()),
            "minimum_partition_effect": float(values.min()),
            "maximum_partition_effect": float(values.max()),
            "partition_seeds": 3,
            "development_specimens": 144,
            "status": "secondary_internal_descriptive_no_external_validity",
        })
    EVIDENCE_ROOT.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(EVIDENCE_ROOT / "grouped_oof_metrics.parquet", index=False)
    return pd.DataFrame(output)


def failure_ledger() -> pd.DataFrame:
    rows = []
    for path in sorted((ROOT / "runs" / "new_study").rglob("failure*.json")):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except Exception as error:
            value = {"status": "UNREADABLE_FAILURE_RECORD", "error": str(error)}
        rows.append({
            "path": str(path.relative_to(ROOT)).replace("\\", "/"),
            "sha256": sha256_file(path),
            "status": value.get("status"),
            "run_id": value.get("run_id"),
            "error_type": value.get("error_type"),
            "error": value.get("error"),
            "seed_replaced": value.get("seed_replaced", False),
            "test_payload_accessed": value.get("test_payload_accessed", False),
        })
    return pd.DataFrame(rows)


def claim_matrix(primary: pd.DataFrame, low: pd.DataFrame, grouped: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for record in primary.to_dict("records"):
        rows.append({
            "claim_id": f"PRIMARY_{record['contrast'].replace('-', '_minus_')}",
            "claim": f"Held-out mean pooled COCO AP difference {record['contrast']}",
            "evidence": "evidence/final_statistics/primary_comparisons.parquet",
            "value": record["estimate"],
            "allowed_scope": "internal held-out operational inherited_bbox_region target",
            "prohibited_extension": "external validity physical particle semantics equivalence or noninferiority",
        })
    rows.extend([
        {
            "claim_id": "LOW_DATA_ROBUSTNESS", "claim": "Low-data effects are secondary across three subset realizations and pipeline seeds",
            "evidence": "evidence/final_statistics/low_data_comparisons.parquet", "value": None,
            "allowed_scope": "internal sensitivity", "prohibited_extension": "universal data efficiency claim",
        },
        {
            "claim_id": "GROUPED_OOF", "claim": "Development-cohort grouped OOF robustness",
            "evidence": "evidence/final_statistics/grouped_oof_comparisons.parquet", "value": None,
            "allowed_scope": "144-specimen internal development cohort", "prohibited_extension": "independent-build or external validation",
        },
        {
            "claim_id": "LABEL_SCOPE", "claim": "Generated arrays align with inherited operational boxes",
            "evidence": "evidence/stronger_generator/NEW_POOL_LABEL_VALIDITY.json", "value": None,
            "allowed_scope": "inherited_bbox_region", "prohibited_extension": "human-verified particle streak spark or cluster semantics",
        },
    ])
    return pd.DataFrame(rows)


def preflight() -> dict[str, Any]:
    config = load_config()
    _, protocol_hash = verify_protocol()
    test_receipt = ROOT / "evidence" / "test_campaign" / "W17_TEST_COMPLETENESS.json"
    grouped_receipt = ROOT / "evidence" / "grouped_internal" / "W16_DETECTOR_COMPLETENESS.json"
    checks = {
        "test_complete": test_receipt.exists() and json.loads(test_receipt.read_text(encoding="utf-8"))["status"] == "COMPLETE",
        "grouped_complete": grouped_receipt.exists() and json.loads(grouped_receipt.read_text(encoding="utf-8"))["status"] == "COMPLETE",
        "primary_family_exact": config["primary_contrasts"] == ["NP-NB", "NP-NF", "NP-N1", "NP-NR"],
        "draws_20000": config["bootstrap_draws"] == 20000,
    }
    return {
        "schema_version": 1,
        "status": "PASS" if all(checks.values()) else "WAITING_FOR_RESULTS" if all(value for key, value in checks.items() if key not in {"test_complete", "grouped_complete"}) else "FAIL",
        "checks": checks,
        "protocol_sha256": protocol_hash,
        "config_sha256": sha256_file(CONFIG_PATH),
    }


def run() -> dict[str, Any]:
    gate = preflight()
    if gate["status"] != "PASS":
        raise RuntimeError(f"W17 statistics preflight is {gate['status']}")
    config = load_config()
    EVIDENCE_ROOT.mkdir(parents=True, exist_ok=True)
    primary, samples, diagnostics = primary_bootstrap(config)
    low = low_data_statistics(config)
    grouped = grouped_oof_statistics()
    failures = failure_ledger()
    claims = claim_matrix(primary, low, grouped)
    paths = {
        "primary_comparisons.parquet": primary,
        "low_data_comparisons.parquet": low,
        "grouped_oof_comparisons.parquet": grouped,
        "ALL_COMPARISONS.parquet": pd.concat([primary, low, grouped], ignore_index=True, sort=False),
    }
    for name, frame in paths.items():
        frame.to_parquet(EVIDENCE_ROOT / name, index=False)
    failures.to_csv(EVIDENCE_ROOT / "FAILURE_LEDGER.csv", index=False)
    claims.to_csv(EVIDENCE_ROOT / "CLAIM_EVIDENCE_MATRIX.csv", index=False)
    np.savez_compressed(EVIDENCE_ROOT / "primary_bootstrap_samples.npz", contrasts=np.asarray(config["primary_contrasts"]), samples=samples)
    artifact_paths = [*(EVIDENCE_ROOT / name for name in paths), EVIDENCE_ROOT / "FAILURE_LEDGER.csv", EVIDENCE_ROOT / "CLAIM_EVIDENCE_MATRIX.csv", EVIDENCE_ROOT / "primary_bootstrap_samples.npz", EVIDENCE_ROOT / "grouped_oof_metrics.parquet"]
    receipt = {
        "schema_version": 1,
        "status": "PASS",
        "primary_comparisons": len(primary),
        "low_data_comparisons": len(low),
        "grouped_oof_comparisons": len(grouped),
        "failure_records": len(failures),
        "diagnostics": diagnostics,
        "scope": "internal_protocol_frozen_operational_target",
        "external_validity_claim": "PROHIBITED",
        "equivalence_or_noninferiority_claim": "PROHIBITED",
        "artifacts": {str(path.relative_to(ROOT)).replace("\\", "/"): sha256_file(path) for path in artifact_paths},
        "test_payload_accessed": True,
    }
    atomic_json(EVIDENCE_ROOT / "W17_STATISTICS_COMPLETENESS.json", receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--preflight", action="store_true")
    action.add_argument("--run", action="store_true")
    args = parser.parse_args()
    result = preflight() if args.preflight else run()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] != "FAIL" else 1


if __name__ == "__main__":
    raise SystemExit(main())
