"""Create the frozen nested W14 training-only specimen subsets."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "generators"))
from run_w09_factorial import atomic_json, read_csv, sha256_file, verify_inputs, verify_protocol  # noqa: E402


CONFIG_PATH = ROOT / "configs" / "w14_low_data.json"
OUTPUT_ROOT = ROOT / "data" / "manifests" / "w14"
EVIDENCE_ROOT = ROOT / "evidence" / "low_data"


def load_config() -> dict[str, Any]:
    value = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if value["status"] != "FROZEN_BEFORE_W14_OUTCOMES" or value["held_out_test_access"] != "PROHIBITED_DURING_SUBSET_SELECTION_TRAINING_AND_MODEL_SELECTION":
        raise RuntimeError("W14 subset specification is not frozen and test-locked")
    return value


def atomic_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def specimen_matrix(rows: list[dict[str, str]]) -> tuple[list[str], np.ndarray, dict[str, str]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row["specimen"], []).append(row)
    specimen_ids = sorted(grouped)
    materials = {}
    matrix = []
    for specimen in specimen_ids:
        values = grouped[specimen]
        material_set = {row["material"] for row in values}
        if len(material_set) != 1:
            raise RuntimeError("A specimen crosses material labels")
        materials[specimen] = next(iter(material_set))
        centers_x = [(float(row["bbox_x_px"]) + float(row["bbox_width_px"]) / 2) / 300 for row in values]
        centers_y = [(float(row["bbox_y_px"]) + float(row["bbox_height_px"]) / 2) / 300 for row in values]
        log_areas = [math.log(float(row["bbox_width_px"]) * float(row["bbox_height_px"])) for row in values]
        matrix.append([
            np.mean([float(row["laser_power_w"]) for row in values]),
            np.mean([float(row["scan_speed_mm_s"]) for row in values]),
            np.mean([float(row["line_energy_j_mm"]) for row in values]),
            np.mean([float(row["lamination_direction_deg"]) for row in values]),
            np.mean(centers_x),
            np.mean(centers_y),
            np.mean(log_areas),
        ])
    matrix_np = np.asarray(matrix, dtype=np.float64)
    scale = matrix_np.std(axis=0, ddof=0)
    scale[scale < 1e-12] = 1.0
    matrix_np = (matrix_np - matrix_np.mean(axis=0)) / scale
    return specimen_ids, matrix_np, materials


def nested_order(specimen_ids: list[str], matrix: np.ndarray, materials: dict[str, str], seed: int) -> list[int]:
    if matrix.shape != (len(specimen_ids), 7) or len(set(specimen_ids)) != len(specimen_ids):
        raise ValueError("Invalid specimen feature matrix")
    rng = np.random.default_rng(seed)
    tie_rank = rng.permutation(len(specimen_ids))
    rank = np.empty(len(specimen_ids), dtype=int)
    rank[tie_rank] = np.arange(len(specimen_ids))
    selected: list[int] = []
    for material in sorted(set(materials.values())):
        candidates = [index for index, specimen in enumerate(specimen_ids) if materials[specimen] == material]
        chosen = candidates[int(rng.integers(0, len(candidates)))]
        selected.append(chosen)
    while len(selected) < len(specimen_ids):
        remaining = [index for index in range(len(specimen_ids)) if index not in selected]
        minimum_distance = {
            index: float(np.min(np.mean((matrix[selected] - matrix[index]) ** 2, axis=1)))
            for index in remaining
        }
        chosen = min(remaining, key=lambda index: (-minimum_distance[index], int(rank[index]), specimen_ids[index]))
        selected.append(chosen)
    return selected


def build() -> dict[str, Any]:
    config = load_config()
    _, protocol_hash = verify_protocol()
    inputs = verify_inputs()
    train_path = ROOT / inputs["outputs"]["train"]["path"]
    rows = read_csv(train_path)
    if len(rows) != 3584 or len({row["specimen"] for row in rows}) != 112 or any(row["split"] != "train" for row in rows):
        raise RuntimeError("W14 source is not the frozen training-only population")
    specimen_ids, matrix, materials = specimen_matrix(rows)
    summary_rows = []
    artifacts = {}
    rosters: dict[str, list[str]] = {}
    for realization in config["subset_realizations"]:
        order = nested_order(specimen_ids, matrix, materials, int(realization["selection_seed"]))
        previous: set[str] = set()
        for size in config["subset_sizes_specimens"]:
            selected = {specimen_ids[index] for index in order[:size]}
            if not previous <= selected or len(selected) != size:
                raise RuntimeError("W14 nesting contract failed")
            previous = selected
            selected_rows = sorted((row for row in rows if row["specimen"] in selected), key=lambda row: row["sample_id"])
            if len(selected_rows) != size * 32:
                raise RuntimeError("W14 selected specimen does not contribute exactly 32 images")
            path = OUTPUT_ROOT / f"{realization['id']}_n{size}.csv"
            atomic_csv(path, selected_rows, list(rows[0]))
            digest = sha256_file(path)
            key = f"{realization['id']}_n{size}"
            rosters[key] = sorted(selected)
            artifacts[str(path.relative_to(ROOT)).replace("\\", "/")] = digest
            counts: dict[str, int] = {}
            for specimen in selected:
                counts[materials[specimen]] = counts.get(materials[specimen], 0) + 1
            for rank_index, index in enumerate(order[:size], 1):
                summary_rows.append({
                    "realization": realization["id"],
                    "selection_seed": realization["selection_seed"],
                    "subset_size_specimens": size,
                    "subset_images": len(selected_rows),
                    "specimen": specimen_ids[index],
                    "nested_rank": rank_index,
                    "material": materials[specimen_ids[index]],
                    "subset_manifest": str(path.relative_to(ROOT)).replace("\\", "/"),
                    "subset_manifest_sha256": digest,
                    "material_counts_json": json.dumps(counts, sort_keys=True),
                    "test_payload_accessed": False,
                })
    manifest_path = EVIDENCE_ROOT / "low_data_subset_manifest.csv"
    atomic_csv(manifest_path, summary_rows, list(summary_rows[0]))
    artifacts[str(manifest_path.relative_to(ROOT)).replace("\\", "/")] = sha256_file(manifest_path)
    receipt = {
        "schema_version": 1,
        "status": "PASS",
        "protocol_sha256": protocol_hash,
        "config_sha256": sha256_file(CONFIG_PATH),
        "source_train_manifest_sha256": sha256_file(train_path),
        "subset_manifests": 9,
        "realizations": 3,
        "sizes": config["subset_sizes_specimens"],
        "nested_checks": {
            realization["id"]: (
                set(rosters[f"{realization['id']}_n14"]) < set(rosters[f"{realization['id']}_n28"])
                and set(rosters[f"{realization['id']}_n28"]) < set(rosters[f"{realization['id']}_n56"])
            )
            for realization in config["subset_realizations"]
        },
        "artifacts": artifacts,
        "model_outcomes_used": False,
        "test_payload_accessed": False,
    }
    atomic_json(EVIDENCE_ROOT / "W14_SUBSET_RECEIPT.json", receipt)
    return receipt


def main() -> int:
    argparse.ArgumentParser(description=__doc__).parse_args()
    print(json.dumps(build(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
