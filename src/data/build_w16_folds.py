"""Build the frozen development-only W16 specimen-grouped outer folds."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "generators"))
from run_w09_factorial import atomic_json, read_csv, sha256_file, verify_inputs, verify_protocol  # noqa: E402

CONFIG_PATH = ROOT / "configs" / "w16_grouped_internal.json"
OUTPUT_ROOT = ROOT / "data" / "manifests" / "w16"
EVIDENCE_ROOT = ROOT / "evidence" / "grouped_internal"


def load_config() -> dict[str, Any]:
    value = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if value["status"] != "FROZEN_BEFORE_W16_OUTCOMES":
        raise RuntimeError("W16 configuration is not frozen")
    if value["held_out_test_access"] != "PROHIBITED_DURING_W16_FOLD_CONSTRUCTION_TRAINING_AND_EVALUATION":
        raise RuntimeError("W16 test lock differs")
    return value


def specimen_assignment(rows: list[dict[str, str]], seed: int, folds: int) -> dict[str, int]:
    specimen_rows: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        specimen_rows.setdefault(row["specimen"], []).append(row)
    strata: dict[str, list[str]] = {}
    for specimen, values in specimen_rows.items():
        row_strata = {row["stratum"] for row in values}
        if len(values) != 32 or len(row_strata) != 1:
            raise RuntimeError("W16 specimen row or stratum contract differs")
        strata.setdefault(next(iter(row_strata)), []).append(specimen)
    assignment: dict[str, int] = {}
    for stratum, specimens in sorted(strata.items()):
        ordered = sorted(specimens, key=lambda specimen: hashlib.sha256(f"{seed}|{stratum}|{specimen}".encode()).hexdigest())
        offset = int(hashlib.sha256(f"{seed}|{stratum}|offset".encode()).hexdigest()[:8], 16) % folds
        for index, specimen in enumerate(ordered):
            assignment[specimen] = (index + offset) % folds + 1
    return assignment


def atomic_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def build() -> dict[str, Any]:
    config = load_config()
    _, protocol_hash = verify_protocol()
    inputs = verify_inputs()
    train = read_csv(ROOT / inputs["outputs"]["train"]["path"])
    validation = read_csv(ROOT / inputs["outputs"]["validation"]["path"])
    rows = [{**row, "source_split": row["split"]} for row in train + validation]
    if len(rows) != 4608 or len({row["specimen"] for row in rows}) != 144:
        raise RuntimeError("W16 development population differs")
    if {row["split"] for row in rows} != {"train", "validation"}:
        raise RuntimeError("W16 received a role outside train/validation")
    fieldnames = list(rows[0])
    artifacts: list[dict[str, Any]] = []
    for seed in config["partition_seeds"]:
        assignment = specimen_assignment(rows, int(seed), int(config["folds_per_partition"]))
        if set(assignment) != {row["specimen"] for row in rows}:
            raise RuntimeError("W16 assignment is incomplete")
        for fold in range(1, int(config["folds_per_partition"]) + 1):
            outer = [{**row, "split": "oof"} for row in rows if assignment[row["specimen"]] == fold]
            inner = [{**row, "split": "train"} for row in rows if assignment[row["specimen"]] != fold]
            if {row["specimen"] for row in inner} & {row["specimen"] for row in outer}:
                raise RuntimeError("W16 fold leakage")
            for role, values in (("train", inner), ("oof", outer)):
                path = OUTPUT_ROOT / f"seed{seed}_fold{fold}_{role}.csv"
                atomic_csv(path, values, fieldnames)
                artifacts.append({
                    "partition_seed": seed,
                    "fold": fold,
                    "role": role,
                    "path": str(path.relative_to(ROOT)).replace("\\", "/"),
                    "sha256": sha256_file(path),
                    "rows": len(values),
                    "specimens": len({row["specimen"] for row in values}),
                })
        fold_sets = [set(specimen for specimen, value in assignment.items() if value == fold) for fold in range(1, 4)]
        if set.union(*fold_sets) != set(assignment) or any(left & right for index, left in enumerate(fold_sets) for right in fold_sets[index + 1:]):
            raise RuntimeError("W16 outer folds do not form a partition")
    receipt = {
        "schema_version": 1,
        "status": "PASS",
        "protocol_sha256": protocol_hash,
        "config_sha256": sha256_file(CONFIG_PATH),
        "source_manifest_hashes": {
            "train": inputs["outputs"]["train"]["sha256"],
            "validation": inputs["outputs"]["validation"]["sha256"],
        },
        "development_rows": 4608,
        "development_specimens": 144,
        "partition_seeds": config["partition_seeds"],
        "folds_per_partition": config["folds_per_partition"],
        "artifacts": artifacts,
        "held_out_test_rows_read": 0,
        "held_out_test_image_or_label_payload_accessed": False,
        "protocol_incident_disclosure": "docs/PROTOCOL_INCIDENT_001_TEST_IDENTIFIER_DISPLAY.md",
    }
    EVIDENCE_ROOT.mkdir(parents=True, exist_ok=True)
    atomic_json(EVIDENCE_ROOT / "W16_FOLD_RECEIPT.json", receipt)
    return receipt


if __name__ == "__main__":
    print(json.dumps(build(), indent=2, sort_keys=True))
