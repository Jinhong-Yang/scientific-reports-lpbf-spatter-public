"""Analyze the complete W09 validation-only 2x2 generator factorial."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "vendor" / "parquet"))
import pandas as pd
import numpy as np
from scipy import stats


ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = ROOT / "runs" / "new_study" / "W09_generator_factorial"
HISTORICAL_ROOT = ROOT / "historical" / "LPBF_REV02_20260902" / "runs" / "generators"
OUTPUT_ROOT = ROOT / "evidence" / "generator_factorial"
ARMS = ("F00", "F01", "F10", "F11")
SEEDS = tuple(range(61001, 61011))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    import os
    import tempfile

    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    import os
    import tempfile

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def factorial_effects(values: dict[str, float]) -> dict[str, float]:
    if set(values) != set(ARMS):
        raise ValueError("Exactly F00/F01/F10/F11 are required")
    return {
        "proxy_main_effect": 0.5 * ((values["F10"] - values["F00"]) + (values["F11"] - values["F01"])),
        "boundary_moment_main_effect": 0.5 * ((values["F01"] - values["F00"]) + (values["F11"] - values["F10"])),
        "interaction": (values["F11"] - values["F10"]) - (values["F01"] - values["F00"]),
    }


def n2_rows(run_root: Path = RUN_ROOT) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    protocol_hashes: set[str] = set()
    for arm in ARMS:
        for seed in SEEDS:
            directory = run_root / f"{arm}_s{seed}"
            summary_path = directory / "summary.json"
            if not summary_path.exists():
                raise RuntimeError(f"Missing frozen run: {arm}_s{seed}")
            summary = load_json(summary_path)
            identity = load_json(directory / "identity.json")
            if summary["status"] != "COMPLETED" or summary["test_payload_accessed"] is not False:
                raise RuntimeError(f"Invalid run status or test access: {arm}_s{seed}")
            if summary["epochs"] != 10 or summary["optimizer_updates"] != 1120:
                raise RuntimeError(f"Run budget differs: {arm}_s{seed}")
            if identity["arm"] != arm or identity["generator_seed"] != seed:
                raise RuntimeError(f"Run identity differs: {arm}_s{seed}")
            protocol_hashes.add(identity["protocol_sha256"])
            rows.append(
                {
                    "evidence_layer": "N2_NEW_STUDY_VALIDATION",
                    "arm": arm,
                    "seed": seed,
                    "best_epoch": summary["best_epoch"],
                    "best_validation_mse": summary["best_validation_mse"],
                    "optimizer_updates": summary["optimizer_updates"],
                    "elapsed_seconds": summary["elapsed_seconds"],
                    "gpu_hours": summary["gpu_hours"],
                    "summary_sha256": sha256_file(summary_path),
                    "test_payload_accessed": False,
                }
            )
    if len(protocol_hashes) != 1:
        raise RuntimeError("N2 runs do not share one protocol lock")
    return rows


def historical_rows(root: Path = HISTORICAL_ROOT) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for arm in ARMS:
        for ordinal in range(10):
            seed = 41001 + ordinal
            path = root / f"{arm}_s{seed}" / "summary.json"
            if not path.exists():
                raise RuntimeError(f"Missing recovered R1 summary: {arm}_s{seed}")
            summary = load_json(path)
            if summary.get("status") not in {"completed", "COMPLETED"}:
                raise RuntimeError(f"Recovered R1 run is not complete: {arm}_s{seed}")
            rows.append(
                {
                    "evidence_layer": "R1_RECOVERED_HISTORICAL",
                    "arm": arm,
                    "seed": seed,
                    "seed_ordinal": ordinal + 1,
                    "best_validation_mse": float(summary["best_validation_mse"]),
                    "summary_sha256": sha256_file(path),
                }
            )
    return rows


def build_effect_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_seed: dict[int, dict[str, float]] = {}
    for row in rows:
        by_seed.setdefault(int(row["seed"]), {})[str(row["arm"])] = float(row["best_validation_mse"])
    output: list[dict[str, Any]] = []
    for seed in sorted(by_seed):
        values = by_seed[seed]
        effects = factorial_effects(values)
        output.append({"seed": seed, **values, **effects})
    return output


def reproduction_rows(n2: list[dict[str, Any]], historical: list[dict[str, Any]]) -> list[dict[str, Any]]:
    n2_map = {(row["arm"], int(row["seed"]) - 61000): row for row in n2}
    old_map = {(row["arm"], row["seed_ordinal"]): row for row in historical}
    rows: list[dict[str, Any]] = []
    for arm in ARMS:
        for ordinal in range(1, 11):
            new = n2_map[(arm, ordinal)]
            old = old_map[(arm, ordinal)]
            rows.append(
                {
                    "arm": arm,
                    "seed_ordinal": ordinal,
                    "historical_seed": old["seed"],
                    "n2_seed": new["seed"],
                    "historical_validation_mse": old["best_validation_mse"],
                    "n2_validation_mse": new["best_validation_mse"],
                    "n2_minus_historical_mse": new["best_validation_mse"] - old["best_validation_mse"],
                    "interpretation": "descriptive_cross_layer_drift_not_a_paired_population_effect",
                }
            )
    return rows


def exact_sign_flip_p(values: list[float]) -> float:
    vector = np.asarray(values, dtype=np.float64)
    observed = abs(float(vector.mean()))
    statistics = []
    for mask in range(1 << len(vector)):
        signs = np.asarray([1.0 if mask & (1 << index) else -1.0 for index in range(len(vector))])
        statistics.append(abs(float(np.mean(vector * signs))))
    return float(np.mean(np.asarray(statistics) >= observed - 1e-15))


def effect_summary_rows(effects: list[dict[str, Any]]) -> list[dict[str, Any]]:
    names = ("proxy_main_effect", "boundary_moment_main_effect", "interaction")
    rows = []
    for name in names:
        values = np.asarray([row[name] for row in effects], dtype=np.float64)
        mean, sd = float(values.mean()), float(values.std(ddof=1))
        half = float(stats.t.ppf(0.975, len(values) - 1) * sd / math.sqrt(len(values)))
        rows.append({"effect": name, "seeds": len(values), "mean_validation_mse_effect": mean,
                     "sd": sd, "student_t_95_ci_low": mean - half, "student_t_95_ci_high": mean + half,
                     "exact_two_sided_sign_flip_p": exact_sign_flip_p(values)})
    ordered = sorted(range(len(rows)), key=lambda index: rows[index]["exact_two_sided_sign_flip_p"])
    adjusted = [0.0] * len(rows)
    running = 0.0
    for rank, index in enumerate(ordered):
        running = max(running, min(1.0, (len(rows) - rank) * rows[index]["exact_two_sided_sign_flip_p"]))
        adjusted[index] = running
    for row, value in zip(rows, adjusted):
        row["holm_adjusted_p_three_effects"] = value
    return rows


def analyze() -> dict[str, Any]:
    n2 = n2_rows()
    historical = historical_rows()
    effects = build_effect_rows(n2)
    effect_summary = effect_summary_rows(effects)
    reproduction = reproduction_rows(n2, historical)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    seed_csv = OUTPUT_ROOT / "factorial_seed_metrics.csv"
    seed_parquet = OUTPUT_ROOT / "factorial_seed_metrics.parquet"
    effects_csv = OUTPUT_ROOT / "factorial_effects_validation.csv"
    effects_summary_csv = OUTPUT_ROOT / "factorial_effects_summary.csv"
    reproduction_csv = OUTPUT_ROOT / "reproduction_comparison.csv"
    atomic_csv(seed_csv, n2)
    pd.DataFrame(n2).to_parquet(seed_parquet, index=False)
    atomic_csv(effects_csv, effects)
    atomic_csv(effects_summary_csv, effect_summary)
    atomic_csv(reproduction_csv, reproduction)
    receipt = {
        "schema_version": 1,
        "status": "PASS",
        "scope": "validation_only_pretest",
        "n2_runs": len(n2),
        "historical_runs": len(historical),
        "paired_factorial_seeds": len(effects),
        "test_payload_accessed": False,
        "interpretation": "Validation metrics support execution diagnostics and predeclared factorial estimates; they are not held-out detector outcomes.",
        "artifacts": {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in (seed_csv, seed_parquet, effects_csv, effects_summary_csv, reproduction_csv)
        },
    }
    atomic_json(OUTPUT_ROOT / "W09_ANALYSIS.json", receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    print(json.dumps(analyze(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
