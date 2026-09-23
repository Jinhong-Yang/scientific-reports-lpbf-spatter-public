#!/usr/bin/env python3
"""Synthetic-only W07 statistical plan prevalidation."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[2]
REV02 = ROOT / "src" / "reproduction" / "rev02_implementation_7b20c457"
HISTORICAL = ROOT / "historical" / "metal_spatter_pinn"
sys.path.insert(0, str(REV02 / "scripts"))
sys.path.insert(0, str(HISTORICAL / "src"))

import rev02_detector as detector  # noqa: E402


PRIMARY_CONTRASTS = ["NP-NB", "NP-NF", "NP-N1", "NP-NR"]


def factorial_effects(values: dict[str, float]) -> dict[str, float]:
    if set(values) != {"F00", "F01", "F10", "F11"}:
        raise ValueError("Factorial cell set must be exactly F00/F01/F10/F11")
    return {
        "optical_proxy_main": 0.5 * ((values["F10"] - values["F00"]) + (values["F11"] - values["F01"])),
        "boundary_moment_main": 0.5 * ((values["F01"] - values["F00"]) + (values["F11"] - values["F10"])),
        "interaction": (values["F11"] - values["F10"]) - (values["F01"] - values["F00"]),
    }


def exact_sign_flip(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("Finite one-dimensional paired differences are required")
    observed = abs(float(values.mean()))
    exceed = 0
    total = 2 ** len(values)
    for signs in itertools.product((-1.0, 1.0), repeat=len(values)):
        statistic = abs(float(np.mean(values * np.asarray(signs))))
        exceed += int(statistic >= observed - 1e-15)
    return {
        "n": len(values),
        "observed_abs_mean": observed,
        "two_sided_p": exceed / total,
        "enumerations": total,
    }


def holm_adjust(p_values: dict[str, float | None]) -> dict[str, float | None]:
    defined = [(name, float(value)) for name, value in p_values.items() if value is not None]
    if any(value <= 0 or value > 1 or not math.isfinite(value) for _, value in defined):
        raise ValueError("Defined p-values must be finite in (0,1]")
    ordered = sorted(defined, key=lambda item: (item[1], item[0]))
    adjusted: dict[str, float | None] = {name: None for name in p_values}
    running = 0.0
    total = len(ordered)
    for rank, (name, value) in enumerate(ordered):
        running = max(running, min(1.0, (total - rank) * value))
        adjusted[name] = running
    return adjusted


def bootstrap_mean_ci(
    differences: np.ndarray,
    seed: int,
    draws: int,
    confidence: float,
    resample_replicates: bool,
) -> dict[str, float | int | bool]:
    differences = np.asarray(differences, dtype=float)
    if differences.ndim != 2 or min(differences.shape) < 2 or not np.isfinite(differences).all():
        raise ValueError("A finite cluster-by-replicate matrix with at least 2x2 cells is required")
    if draws < 100:
        raise ValueError("At least 100 bootstrap draws are required")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0,1)")
    rng = np.random.default_rng(seed)
    clusters, replicates = differences.shape
    cluster_indices = rng.integers(0, clusters, size=(draws, clusters))
    if resample_replicates:
        replicate_indices = rng.integers(0, replicates, size=(draws, replicates))
        boot = differences[cluster_indices[:, :, None], replicate_indices[:, None, :]].mean(axis=(1, 2))
    else:
        boot = differences[cluster_indices].mean(axis=(1, 2))
    alpha = 1.0 - confidence
    low, high = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
    return {
        "estimate": float(differences.mean()),
        "low": float(low),
        "high": float(high),
        "confidence": confidence,
        "draws": draws,
        "resample_replicates": resample_replicates,
    }


def simulate_coverage(
    *,
    effect: float,
    clusters: int,
    replicates: int,
    simulations: int,
    draws: int,
    seed: int,
    confidence: float = 0.95,
) -> dict[str, float | int]:
    rng = np.random.default_rng(seed)
    covered = 0
    rejected_zero = 0
    widths = []
    estimates = []
    for simulation in range(simulations):
        cluster_component = rng.normal(0.0, 0.02, size=(clusters, 1))
        replicate_component = rng.normal(0.0, 0.005, size=(1, replicates))
        residual = rng.normal(0.0, 0.01, size=(clusters, replicates))
        differences = effect + cluster_component + replicate_component + residual
        interval = bootstrap_mean_ci(
            differences,
            seed=seed + 10000 + simulation,
            draws=draws,
            confidence=confidence,
            resample_replicates=True,
        )
        covered += int(interval["low"] <= effect <= interval["high"])
        rejected_zero += int(interval["low"] > 0 or interval["high"] < 0)
        widths.append(interval["high"] - interval["low"])
        estimates.append(interval["estimate"])
    rate = covered / simulations
    rejection_rate = rejected_zero / simulations
    return {
        "effect": effect,
        "clusters": clusters,
        "replicates": replicates,
        "simulations": simulations,
        "bootstrap_draws": draws,
        "coverage": rate,
        "coverage_mc_se": math.sqrt(rate * (1 - rate) / simulations),
        "zero_rejection_rate": rejection_rate,
        "zero_rejection_mc_se": math.sqrt(rejection_rate * (1 - rejection_rate) / simulations),
        "mean_interval_width": float(np.mean(widths)),
        "mean_estimate": float(np.mean(estimates)),
        "synthetic_cluster_sd": 0.02,
        "synthetic_replicate_sd": 0.005,
        "synthetic_residual_sd": 0.01,
    }


def planning_grid() -> pd.DataFrame:
    rows = []
    family_alpha = 0.05 / len(PRIMARY_CONTRASTS)
    critical = float(stats.norm.ppf(1 - family_alpha / 2))
    for clusters, replicates, effect in itertools.product([16, 32, 64], [3, 5, 10], [0.0, 0.005, 0.01, 0.02]):
        standard_error = math.sqrt(0.02**2 / clusters + 0.005**2 / replicates + 0.01**2 / (clusters * replicates))
        delta = effect / standard_error if standard_error else 0.0
        power = float(stats.norm.cdf(-critical - delta) + 1 - stats.norm.cdf(critical - delta))
        rows.append(
            {
                "evidence_layer": "SYNTHETIC_FIXTURE_ONLY",
                "clusters": clusters,
                "pipeline_replicates": replicates,
                "assumed_effect_AP": effect,
                "assumed_cluster_sd_AP": 0.02,
                "assumed_replicate_sd_AP": 0.005,
                "assumed_residual_sd_AP": 0.01,
                "family_comparisons": len(PRIMARY_CONTRASTS),
                "two_sided_family_alpha": 0.05,
                "per_comparison_alpha": family_alpha,
                "normal_critical_value": critical,
                "approx_standard_error_AP": standard_error,
                "approx_simultaneous_ci_half_width_AP": critical * standard_error,
                "approx_normal_power": power,
                "pilot_calibrated": False,
            }
        )
    return pd.DataFrame(rows)


def paired_cluster_coco_bootstrap(
    left: list[dict[str, object]],
    right: list[dict[str, object]],
    *,
    draws: int,
    seed: int,
) -> dict[str, object]:
    if draws < 10:
        raise ValueError("At least 10 COCO bootstrap draws are required")
    keys = ("cluster_id", "replicate", "item_id")
    left_map = {tuple(row[key] for key in keys): row for row in left}
    right_map = {tuple(row[key] for key in keys): row for row in right}
    if len(left_map) != len(left) or len(right_map) != len(right):
        raise ValueError("Duplicate cluster/replicate/item keys are forbidden before resampling")
    if set(left_map) != set(right_map):
        raise ValueError("Paired arms must have identical cluster/replicate/item keys")
    clusters = sorted({str(row["cluster_id"]) for row in left})
    replicates = sorted({int(row["replicate"]) for row in left})
    if len(clusters) < 2 or len(replicates) < 2:
        raise ValueError("At least two clusters and two pipeline replicates are required")
    grouped_left = {
        (cluster, replicate): [
            row for row in left if str(row["cluster_id"]) == cluster and int(row["replicate"]) == replicate
        ]
        for cluster in clusters for replicate in replicates
    }
    grouped_right = {
        (cluster, replicate): [
            row for row in right if str(row["cluster_id"]) == cluster and int(row["replicate"]) == replicate
        ]
        for cluster in clusters for replicate in replicates
    }
    if any(not rows for rows in grouped_left.values()) or any(not rows for rows in grouped_right.values()):
        raise ValueError("Every cluster/replicate cell must contain at least one image record")

    def strip_lineage(row: dict[str, object]) -> dict[str, object]:
        return {
            key: value for key, value in row.items()
            if key not in {"cluster_id", "replicate", "item_id"}
        }

    rng = np.random.default_rng(seed)
    differences = []
    duplicate_cluster_draws = 0
    maximum_resampled_images = 0
    for _ in range(draws):
        sampled_clusters = rng.choice(clusters, size=len(clusters), replace=True).tolist()
        sampled_replicates = rng.choice(replicates, size=len(replicates), replace=True).tolist()
        duplicate_cluster_draws += int(len(set(sampled_clusters)) < len(sampled_clusters))
        left_rep_ap = []
        right_rep_ap = []
        for replicate in sampled_replicates:
            left_rows = [
                strip_lineage(row)
                for cluster in sampled_clusters
                for row in grouped_left[(cluster, int(replicate))]
            ]
            right_rows = [
                strip_lineage(row)
                for cluster in sampled_clusters
                for row in grouped_right[(cluster, int(replicate))]
            ]
            maximum_resampled_images = max(maximum_resampled_images, len(left_rows), len(right_rows))
            left_ap = detector.official_coco_metrics(left_rows)["COCO_AP"]
            right_ap = detector.official_coco_metrics(right_rows)["COCO_AP"]
            if left_ap is None or right_ap is None:
                raise ValueError("Primary pooled COCO AP is undefined in a bootstrap draw")
            left_rep_ap.append(left_ap)
            right_rep_ap.append(right_ap)
        differences.append(float(np.mean(left_rep_ap) - np.mean(right_rep_ap)))
    values = np.asarray(differences)
    return {
        "draws": draws,
        "clusters": len(clusters),
        "pipeline_replicates": len(replicates),
        "mean_difference": float(values.mean()),
        "minimum_difference": float(values.min()),
        "maximum_difference": float(values.max()),
        "duplicate_cluster_draws": duplicate_cluster_draws,
        "maximum_resampled_images_per_replicate": maximum_resampled_images,
        "paired_cluster_and_replicate_draws": True,
        "new_evaluation_image_ids_assigned_by_coco_wrapper": True,
    }


def coco_resampling_fixture() -> dict[str, object]:
    left = []
    right = []
    for replicate in (1, 2):
        for cluster_index in range(4):
            base = {
                "cluster_id": f"cluster-{cluster_index}",
                "replicate": replicate,
                "item_id": f"item-{cluster_index}",
                "gt_boxes": [[10, 10, 30, 30]],
                "width": 300,
                "height": 300,
            }
            left.append({**base, "boxes": [[10, 10, 30, 30]], "scores": [0.9], "labels": [1]})
            if cluster_index < 2:
                right.append({**base, "boxes": [[10, 10, 30, 30]], "scores": [0.9], "labels": [1]})
            else:
                right.append({**base, "boxes": [], "scores": [], "labels": []})
    result = paired_cluster_coco_bootstrap(left, right, draws=40, seed=74003)
    aa = paired_cluster_coco_bootstrap(left, [dict(row) for row in left], draws=20, seed=74004)
    if aa["minimum_difference"] != 0 or aa["maximum_difference"] != 0:
        raise RuntimeError("COCO A/A bootstrap fixture is nonzero")
    if result["duplicate_cluster_draws"] == 0:
        raise RuntimeError("COCO bootstrap fixture did not exercise duplicate cluster draws")
    return {"known_difference_fixture": result, "aa_fixture": aa}


def run_prevalidation(output_dir: Path) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    null = simulate_coverage(
        effect=0.0, clusters=32, replicates=5, simulations=300, draws=1000, seed=74001
    )
    known = simulate_coverage(
        effect=0.02, clusters=32, replicates=5, simulations=300, draws=1000, seed=74002
    )
    sign_flip = exact_sign_flip(np.ones(5))
    factorial = factorial_effects({"F00": 0.0, "F01": 2.0, "F10": 1.0, "F11": 5.0})
    holm = holm_adjust({"c1": 0.01, "c2": 0.03, "c3": 0.02, "c4": None})
    coco_fixture = coco_resampling_fixture()
    acceptance = {
        "null_coverage_reasonable": 0.90 <= null["coverage"] <= 0.99,
        "null_type_i_below_0p10": null["zero_rejection_rate"] < 0.10,
        "known_effect_mean_close": abs(known["mean_estimate"] - 0.02) < 0.003,
        "five_seed_minimum_two_sided_sign_flip_p_is_0p0625": math.isclose(sign_flip["two_sided_p"], 0.0625),
        "factorial_fixture_matches": factorial == {
            "optical_proxy_main": 2.0,
            "boundary_moment_main": 3.0,
            "interaction": 2.0,
        },
        "holm_fixture_matches": holm == {"c1": 0.03, "c2": 0.04, "c3": 0.04, "c4": None},
        "raw_record_coco_resampling_aa_is_zero": coco_fixture["aa_fixture"]["mean_difference"] == 0,
        "raw_record_coco_resampling_exercised_duplicate_clusters": coco_fixture["known_difference_fixture"]["duplicate_cluster_draws"] > 0,
    }
    grid = planning_grid()
    grid_path = output_dir / "power_precision_simulation.csv"
    grid.to_csv(grid_path, index=False)
    report = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "PASS_WITH_LIMITATION" if all(acceptance.values()) else "FAILED",
        "evidence_layer": "SYNTHETIC_FIXTURE_ONLY",
        "observed_study_outcomes_used": False,
        "test_split_used": False,
        "primary_family": PRIMARY_CONTRASTS,
        "primary_family_status": "PROPOSED_NOT_W08_FROZEN",
        "confidence": {
            "ordinary_paired_cluster_interval": 0.95,
            "four_comparison_bonferroni_interval": 0.9875,
            "bootstrap_draws_proposal": 20000,
            "bootstrap_draws_used_in_fixture": 1000,
        },
        "fixtures": {
            "null_clustered": null,
            "known_effect_clustered": known,
            "five_seed_sign_flip": sign_flip,
            "factorial": factorial,
            "holm": holm,
            "raw_record_coco_resampling": coco_fixture,
        },
        "acceptance": acceptance,
        "planning_grid_rows": int(len(grid)),
        "limitations": [
            "Synthetic variance components are test fixtures, not estimates from the LPBF study.",
            "The planning grid uses a normal approximation and must be recalibrated from W06 dev-pilot variance components.",
            "The raw-record COCO fixture validates paired resampling mechanics; production intervals still require frozen N2 predictions and 20,000-draw Monte Carlo checks.",
            "External-build uncertainty cannot be estimated because no independent cohort is available.",
        ],
    }
    (output_dir / "statistics_tests.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = run_prevalidation(args.output_dir)
    print(json.dumps(report, indent=2))
    return 0 if report["status"] != "FAILED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
