"""Frozen REV02 inference; import-safe numerical functions and gated CLI.

No dataset is loaded on import.  ``all`` is the only CLI mode that can read
test endpoints, and it calls the common T7 gate before constructing their paths.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy import stats

GENERATOR_SEEDS = tuple(range(41001, 41011))
DETECTOR_SEEDS = (51001, 51002, 51003)
ARMS = ("F00", "F01", "F10", "F11")
EFFECTS = ("proxy_main", "boundary_moment_main", "interaction")
ENDPOINTS = (
    "fresh_collocation_proxy_RMS", "reconstruction_PSNR", "PFFD_std",
    "within_condition_one_minus_SSIM", "boundary_energy",
    "moment_centroid_error", "moment_spread_error",
)
ALIASES = {
    "reconstruction_PSNR": "reconstruction_psnr_db",
    "within_condition_one_minus_SSIM": "generated_diversity_one_minus_ssim",
    "reconstruction_SSIM": "reconstruction_ssim_mean",
}
CONTRASTS = ("D2_minus_D0", "D4_minus_D0", "D2_minus_D5_at_D2_ratio", "D4_minus_D5_at_D4_ratio")
MILESTONES = (896, 1792, 4480, 8960)
METRIC = "specimen_level_study_mAP_50_95"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def exact_sign_flip(values: Sequence[float]) -> dict[str, Any]:
    """Enumerate 1024 sign patterns; undefined exactly-constant effects."""
    x = np.asarray(values, dtype=np.float64)
    _require(x.shape == (10,) and bool(np.isfinite(x).all()), "Exactly ten finite paired seed effects required")
    degenerate = bool(np.all(x == x[0]))
    mean = float(x.mean())
    sd = 0.0 if degenerate else float(x.std(ddof=1))
    half = float(stats.t.ppf(0.975, 9) * sd / np.sqrt(10))
    result = {
        "n_paired_seeds": 10, "mean_effect": mean, "sample_SD": sd,
        "Student_t_CI_low": mean - half, "Student_t_CI_high": mean + half,
        "interval_role": "degenerate_descriptive" if degenerate else "secondary_normality_dependent",
        "status": "undefined_degenerate" if degenerate else "defined",
        "raw_p": None, "tail_count": None, "permutation_count": 1024,
        "near_degenerate": bool(not degenerate and sd <= 1e-12),
    }
    if not degenerate:
        signs = 2 * ((np.arange(1024, dtype=np.uint16)[:, None] >> np.arange(10)) & 1).astype(np.int8) - 1
        means = np.mean(signs * x[None, :], axis=1)
        count = int(np.count_nonzero(np.abs(means) >= abs(mean) - 1e-12))
        _require(count >= 2, "Exact two-sided tail must include observed and opposite signs")
        result.update(raw_p=count / 1024, tail_count=count)
    return result


def holm_defined(p_values: Mapping[str, float | None]) -> dict[str, float | None]:
    """Exclude undefined cells, retaining them as public nulls."""
    for value in p_values.values():
        _require(value is None or (np.isfinite(value) and 0 < value <= 1), "Invalid/zero p-value")
    rank = {key: i for i, key in enumerate(p_values)}
    defined = sorted(((key, value) for key, value in p_values.items() if value is not None), key=lambda x: (x[1], rank[x[0]]))
    result = {key: None for key in p_values}
    running = 0.0
    for i, (key, value) in enumerate(defined):
        running = max(running, min(1.0, (len(defined) - i) * float(value)))
        result[key] = running
    return result


def bh_nominal_42(p_values: Mapping[str, float | None]) -> dict[str, float | None]:
    """Undefined cells enter sorting as 1, remain null in public output."""
    _require(len(p_values) == 42, "BH sensitivity requires all 42 nominal hypotheses")
    for value in p_values.values():
        _require(value is None or (np.isfinite(value) and 0 < value <= 1), "Invalid/zero p-value")
    rank = {key: i for i, key in enumerate(p_values)}
    ranked = sorted(((key, 1.0 if value is None else float(value)) for key, value in p_values.items()), key=lambda x: (x[1], rank[x[0]]))
    adjusted: dict[str, float | None] = {}
    running = 1.0
    for i in range(41, -1, -1):
        key, value = ranked[i]
        running = min(running, 42 * value / (i + 1))
        adjusted[key] = None if p_values[key] is None else min(1.0, running)
    return {key: adjusted[key] for key in p_values}


def factorial_effects(values: Mapping[str, float]) -> dict[str, float]:
    _require(set(values) == set(ARMS), "Four factorial arms required")
    return {
        "proxy_main": 0.5 * ((values["F10"] - values["F00"]) + (values["F11"] - values["F01"])),
        "boundary_moment_main": 0.5 * ((values["F01"] - values["F00"]) + (values["F11"] - values["F10"])),
        "interaction": (values["F11"] - values["F10"]) - (values["F01"] - values["F00"]),
    }


def _endpoint(row: Mapping[str, Any], key: str) -> float:
    value = row.get(key, row.get(ALIASES.get(key, "__no_alias__")))
    _require(value is not None, f"Missing endpoint {key}")
    result = float(value)
    _require(bool(np.isfinite(result)), f"Nonfinite endpoint {key}")
    return result


def summarize_factorial(rows: Sequence[Mapping[str, Any]], *, splits: Sequence[str] = ("validation",)) -> dict[str, Any]:
    _require(tuple(splits) in (("validation",), ("validation", "test")), "Only validation or validation+test supported")
    by_key: dict[tuple, dict] = {}
    normalized = []
    for row in rows:
        key = (str(row["split"]), str(row["arm"]), int(row["seed"]))
        _require(key not in by_key, f"Duplicate factorial cell {key}")
        _require(key[0] in splits and key[1] in ARMS and key[2] in GENERATOR_SEEDS, f"Unexpected factorial cell {key}")
        normalized_row = {"split": key[0], "arm": key[1], "seed": key[2], **{metric: _endpoint(row, metric) for metric in ENDPOINTS + ("reconstruction_SSIM",)}}
        by_key[key] = normalized_row
        normalized.append(normalized_row)
    expected = {(split, arm, seed) for split in splits for arm in ARMS for seed in GENERATOR_SEEDS}
    _require(set(by_key) == expected, f"Incomplete factorial grid; missing {len(expected - set(by_key))}")
    per_seed, summary, arm_summary = [], [], []
    for split in splits:
        for arm in ARMS:
            for endpoint in ENDPOINTS + ("reconstruction_SSIM",):
                x = np.asarray([by_key[split, arm, seed][endpoint] for seed in GENERATOR_SEEDS], dtype=np.float64)
                arm_summary.append({"split": split, "arm": arm, "endpoint": endpoint, "n": 10,
                                    "mean": float(x.mean()), "sample_SD": float(x.std(ddof=1)),
                                    "role": "mandatory_secondary_descriptive" if endpoint == "reconstruction_SSIM" else "primary"})
        for endpoint in ENDPOINTS:
            vectors = {effect: [] for effect in EFFECTS}
            for seed in GENERATOR_SEEDS:
                effects = factorial_effects({arm: by_key[split, arm, seed][endpoint] for arm in ARMS})
                for effect, value in effects.items():
                    vectors[effect].append(value)
                    per_seed.append({"split": split, "seed": seed, "endpoint": endpoint, "effect": effect, "value": value})
            inferences = {effect: exact_sign_flip(vectors[effect]) for effect in EFFECTS}
            adjusted = holm_defined({effect: inferences[effect]["raw_p"] for effect in EFFECTS})
            excluded = [effect for effect in EFFECTS if inferences[effect]["raw_p"] is None]
            for effect in EFFECTS:
                summary.append({"split": split, "endpoint": endpoint, "effect": effect, **inferences[effect],
                                "Holm_adjusted_p": adjusted[effect], "family_id": f"{split}::{endpoint}",
                                "nominal_family_size": 3, "tested_family_size": 3 - len(excluded),
                                "excluded_effect_ids": ";".join(excluded), "BH_q": None,
                                "BH_status": "pending_T7" if len(splits) == 1 else "computed"})
    if len(splits) == 2:
        p = {f"{r['split']}::{r['endpoint']}::{r['effect']}": r["raw_p"] for r in summary}
        adjusted = bh_nominal_42(p)
        for row in summary:
            row["BH_q"] = adjusted[f"{row['split']}::{row['endpoint']}::{row['effect']}"]
            row["BH_nominal_family_size"] = 42
            row["BH_defined_test_count"] = sum(value is not None for value in p.values())
            row["BH_degenerate_test_count"] = sum(value is None for value in p.values())
    return {"per_seed_endpoints": normalized, "per_seed_effects": per_seed,
            "arm_summary": arm_summary, "factorial_summary": summary,
            "global_BH_status": "pending_T7" if len(splits) == 1 else "complete_nominal_42"}


def crossed_bootstrap(left: np.ndarray, right: np.ndarray, *, draws: int = 10000, seed: int = 61005) -> dict[str, Any]:
    left, right = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    _require(left.ndim == 2 and left.shape == right.shape and min(left.shape) >= 2, "Complete paired seed-by-specimen matrices required")
    _require(bool(np.isfinite(left).all() and np.isfinite(right).all()), "Nonfinite paired matrix")
    _require(draws > 0, "Positive bootstrap draws required")
    difference = left - right
    point = float(difference.mean(axis=1).mean())
    if bool(np.all(difference == difference.flat[0])):
        return {"mean_difference": point, "CI_low": point, "CI_high": point, "raw_p": None,
                "status": "undefined_degenerate", "draws": draws, "random_seed": seed,
                "n_seeds": left.shape[0], "n_specimens": left.shape[1], "bootstrap_draw_sha256": None}
    rng = np.random.default_rng(seed)
    boot = np.empty(draws, dtype=np.float64)
    for i in range(draws):
        rows = rng.integers(0, left.shape[0], size=left.shape[0])
        cols = rng.integers(0, left.shape[1], size=left.shape[1])
        boot[i] = difference[np.ix_(rows, cols)].mean()
    low, high = np.quantile(boot, [0.025, 0.975], method="linear")
    p = min(1.0, 2 * min((1 + int(np.count_nonzero(boot <= 0))) / (draws + 1), (1 + int(np.count_nonzero(boot >= 0))) / (draws + 1)))
    return {"mean_difference": point, "CI_low": float(low), "CI_high": float(high), "raw_p": p,
            "status": "defined", "draws": draws, "random_seed": seed,
            "n_seeds": left.shape[0], "n_specimens": left.shape[1],
            "bootstrap_draw_sha256": hashlib.sha256(boot.tobytes()).hexdigest()}


def _family(value: str) -> str:
    mapping = {"fasterrcnn": "fasterrcnn", "Faster_RCNN": "fasterrcnn", "Faster R-CNN": "fasterrcnn",
               "retinanet": "retinanet", "RetinaNet": "retinanet"}
    _require(value in mapping, f"Unrecognized detector family {value}")
    return mapping[value]


def summarize_detector(rows: Sequence[Mapping[str, Any]], selected_ratios: Mapping[str, float], *, splits: Sequence[str] = ("validation",), expected_specimens: int = 32, draws: int = 10000) -> list[dict[str, Any]]:
    """Complete 48/16-cell families. Aliases never multiply a cell's records."""
    _require(set(selected_ratios) == {"D2", "D4"}, "Both frozen neural ratios required")
    grid: dict[tuple, dict[tuple, tuple[float, str]]] = {}
    for row in rows:
        family, split = _family(str(row["family"])), str(row["split"])
        _require(split in splits, f"Unexpected split {split}")
        arm, axis = str(row["arm"]), str(row["axis"])
        _require(arm in ("D0", "D1", "D2", "D4", "D5"), f"Unexpected detector arm {arm}")
        _require(axis in (("B1B3", "B2", "B4") if family == "fasterrcnn" else ("B4",)), "Unexpected axis")
        milestone, seed = int(row["milestone"]), int(row["seed"])
        _require(milestone in MILESTONES and seed in DETECTOR_SEEDS, "Unexpected milestone or seed")
        ratio = float(row["ratio"])
        _require(bool(np.isfinite(ratio)), "Nonfinite ratio")
        expected_ratios = {0.0} if arm in ("D0", "D1") else ({float(selected_ratios[arm])} if arm in ("D2", "D4") else set(map(float, selected_ratios.values())))
        _require(ratio in expected_ratios, f"Unexpected frozen ratio for {arm}")
        key = (family, split, axis, milestone, arm, ratio)
        cell_key = (seed, str(row["specimen"]))
        value = float(row.get(METRIC, row.get("study_mAP_50_95", "nan")))
        _require(bool(np.isfinite(value) and 0 <= value <= 1), "Invalid specimen AP")
        prediction_hash = str(row.get("prediction_sha256", ""))
        _require(len(prediction_hash) == 64 and all(c in "0123456789abcdefABCDEF" for c in prediction_hash), "Prediction SHA256 required")
        cell = grid.setdefault(key, {})
        _require(cell_key not in cell, f"Duplicate detector record {key} {cell_key}")
        cell[cell_key] = (value, prediction_hash)
    result: list[dict[str, Any]] = []
    for family, axes in (("fasterrcnn", ("B1B3", "B2", "B4")), ("retinanet", ("B4",))):
        for split in splits:
            family_results = []
            specimen_roster = None
            for axis in axes:
                for milestone in MILESTONES:
                    for contrast in CONTRASTS:
                        neural = contrast[:2]
                        comparator = "D0" if contrast.endswith("D0") else "D5"
                        ratio = float(selected_ratios[neural])
                        left = grid.get((family, split, axis, milestone, neural, ratio))
                        right = grid.get((family, split, axis, milestone, comparator, 0.0 if comparator == "D0" else ratio))
                        _require(left is not None and right is not None, f"Missing detector contrast cell {family} {split} {axis} {milestone} {contrast}")
                        _require(set(left) == set(right), "Unpaired specimen or seed records")
                        specimens = sorted({specimen for _, specimen in left})
                        _require(len(specimens) == expected_specimens, "Wrong specimen count")
                        expected = {(seed, specimen) for seed in DETECTOR_SEEDS for specimen in specimens}
                        _require(set(left) == expected, "Incomplete three-seed specimen grid")
                        if specimen_roster is None:
                            specimen_roster = specimens
                        _require(specimens == specimen_roster, "Specimen roster differs across family cells")
                        lm = np.asarray([[left[seed, specimen][0] for specimen in specimens] for seed in DETECTOR_SEEDS])
                        rm = np.asarray([[right[seed, specimen][0] for specimen in specimens] for seed in DETECTOR_SEEDS])
                        identity = {"left": sorted({v[1] for v in left.values()}), "right": sorted({v[1] for v in right.values()})}
                        stats_row = crossed_bootstrap(lm, rm, draws=draws)
                        family_results.append({"family": family, "split": split, "axis": axis, "milestone": milestone,
                                               "contrast": contrast, "neural_ratio": ratio, **stats_row,
                                               "prediction_pair_identity_sha256": hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest(),
                                               "inference_role": "multiplicity_adjusted_bootstrap_summary_not_exact_randomization"})
            p = {f"{r['axis']}::{r['milestone']}::{r['contrast']}": r["raw_p"] for r in family_results}
            adjusted = holm_defined(p)
            excluded = [k for k, v in p.items() if v is None]
            nominal = 48 if family == "fasterrcnn" else 16
            _require(len(p) == nominal, "Incorrect detector multiplicity family")
            for row in family_results:
                key = f"{row['axis']}::{row['milestone']}::{row['contrast']}"
                row.update(Holm_adjusted_p=adjusted[key], family_id=f"{family}::{split}",
                           nominal_family_size=nominal, tested_family_size=nominal - len(excluded),
                           excluded_contrast_ids=";".join(excluded),
                           benefit_supported=bool(row["mean_difference"] > 0 and adjusted[key] is not None and adjusted[key] < 0.05))
            result.extend(family_results)
    return result


def describe_nopde(rows: Sequence[Mapping[str, Any]], *, split: str) -> dict[str, Any]:
    """Frozen descriptive SD-band rule; never an equivalence test."""
    by_key = {(str(row["arm"]), int(row["seed"])): row for row in rows if row["split"] == split}
    _require(len(by_key) == len(rows), "Duplicate or wrong-split NoPDE cells")
    _require(set(by_key) == {(arm, seed) for arm in ("F11", "G2N") for seed in GENERATOR_SEEDS}, "Complete G2/G2N ten-seed grid required")
    endpoints = []
    for endpoint in ENDPOINTS:
        g2 = np.asarray([_endpoint(by_key["F11", seed], endpoint) for seed in GENERATOR_SEEDS])
        gn = np.asarray([_endpoint(by_key["G2N", seed], endpoint) for seed in GENERATOR_SEEDS])
        paired = gn - g2
        difference = float(paired.mean())
        band = 0.0 if np.all(paired == paired[0]) else float(paired.std(ddof=1))
        endpoints.append({"split": split, "endpoint": endpoint, "G2N_minus_G2": difference,
                          "G2_SD": float(g2.std(ddof=1)), "G2N_SD": float(gn.std(ddof=1)),
                          "paired_difference_SD": band, "SD_band": band,
                          "within_SD_band": abs(difference) <= band})
    return {"endpoints": endpoints, "all_within_SD_band": all(r["within_SD_band"] for r in endpoints),
            "interpretation": "predeclared_descriptive_SD_rule_not_equivalence_or_universal_NoPDE_claim"}


def contextual_D1(rows: Sequence[Mapping[str, Any]], *, splits=("validation",), expected_specimens=32, draws=10000) -> list[dict]:
    output = []
    for split in splits:
        for axis in ("B1B3", "B2", "B4"):
            for milestone in MILESTONES:
                selected = [r for r in rows if _family(str(r["family"])) == "fasterrcnn" and r["split"] == split and r["axis"] == axis and int(r["milestone"]) == milestone and r["arm"] in ("D0", "D1")]
                lookup = {}
                for r in selected:
                    key = (r["arm"], int(r["seed"]), str(r["specimen"]))
                    _require(key not in lookup, "Duplicate D1 contextual cell")
                    lookup[key] = float(r.get(METRIC, r.get("study_mAP_50_95", "nan")))
                specimens = sorted({r["specimen"] for r in selected})
                expected = {(arm, seed, specimen) for arm in ("D0", "D1") for seed in DETECTOR_SEEDS for specimen in specimens}
                _require(len(specimens) == expected_specimens and set(lookup) == expected, "Incomplete D1 contextual grid")
                left = np.asarray([[lookup["D1", seed, specimen] for specimen in specimens] for seed in DETECTOR_SEEDS])
                right = np.asarray([[lookup["D0", seed, specimen] for specimen in specimens] for seed in DETECTOR_SEEDS])
                value = crossed_bootstrap(left, right, draws=draws)
                output.append({"family": "fasterrcnn", "split": split, "axis": axis, "milestone": milestone,
                               "contrast": "D1_minus_D0", **value, "Holm_adjusted_p": None,
                               "inference_role": "contextual_unadjusted_descriptive_outside_primary_family", "benefit_claim_permitted": False})
    return output


def _common():
    import rev02_common
    return rev02_common


def _verify_upstream_summary(c, path: Path) -> dict:
    from rev02_generator import verify_current_artifact
    verify_current_artifact(path.parent)
    row = c.load_json(path)
    _require(row.get("status") in ("complete", "completed"), f"Incomplete source summary {path}")
    artifacts = row.get("artifact_hashes")
    _require(isinstance(artifacts, dict) and bool(artifacts), "Generator source lacks artifact ledger")
    for relative, digest in artifacts.items():
        _require(c.sha256_file(path.parent / relative) == digest, f"Changed generator artifact {relative}")
    identity_path = path.parent / "identity.json"
    _require(identity_path.exists() and "identity.json" in artifacts, "Generator identity must be artifact-bound")
    identity = c.load_json(identity_path)
    _require(hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest() == row.get("identity_sha256"), "Generator identity hash mismatch")
    provenance = identity["provenance"]
    for item in provenance["input_hashes"].values():
        _require(c.sha256_file(Path(item["path"])) == item["sha256"], "Changed generator input")
    _require(provenance["implementation_lock_sha256"] == c.sha256_file(c.REV_ROOT / "IMPLEMENTATION_LOCK.json"), "Changed source implementation lock")
    return row


def _verify_detector_ledger(c, path: Path, target: Path | None = None) -> dict:
    row = c.load_json(path)
    _require(row.get("status") in ("complete", "completed", "pretest_complete"), "Incomplete detector upstream receipt")
    files = row.get("files")
    _require(isinstance(files, dict) and bool(files), "Detector receipt lacks files ledger")
    checked = {}
    for value, digest in files.items():
        file_path = Path(value)
        if not file_path.is_absolute():
            file_path = path.parent / file_path
        _require(c.sha256_file(file_path) == digest, f"Changed detector artifact {file_path}")
        checked[str(file_path.resolve())] = digest
        if file_path.name in ("completion.json", "test_completion.json"):
            _verify_detector_ledger(c, file_path)
    if target is not None:
        _require(str(target.resolve()) in checked, f"Aggregate not bound by upstream receipt {target}")
    return row


def _summary_cli(split_mode: str) -> dict[str, Any]:
    c = _common()
    protocol = c.load_protocol("T5")
    if split_mode == "all":
        c.require_test_unlocked()
    splits = ("validation", "test") if split_mode == "all" else ("validation",)
    inputs, generator_rows, detector_rows = {}, [], []
    for split in splits:
        for arm in ARMS:
            for seed in GENERATOR_SEEDS:
                path = c.ARTIFACT_ROOT / "evaluation" / "generators" / split / f"{arm}_s{seed}" / "summary.json"
                inputs[f"generator_{split}_{arm}_{seed}"] = path
                row = _verify_upstream_summary(c, path)
                _require((row["split"], row["arm"], int(row["seed"])) == (split, arm, seed), "Generator path/payload mismatch")
                generator_rows.append(row)
        for task in ("T2", "T6B"):
            path = c.REV_ROOT / "results" / "REV02" / task / "raw" / f"specimen_metrics_{split}.csv"
            ledger = path.parent.parent / ("VALIDATION_COMPLETE.json" if split == "validation" else "TEST_AGGREGATE_COMPLETE.json")
            _verify_detector_ledger(c, ledger, path)
            inputs[f"detector_ledger_{task}_{split}"] = ledger
            inputs[f"detector_{task}_{split}"] = path
            detector_rows.extend(c.read_csv(path))
    ratios = {}
    selection_ledger = c.REV_ROOT / "results" / "REV02" / "T6A" / "PRETEST_SELECTION_FREEZE.json"
    _verify_detector_ledger(c, selection_ledger)
    inputs["selection_ledger"] = selection_ledger
    for arm in ("D2", "D4"):
        path = c.REV_ROOT / "results" / "REV02" / "T6A" / f"{arm}_ratio_selection.json"
        inputs[f"ratio_{arm}"] = path
        _verify_detector_ledger(c, selection_ledger, path)
        choice = c.load_json(path)
        _require(choice.get("selection_split") == "validation" and choice.get("test_used") is False, "Invalid validation-only ratio selection")
        ratios[arm] = float(choice["selected_ratio"])
    output = c.task_dir("T5") / split_mode
    provenance = c.run_provenance("T5", {"split_mode": split_mode}, inputs)
    identity = provenance
    receipt_path = output / "completed_receipt.json"
    if receipt_path.exists():
        receipt = c.load_json(receipt_path)
        _require(receipt["identity"] == identity, "Statistics completion identity changed")
        for relative, digest in receipt["outputs"].items():
            _require(c.sha256_file(output / relative) == digest, f"Changed statistics output {relative}")
        return c.load_json(output / "summary.json")
    _require(not output.exists(), "Incomplete statistics output preserved; documented recovery required")
    output.mkdir(parents=True)
    c.record_event("statistics_started", {"split_mode": split_mode, "identity": identity})
    try:
        factorial = summarize_factorial(generator_rows, splits=splits)
        detector = summarize_detector(detector_rows, ratios, splits=splits)
        d1 = contextual_D1(detector_rows, splits=splits)
        raw = output / "raw"
        raw.mkdir()
        c.write_csv(raw / "per_seed_generator_endpoints.csv", factorial["per_seed_endpoints"])
        c.write_csv(raw / "factorial_effects_per_seed.csv", factorial["per_seed_effects"])
        c.write_csv(raw / "factorial_effects_rev02.csv", factorial["factorial_summary"])
        c.write_csv(raw / "generator_arm_descriptives.csv", factorial["arm_summary"])
        if split_mode == "all":
            c.write_csv(raw / "factorial_BH_42_sensitivity.csv", factorial["factorial_summary"])
        c.write_csv(raw / "detector_paired_uncertainty.csv", detector)
        c.write_csv(raw / "detector_D1_contextual.csv", d1)
        summary = {"task_id": "T5", "split_mode": split_mode, "status": "pretest_complete" if split_mode == "validation" else "complete",
                   "generator_cells": len(generator_rows), "factorial_tests": len(factorial["factorial_summary"]),
                   "global_BH_status": factorial["global_BH_status"], "detector_contrast_cells": len(detector),
                   "finished_utc": c.utc_now(), "figures_status": "pending_root_manuscript_renderer", "scientific_protocol": protocol["protocol_id"],
                   "provenance": provenance}
        c.write_json(output / "summary.json", summary)
        outputs = {p.relative_to(output).as_posix(): c.sha256_file(p) for p in sorted(output.rglob("*")) if p.is_file()}
        c.write_json(receipt_path, {"identity": identity, "outputs": outputs, "completed_at": c.utc_now()})
        c.write_task_summary("T5", summary)
        c.append_run_log("T5", f"Completed {split_mode}: {len(generator_rows)} generator cells; {len(detector)} primary detector contrasts; {len(d1)} contextual D1 contrasts; global BH {factorial['global_BH_status']}.")
        c.record_event("statistics_completed", {"split_mode": split_mode, "receipt_sha256": c.sha256_file(receipt_path)})
        return summary
    except Exception as exc:
        failure = {"timestamp": c.utc_now(), "type": type(exc).__name__, "message": str(exc), "partial_outputs_preserved": True}
        c.write_json(output / "failure.json", failure)
        c.record_event("statistics_failed", failure)
        raise


def pretest_evidence() -> dict:
    c = _common()
    c.load_protocol("T5")
    c.verify_implementation_lock()
    output = c.task_dir("T5") / "validation"
    receipt = c.load_json(output / "completed_receipt.json")
    summary = c.load_json(output / "summary.json")
    _require(summary.get("status") == "pretest_complete" and summary.get("split_mode") == "validation", "T5 validation incomplete")
    _require(bool(receipt.get("outputs")) and bool(receipt["identity"].get("input_hashes")), "Empty statistics receipt")
    expected_provenance = c.run_provenance("T5", {"split_mode": "validation"}, {key: item["path"] for key, item in receipt["identity"]["input_hashes"].items()})
    _require(receipt["identity"] == expected_provenance, "Changed statistical implementation/environment/input provenance")
    for relative, digest in receipt["outputs"].items():
        _require(c.sha256_file(output / relative) == digest, "Changed statistical output")
    for item in receipt["identity"]["input_hashes"].values():
        _require(c.sha256_file(Path(item["path"])) == item["sha256"], "Changed statistical input")
    gen = c.read_csv(output / "raw" / "per_seed_generator_endpoints.csv")
    verified = summarize_factorial(gen, splits=("validation",))
    detector = c.read_csv(output / "raw" / "detector_paired_uncertainty.csv")
    expected = {(family, axis, str(milestone), contrast) for family, axes in (("fasterrcnn", ("B1B3", "B2", "B4")), ("retinanet", ("B4",))) for axis in axes for milestone in MILESTONES for contrast in CONTRASTS}
    _require(len(detector) == 64 and {(r["family"], r["axis"], r["milestone"], r["contrast"]) for r in detector} == expected, "Incomplete detector contrast families")
    _require(all(r["split"] == "validation" and int(r["n_seeds"]) == 3 and int(r["n_specimens"]) == 32 for r in detector), "Invalid detector paired grid dimensions")
    _require(summary["global_BH_status"] == verified["global_BH_status"] == "pending_T7", "Pretest BH status must remain pending")
    return {"T5": {"task_id": "T5", "status": "pretest_complete", "expected_units": 104, "completed_units": 104,
                   "unit_definition": "40 factorial validation cells plus 64 primary detector contrast cells", "missing_units": [],
                   "integrity_checks_passed": True, "last_training_validation_end_utc": receipt["completed_at"],
                   "output_paths": [str(output / "completed_receipt.json")] + [str(output / path) for path in receipt["outputs"]]}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    command = sub.add_parser("summarize")
    command.add_argument("--split", choices=("validation", "all"), required=True)
    args = parser.parse_args(argv)
    result = _summary_cli(args.split)
    print(json.dumps(result, indent=2, allow_nan=False))
    return result


if __name__ == "__main__":
    main()
