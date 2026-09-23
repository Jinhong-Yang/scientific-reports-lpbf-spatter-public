"""Synthetic-only tests: no REV02 manifests, checkpoints or test IDs accessed."""
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import hashlib

import numpy as np
import pytest
from PIL import Image

SCRIPT_ROOT = Path(__file__).resolve().parents[1] / "scripts"


def load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPT_ROOT / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


a = load("rev02_memorization")
s = load("rev02_statistics")


def test_exact_sign_flip_minimum_ties_and_degenerate():
    assert s.exact_sign_flip(np.arange(1, 11))["raw_p"] == 2 / 1024
    assert s.exact_sign_flip([-1, 1] * 5)["raw_p"] == 1
    for value in (0.0, 0.37, -2.0):
        result = s.exact_sign_flip([value] * 10)
        assert result["raw_p"] is None
        assert result["status"] == "undefined_degenerate"
        assert result["sample_SD"] == 0
        json.dumps(result, allow_nan=False)
    near = s.exact_sign_flip([1.0] * 9 + [1.0 + 1e-13])
    assert near["raw_p"] == 2 / 1024 and near["near_degenerate"]
    with pytest.raises(ValueError):
        s.exact_sign_flip([1] * 9)
    with pytest.raises(ValueError):
        s.exact_sign_flip([1] * 9 + [np.nan])


@pytest.mark.parametrize("p,expected", [
    ({"a": None, "b": None, "c": None}, {"a": None, "b": None, "c": None}),
    ({"a": .02, "b": None, "c": None}, {"a": .02, "b": None, "c": None}),
    ({"a": .02, "b": .03, "c": None}, {"a": .04, "b": .04, "c": None}),
    ({"a": .01, "b": .03, "c": .02}, {"a": .03, "b": .04, "c": .04}),
])
def test_holm_defined(p, expected):
    assert s.holm_defined(p) == expected


def test_holm_rejects_fake_zero_and_bh_nominal42():
    with pytest.raises(ValueError):
        s.holm_defined({"a": 0.0})
    p = {str(i): None for i in range(42)}
    p["0"], p["1"] = .001, .002
    q = s.bh_nominal_42(p)
    assert q["0"] == pytest.approx(.042) and q["1"] == pytest.approx(.042)
    assert all(q[str(i)] is None for i in range(2, 42))
    with pytest.raises(ValueError):
        s.bh_nominal_42({"a": .01})


def factorial_rows(splits=("validation",)):
    rows = []
    for split in splits:
        for j, arm in enumerate(s.ARMS):
            for k, seed in enumerate(s.GENERATOR_SEEDS):
                row = {"split": split, "arm": arm, "seed": seed}
                for metric in s.ENDPOINTS + ("reconstruction_SSIM",):
                    row[metric] = .1 + .001 * k + j * .001 * (k + 1) ** 2
                rows.append(row)
    return rows


def test_factorial_formulas_and_grid_bh42():
    assert s.factorial_effects({"F00": 0, "F01": 2, "F10": 1, "F11": 5}) == {"proxy_main": 2, "boundary_moment_main": 3, "interaction": 2}
    result = s.summarize_factorial(factorial_rows())
    assert len(result["per_seed_endpoints"]) == 40
    assert len(result["factorial_summary"]) == 21
    assert result["global_BH_status"] == "pending_T7"
    assert all(r["endpoint"] != "reconstruction_SSIM" for r in result["factorial_summary"])
    full = s.summarize_factorial(factorial_rows(("validation", "test")), splits=("validation", "test"))
    assert len(full["factorial_summary"]) == 42
    assert full["global_BH_status"] == "complete_nominal_42"
    with pytest.raises(ValueError):
        s.summarize_factorial(factorial_rows()[:-1])
    with pytest.raises(ValueError):
        s.summarize_factorial(factorial_rows() + factorial_rows()[:1])
    with pytest.raises(ValueError):
        s.summarize_factorial(factorial_rows(("test",)))


def test_crossed_bootstrap_pairing_reproducibility_degeneracy():
    left = np.asarray([[.1, .2], [.2, .4], [.3, .5]])
    right = np.zeros_like(left)
    assert s.crossed_bootstrap(left, right, draws=50) == s.crossed_bootstrap(left, right, draws=50)
    assert s.crossed_bootstrap(left, right, draws=50)["raw_p"] == 2 / 51
    assert s.crossed_bootstrap(left, left)["raw_p"] is None
    assert s.crossed_bootstrap(np.ones((3, 2)), np.zeros((3, 2)))["raw_p"] is None
    with pytest.raises(ValueError):
        s.crossed_bootstrap(left, right[:, :1])


def detector_rows():
    rows = []
    for family, axes in (("fasterrcnn", ("B1B3", "B2", "B4")), ("retinanet", ("B4",))):
        for axis in axes:
            for milestone in s.MILESTONES:
                for arm in ("D0", "D1", "D2", "D4", "D5") if family == "fasterrcnn" else ("D0", "D2", "D4", "D5"):
                    for i, seed in enumerate(s.DETECTOR_SEEDS):
                        for j in range(2):
                            rows.append({"family": family, "split": "validation", "axis": axis, "milestone": milestone,
                                         "arm": arm, "ratio": 0 if arm in ("D0", "D1") else .5, "seed": seed, "specimen": f"toy{j}",
                                         s.METRIC: .2 + int(arm[1]) * .01 * (i + j + 1), "prediction_sha256": "1" * 64})
    return rows


def test_detector_holm48_16_and_aliases():
    rows = detector_rows()
    result = s.summarize_detector(rows, {"D2": .5, "D4": .5}, expected_specimens=2, draws=20)
    assert len(result) == 64
    assert {r["nominal_family_size"] for r in result} == {48, 16}
    assert all(r["n_seeds"] == 3 and r["n_specimens"] == 2 for r in result)
    d1 = s.contextual_D1(rows, expected_specimens=2, draws=20)
    assert len(d1) == 12 and all(r["Holm_adjusted_p"] is None for r in d1)
    with pytest.raises(ValueError):
        s.summarize_detector(rows + rows[:1], {"D2": .5, "D4": .5}, expected_specimens=2, draws=20)
    with pytest.raises(ValueError):
        s.summarize_detector(rows[1:], {"D2": .5, "D4": .5}, expected_specimens=2, draws=20)


def test_canonical_all_sources_and_hash_flip(tmp_path):
    native = np.arange(300 * 300, dtype=np.uint32).reshape(300, 300).astype(np.uint8)
    path = tmp_path / "toy.png"
    Image.fromarray(native).save(path)
    canonical = a.canonical_pixels(path)
    assert canonical.shape == (64, 64) and canonical.dtype == np.uint8
    assert np.array_equal(canonical, a.canonical_pixels(Image.fromarray(native)))
    assert a.pixel_hash(canonical) == a.pixel_hash(canonical.copy())
    with pytest.raises(ValueError):
        a.canonical_pixels(np.ones((64, 64), dtype=float))
    assert a.ssim(canonical, canonical) == 1


def test_cosine_tolerance_and_phash_popcount():
    assert a.cosine_distances(np.array([[1 + 1e-7]]), np.array([[1.]]))[0, 0] == 0
    with pytest.raises(ValueError):
        a.cosine_distances(np.array([[1 + 1e-4]]), np.array([[1.]]))
    assert a.hamming_matrix(np.asarray([0, 255], dtype=np.uint64), np.asarray([0, 15], dtype=np.uint64)).tolist() == [[0, 4], [8, 4]]


def test_calibration_rule_b_and_no_test_calibration():
    rows = [{"query_split": "validation", "reference_split": "train", "feature_min": x / 100, "ssim_max": .5 + x / 1000, "phash_min": 0} for x in range(100)]
    lock = a.calibrate_thresholds(rows)
    assert lock["phash_q01_diagnostic_only"] == 0
    assert a.apply_rule({"exact_original_or_flip": False, "feature_min": 0, "ssim_max": 0, "phash_min": 60}, lock)["suspicious"]
    assert not a.apply_rule({"exact_original_or_flip": "False", "feature_min": 1, "ssim_max": 0}, lock)["suspicious"]
    with pytest.raises(ValueError):
        a.calibrate_thresholds([{**rows[0], "query_split": "test"}])


def test_wilson_and_demotion_threshold():
    zero = a.wilson_interval(0, 50)
    assert zero["Wilson_95_CI_low"] == 0
    assert zero["Wilson_95_CI_high"] == pytest.approx(.071347599, abs=1e-8)
    rows = [{"control_class": label, "suspicious": index < 25} for label in a.CLASSES for index in range(50)]
    assert not a.summarize_controls(rows)["audit_contribution_removed"]
    rows[-26]["suspicious"] = False
    assert a.summarize_controls(rows)["audit_contribution_removed"]


def test_control_plan_stable_distinct_and_latent_epsilon():
    rows = [{"sample_id": f"toy{i:03d}", "image_path": f"toy{i}.png", "split": "validation", "source_kind": "real"} for i in range(60)]
    conditions = np.zeros((60, 13), dtype=np.float32)
    conditions[:, 3] = 1
    conditions[:, 5] = 1
    plan = a.make_control_plan(rows, conditions)
    assert plan == a.make_control_plan(rows, conditions)
    assert len(plan) == 50
    assert all(r["donor_a_id"] != r["donor_b_id"] and len(r["standard_normal_epsilon"]) == 16 for r in plan)
    assert len({r["epsilon_sha256"] for r in plan}) == 50


class ToyEngine:
    def embeddings(self, pixels):
        features = np.asarray(pixels, dtype=np.float64).reshape(len(pixels), -1)
        return features / np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-12)


def test_measure_flip_invariant_and_stable_reference_tie(tmp_path):
    pixels = np.zeros((64, 64), dtype=np.uint8)
    pixels[10:20, 4:15] = 220
    for name, array in (("a", pixels), ("b", pixels), ("query", pixels[:, ::-1])):
        Image.fromarray(array).save(tmp_path / f"{name}.bmp")
    bank = a.make_bank([{"sample_id": name, "image_path": str(tmp_path / f"{name}.bmp")} for name in ("b", "a")], Path, ToyEngine())
    result = a.measure_queries([{"sample_id": "q", "image_path": str(tmp_path / "query.bmp")}], bank, Path, ToyEngine(), query_split="control", reference_split="validation")[0]
    assert result["exact_original_or_flip"]
    assert result["feature_reference_id"] == "a"
    assert result["feature_orientation"] == "horizontal_flip"
    assert result["ssim_max"] == 1 and result["phash_min"] == 0


def test_nopde_uses_paired_difference_SD_not_arm_SD():
    rows = []
    for arm in ("F11", "G2N"):
        for i, seed in enumerate(s.GENERATOR_SEEDS):
            rows.append({"arm": arm, "seed": seed, "split": "validation", **{e: float(i + (1 if arm == "G2N" else 0)) for e in s.ENDPOINTS}})
    result = s.describe_nopde(rows, split="validation")
    assert not result["all_within_SD_band"]
    assert all(r["paired_difference_SD"] == 0 and r["G2N_minus_G2"] == 1 for r in result["endpoints"])


def test_statistics_all_gates_before_reading_any_endpoint(monkeypatch):
    def forbidden():
        raise PermissionError("toy locked test")
    fake = SimpleNamespace(load_protocol=lambda task: {}, require_test_unlocked=forbidden)
    monkeypatch.setattr(s, "_common", lambda: fake)
    with pytest.raises(PermissionError, match="toy locked test"):
        s._summary_cli("all")


def test_audit_cannot_initiate_stage_after_unlock(tmp_path, monkeypatch):
    unlock = tmp_path / "T7_UNLOCK.json"
    unlock.write_text("{}", encoding="utf-8")
    fake = SimpleNamespace(REV_ROOT=tmp_path, load_protocol=lambda task: {},
                           run_provenance=lambda *args: {}, task_dir=lambda task: tmp_path / "T4")
    monkeypatch.setattr(a, "_common", lambda: fake)
    with pytest.raises(ValueError, match="forbidden after T7"):
        a._run_stage("calibration", {}, lambda *args: pytest.fail("must not execute"))
    assert not (tmp_path / "T4" / "calibration").exists()


def test_frozen_source_image_hash_is_not_silently_replaced(tmp_path):
    path = tmp_path / "toy.bin"
    path.write_bytes(b"toy image bytes")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    fake = SimpleNamespace(resolve_path=Path, sha256_file=lambda p: hashlib.sha256(p.read_bytes()).hexdigest())
    rows = [{"sample_id": "toy", "image_path": str(path), "image_sha256": digest}]
    assert a._input_files(fake, rows, "toy") == {"toy_image_toy": path}
    path.write_bytes(b"modified bytes")
    with pytest.raises(ValueError, match="hash mismatch"):
        a._input_files(fake, rows, "toy")


def test_empty_audit_receipt_is_rejected(tmp_path):
    fake = SimpleNamespace(load_json=lambda p: {"provenance": {"input_hashes": {}}, "outputs": {}},
                           verify_implementation_lock=lambda: {})
    with pytest.raises(ValueError, match="Empty audit receipt"):
        a._verify_stage(fake, tmp_path)
