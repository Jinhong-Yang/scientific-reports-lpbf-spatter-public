from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("analyze_w11_quality", ROOT / "src" / "analysis" / "analyze_w11_quality.py")
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_pffd10_removes_only_duplicate_radiance_mass() -> None:
    assert len(MODULE.PFFD10_FEATURES) == 10
    assert "mean_intensity" in MODULE.PFFD10_FEATURES
    assert "radiance_mass" not in MODULE.PFFD10_FEATURES
    image = np.zeros((64, 64), dtype=np.float64)
    image[20:28, 30:42] = 1
    features = MODULE.morphology_features(image)
    assert tuple(features) == MODULE.PFFD10_FEATURES
    assert features["component_count"] == 1
    assert 0 <= features["centroid_x"] <= 1
    assert 0 <= features["centroid_y"] <= 1


def test_distribution_metrics_identical_and_shifted() -> None:
    rng = np.random.default_rng(7)
    left = rng.normal(size=(24, 6))
    assert MODULE.frechet_distance(left, left) < 1e-10
    same_population = rng.normal(size=(240, 6))
    kid_same = MODULE.polynomial_kid(same_population[:120], same_population[120:])
    shifted = left + 2
    assert MODULE.frechet_distance(left, shifted) > 20
    assert MODULE.polynomial_kid(left, shifted) > kid_same


def test_hamming_nearest_and_radial_power_are_deterministic() -> None:
    query = np.asarray([0, 3, 255], dtype=np.uint64)
    reference = np.asarray([0, 1, 254], dtype=np.uint64)
    distance, index = MODULE.hamming_nearest(query, reference, block_size=2)
    assert distance.tolist() == [0, 1, 1]
    assert index.tolist() == [0, 1, 2]
    images = np.zeros((2, 64, 64), dtype=np.float64)
    images[0, 32, 32] = 1
    images[1, 20:30, 25:35] = 1
    first = MODULE.radial_log_power(images)
    second = MODULE.radial_log_power(images)
    assert first.shape == (2, 32)
    assert np.array_equal(first, second)


def test_paired_geometry_uses_native_inherited_box_units() -> None:
    generated = np.zeros((1, 64, 64), dtype=np.float64)
    generated[0, 28:36, 28:36] = 1
    morphology = MODULE.morphology_matrix(generated)
    rows = [{"bbox_x_px": "120", "bbox_y_px": "120", "bbox_width_px": "60", "bbox_height_px": "60"}]
    values = MODULE.paired_measurements(generated, generated.copy(), morphology, rows)
    assert values["radial_log_power_l1"].tolist() == [0.0]
    assert values["laplacian_variance_absolute_difference"].tolist() == [0.0]
    assert values["centroid_distance_to_inherited_bbox_center"][0] < 0.02


def test_preflight_checks_frozen_weights_and_roster() -> None:
    result = MODULE.preflight()
    assert result["status"] == "PASS"
    assert result["test_access"] == "PROHIBITED"
    assert result["inception_weight_hash_match"] is True
    assert result["resnet_weight_hash_match"] is True
