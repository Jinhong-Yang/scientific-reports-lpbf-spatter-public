import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("analyze_w10_mechanisms", ROOT / "src" / "analysis" / "analyze_w10_mechanisms.py")
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_blur_zero_is_identity_and_positive_reduces_roughness():
    image = np.zeros((2, 1, 16, 16), dtype=np.float32)
    image[:, :, 8, 8] = 1
    assert np.array_equal(MODULE.blur_images(image, 0), image)
    conditions = np.zeros((2, 13), dtype=np.float32)
    conditions[:, 9] = 0.25
    conditions[:, 7:9] = 0.0
    raw = MODULE.finite_difference_diagnostics(image, conditions)["own_laplacian_rms"]
    smooth = MODULE.finite_difference_diagnostics(MODULE.blur_images(image, 1.0), conditions)["own_laplacian_rms"]
    assert np.all(smooth < raw)


def test_per_image_mse_keeps_samples_separate():
    target = np.zeros((2, 1, 4, 4), dtype=np.float32)
    predicted = target.copy()
    predicted[1] = 1
    assert np.allclose(MODULE.mse_per_image(predicted, target), [0, 1])
