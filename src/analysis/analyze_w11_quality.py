"""Evaluate W11 pools on one frozen validation target roster without test access."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Iterable

import numpy as np
from PIL import Image
from scipy.ndimage import label, laplace
from skimage.metrics import structural_similarity
import torch
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "vendor" / "parquet"))
import pandas as pd  # noqa: E402

sys.path.insert(0, str(ROOT / "src" / "generators"))
from run_w09_factorial import (  # noqa: E402
    atomic_json,
    image_tensor,
    read_csv,
    sha256_file,
    verify_inputs,
    verify_protocol,
)


CONFIG_PATH = ROOT / "configs" / "w11_quality_metrics.json"
RUN_ROOT = ROOT / "runs" / "new_study" / "W11_pools"
EVIDENCE_ROOT = ROOT / "evidence" / "stronger_generator"
CACHE_ROOT = EVIDENCE_ROOT / "cache"


def load_config() -> dict[str, Any]:
    value = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if value["status"] != "FROZEN_BEFORE_W11_POOL_OUTPUT":
        raise RuntimeError("W11 quality specification is not frozen")
    return value


def object_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".parquet", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".csv", delete=False, mode="w", encoding="utf-8", newline="") as handle:
        temporary = Path(handle.name)
        frame.to_csv(handle, index=False)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


PFFD10_FEATURES = (
    "mean_intensity",
    "max_intensity",
    "centroid_x",
    "centroid_y",
    "variance_x",
    "variance_y",
    "bright_area_fraction",
    "bbox_width_fraction",
    "bbox_height_fraction",
    "component_count",
)


def morphology_features(image: np.ndarray) -> dict[str, float]:
    """Historical morphology definition with the duplicated mass coordinate removed."""
    image = np.clip(np.asarray(image, dtype=np.float64), 0.0, 1.0)
    height, width = image.shape
    y, x = np.mgrid[0:height, 0:width]
    weights = image**2 + 1e-8
    mass = weights.sum()
    cx = float((weights * x).sum() / mass / max(width - 1, 1))
    cy = float((weights * y).sum() / mass / max(height - 1, 1))
    var_x = float((weights * (x / max(width - 1, 1) - cx) ** 2).sum() / mass)
    var_y = float((weights * (y / max(height - 1, 1) - cy) ** 2).sum() / mass)
    threshold = max(0.08, float(image.mean() + 2.5 * image.std()))
    mask = image >= threshold
    _, components = label(mask)
    if mask.any():
        ys, xs = np.nonzero(mask)
        bbox_width = (xs.max() - xs.min() + 1) / width
        bbox_height = (ys.max() - ys.min() + 1) / height
    else:
        bbox_width = bbox_height = 0.0
    return {
        "mean_intensity": float(image.mean()),
        "max_intensity": float(image.max()),
        "centroid_x": cx,
        "centroid_y": cy,
        "variance_x": var_x,
        "variance_y": var_y,
        "bright_area_fraction": float(mask.mean()),
        "bbox_width_fraction": float(bbox_width),
        "bbox_height_fraction": float(bbox_height),
        "component_count": float(components),
    }


def morphology_matrix(images: np.ndarray) -> np.ndarray:
    records = [morphology_features(image) for image in images]
    return np.asarray([[row[key] for key in PFFD10_FEATURES] for row in records], dtype=np.float64)


def fit_scaler(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[str]]:
    center = np.asarray(matrix, dtype=np.float64).mean(axis=0)
    scale = np.asarray(matrix, dtype=np.float64).std(axis=0, ddof=0)
    replaced = np.flatnonzero(scale < 1e-6)
    scale[replaced] = 1.0
    return center, scale, [PFFD10_FEATURES[index] for index in replaced]


def _psd_eigh(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values, vectors = np.linalg.eigh((matrix + matrix.T) * 0.5)
    if values.min() < -1e-6:
        raise FloatingPointError(f"Covariance eigenvalue defect {values.min()}")
    return np.maximum(values, 0.0), vectors


def frechet_distance(left: np.ndarray, right: np.ndarray, epsilon: float = 1e-8) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.ndim != 2 or right.ndim != 2 or left.shape[1] != right.shape[1]:
        raise ValueError("Fréchet inputs must be two feature matrices with equal width")
    if min(len(left), len(right)) < 2 or not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("Fréchet inputs must be finite with at least two rows")
    covariance_left = np.cov(left, rowvar=False) + np.eye(left.shape[1]) * epsilon
    covariance_right = np.cov(right, rowvar=False) + np.eye(right.shape[1]) * epsilon
    eigenvalues, eigenvectors = _psd_eigh(covariance_left)
    root_left = (eigenvectors * np.sqrt(eigenvalues)) @ eigenvectors.T
    inner_values, _ = _psd_eigh(root_left @ covariance_right @ root_left)
    mean_difference = left.mean(axis=0) - right.mean(axis=0)
    value = float(
        mean_difference @ mean_difference
        + np.trace(covariance_left)
        + np.trace(covariance_right)
        - 2.0 * np.sqrt(inner_values).sum()
    )
    if value < -1e-6:
        raise FloatingPointError(f"Negative Fréchet distance {value}")
    return max(0.0, value)


def polynomial_kid(left: np.ndarray, right: np.ndarray) -> float:
    """Unbiased full-sample polynomial-kernel MMD squared."""
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.ndim != 2 or right.ndim != 2 or left.shape[1] != right.shape[1]:
        raise ValueError("KID inputs must be two feature matrices with equal width")
    if min(len(left), len(right)) < 2:
        raise ValueError("KID needs at least two rows in each population")
    dimension = left.shape[1]
    kxx = (left @ left.T / dimension + 1.0) ** 3
    kyy = (right @ right.T / dimension + 1.0) ** 3
    kxy = (left @ right.T / dimension + 1.0) ** 3
    within_left = (kxx.sum() - np.trace(kxx)) / (len(left) * (len(left) - 1))
    within_right = (kyy.sum() - np.trace(kyy)) / (len(right) * (len(right) - 1))
    return float(within_left + within_right - 2.0 * kxy.mean())


class Inception768:
    def __init__(self, weight_path: Path, expected_sha256: str, device: torch.device):
        from torchvision.models import inception_v3

        if sha256_file(weight_path) != expected_sha256:
            raise RuntimeError("Inception-v3 weight hash mismatch")
        self.device = device
        self.model = inception_v3(weights=None, aux_logits=True, init_weights=False, transform_input=False)
        self.model.load_state_dict(torch.load(weight_path, map_location="cpu", weights_only=True))
        self.model.eval().to(device)
        self.activation: torch.Tensor | None = None
        self.hook = self.model.Mixed_6e.register_forward_hook(self._capture)

    def _capture(self, _module: Any, _inputs: Any, output: torch.Tensor) -> None:
        self.activation = output

    @torch.inference_mode()
    def embeddings(self, arrays_uint8: np.ndarray, batch_size: int = 32) -> np.ndarray:
        output = []
        for start in range(0, len(arrays_uint8), batch_size):
            batch = torch.from_numpy(np.ascontiguousarray(arrays_uint8[start:start + batch_size])).to(self.device)
            batch = batch.float().div_(255.0).unsqueeze(1).expand(-1, 3, -1, -1)
            batch = F.interpolate(batch, size=(299, 299), mode="bilinear", align_corners=False)
            mean = torch.tensor((0.485, 0.456, 0.406), device=self.device)[None, :, None, None]
            std = torch.tensor((0.229, 0.224, 0.225), device=self.device)[None, :, None, None]
            self.activation = None
            self.model((batch - mean) / std)
            if self.activation is None or self.activation.shape[1] != 768:
                raise RuntimeError("Inception Mixed_6e activation was not captured")
            output.append(F.adaptive_avg_pool2d(self.activation.float(), 1).flatten(1).cpu().numpy())
        return np.concatenate(output).astype(np.float64)

    def close(self) -> None:
        self.hook.remove()
        del self.model


_BIT_COUNTS = np.asarray([int(index).bit_count() for index in range(256)], dtype=np.uint8)


def pixel_hash(pixels: np.ndarray) -> str:
    pixels = np.ascontiguousarray(pixels, dtype=np.uint8)
    return hashlib.sha256(b"uint8:L:64:64:" + pixels.tobytes()).hexdigest()


def phash(pixels: np.ndarray) -> np.uint64:
    from scipy.fft import dctn

    small = Image.fromarray(pixels).resize((32, 32), Image.Resampling.LANCZOS)
    coefficient = dctn(np.asarray(small, dtype=np.float32), axes=(0, 1), norm="ortho")[:8, :8]
    bits = coefficient > np.median(coefficient[1:, 1:])
    bits[0, 0] = False
    value = 0
    for bit in bits.flat:
        value = (value << 1) | int(bit)
    return np.uint64(value)


class ResNet18Screen:
    def __init__(self, weight_path: Path, expected_sha256: str, device: torch.device):
        from torchvision.models import ResNet18_Weights, resnet18

        if sha256_file(weight_path) != expected_sha256:
            raise RuntimeError("ResNet-18 weight hash mismatch")
        self.device = device
        self.model = resnet18(weights=None)
        self.model.load_state_dict(torch.load(weight_path, map_location="cpu", weights_only=True))
        self.model.fc = torch.nn.Identity()
        self.model.eval().to(device)
        self.transform = ResNet18_Weights.IMAGENET1K_V1.transforms()

    @torch.inference_mode()
    def embeddings(self, arrays_uint8: np.ndarray, batch_size: int = 64) -> np.ndarray:
        output = []
        for start in range(0, len(arrays_uint8), batch_size):
            images = [Image.fromarray(value).convert("RGB") for value in arrays_uint8[start:start + batch_size]]
            batch = torch.stack([self.transform(value) for value in images]).to(self.device)
            output.append(self.model(batch).float().cpu().numpy())
        values = np.concatenate(output).astype(np.float64)
        return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)

    def close(self) -> None:
        del self.model


def hamming_nearest(queries: np.ndarray, reference: np.ndarray, block_size: int = 128) -> tuple[np.ndarray, np.ndarray]:
    distances, indices = [], []
    for start in range(0, len(queries), block_size):
        block = np.bitwise_xor(queries[start:start + block_size, None], reference[None, :])
        counts = _BIT_COUNTS[block.view(np.uint8).reshape(block.shape + (8,))].sum(axis=2, dtype=np.uint16)
        selected = np.argmin(counts, axis=1)
        indices.append(selected)
        distances.append(counts[np.arange(len(selected)), selected])
    return np.concatenate(distances), np.concatenate(indices)


def nearest_similarity(
    arrays_uint8: np.ndarray,
    embeddings: np.ndarray,
    flipped_embeddings: np.ndarray,
    bank_pixels: np.ndarray,
    bank_embeddings: np.ndarray,
    bank_hashes: set[str],
    bank_phashes: np.ndarray,
) -> dict[str, np.ndarray]:
    distance_original = np.clip(1.0 - embeddings @ bank_embeddings.T, 0.0, 2.0)
    distance_flipped = np.clip(1.0 - flipped_embeddings @ bank_embeddings.T, 0.0, 2.0)
    nearest_original = np.argmin(distance_original, axis=1)
    nearest_flipped = np.argmin(distance_flipped, axis=1)
    values_original = distance_original[np.arange(len(arrays_uint8)), nearest_original]
    values_flipped = distance_flipped[np.arange(len(arrays_uint8)), nearest_flipped]
    use_flip = values_flipped < values_original
    nearest = np.where(use_flip, nearest_flipped, nearest_original)
    feature_min = np.where(use_flip, values_flipped, values_original)
    oriented = np.where(use_flip[:, None, None], arrays_uint8[:, :, ::-1], arrays_uint8)
    nearest_ssim = np.asarray([
        structural_similarity(
            oriented[index].astype(np.float32) / 255.0,
            bank_pixels[reference].astype(np.float32) / 255.0,
            data_range=1.0,
            win_size=7,
            gaussian_weights=False,
            use_sample_covariance=True,
        )
        for index, reference in enumerate(nearest)
    ])
    original_hash = np.asarray([pixel_hash(value) in bank_hashes for value in arrays_uint8])
    flipped_hash = np.asarray([pixel_hash(value[:, ::-1]) in bank_hashes for value in arrays_uint8])
    query_phash = np.asarray([phash(value) for value in arrays_uint8] + [phash(value[:, ::-1]) for value in arrays_uint8])
    hamming, hamming_index = hamming_nearest(query_phash, bank_phashes)
    count = len(arrays_uint8)
    choose_flip_phash = hamming[count:] < hamming[:count]
    phash_min = np.where(choose_flip_phash, hamming[count:], hamming[:count])
    phash_reference = np.where(choose_flip_phash, hamming_index[count:], hamming_index[:count])
    return {
        "feature_min": feature_min,
        "feature_reference_index": nearest.astype(np.int32),
        "feature_orientation_flip": use_flip,
        "nearest_reference_ssim": nearest_ssim,
        "exact_original_or_flip": np.logical_or(original_hash, flipped_hash),
        "phash_min": phash_min.astype(np.int16),
        "phash_reference_index": phash_reference.astype(np.int32),
    }


def radial_log_power(images: np.ndarray) -> np.ndarray:
    images = np.asarray(images, dtype=np.float64)
    height, width = images.shape[1:]
    yy, xx = np.mgrid[:height, :width]
    radius = np.floor(np.sqrt((yy - height // 2) ** 2 + (xx - width // 2) ** 2)).astype(int)
    bins = min(height, width) // 2
    result = np.empty((len(images), bins), dtype=np.float64)
    for index, image in enumerate(images):
        spectrum = np.abs(np.fft.fftshift(np.fft.fft2(image - image.mean()))) ** 2
        for band in range(bins):
            result[index, band] = math.log1p(float(spectrum[radius == band].mean()))
    return result


def paired_measurements(
    generated: np.ndarray,
    real: np.ndarray,
    generated_morphology: np.ndarray,
    target_rows: list[dict[str, str]],
) -> dict[str, np.ndarray]:
    generated_laplacian = np.var(laplace(generated), axis=(1, 2))
    real_laplacian = np.var(laplace(real), axis=(1, 2))
    power_difference = np.mean(np.abs(radial_log_power(generated) - radial_log_power(real)), axis=1)
    feature_index = {name: index for index, name in enumerate(PFFD10_FEATURES)}
    target_x = np.asarray([
        (float(row["bbox_x_px"]) + 0.5 * float(row["bbox_width_px"])) / 300.0 for row in target_rows
    ])
    target_y = np.asarray([
        (float(row["bbox_y_px"]) + 0.5 * float(row["bbox_height_px"])) / 300.0 for row in target_rows
    ])
    target_width = np.asarray([float(row["bbox_width_px"]) / 300.0 for row in target_rows])
    target_height = np.asarray([float(row["bbox_height_px"]) / 300.0 for row in target_rows])
    centroid_x = generated_morphology[:, feature_index["centroid_x"]]
    centroid_y = generated_morphology[:, feature_index["centroid_y"]]
    width = generated_morphology[:, feature_index["bbox_width_fraction"]]
    height = generated_morphology[:, feature_index["bbox_height_fraction"]]
    return {
        "radial_log_power_l1": power_difference,
        "laplacian_variance_absolute_difference": np.abs(generated_laplacian - real_laplacian),
        "centroid_distance_to_inherited_bbox_center": np.sqrt((centroid_x - target_x) ** 2 + (centroid_y - target_y) ** 2),
        "bright_bbox_width_absolute_error": np.abs(width - target_width),
        "bright_bbox_height_absolute_error": np.abs(height - target_height),
    }


def validate_inputs(config: dict[str, Any]) -> tuple[dict[str, Any], str, list[dict[str, str]], list[dict[str, str]]]:
    lock, protocol_hash = verify_protocol()
    if config["seeds"] != lock["seed_rosters"]["primary_full_pipeline"]:
        raise RuntimeError("W11 quality seed roster differs from protocol lock")
    pool_receipt_path = EVIDENCE_ROOT / "W11_POOL_COMPLETENESS.json"
    if not pool_receipt_path.exists():
        raise RuntimeError("W11 pools have not been summarized")
    pool_receipt = json.loads(pool_receipt_path.read_text(encoding="utf-8"))
    if pool_receipt["status"] != "COMPLETE" or pool_receipt.get("test_payload_accessed") is not False:
        raise RuntimeError("W11 pool receipt is not complete and test-free")
    source_hash_path = EVIDENCE_ROOT / "source_array_hashes.json"
    source_hashes = json.loads(source_hash_path.read_text(encoding="utf-8"))["arrays"]
    for relative, digest in source_hashes.items():
        if sha256_file(ROOT / relative) != digest:
            raise RuntimeError(f"Changed W11 source array: {relative}")
    inputs = verify_inputs()
    train_rows = read_csv(ROOT / inputs["outputs"]["train"]["path"])
    validation_rows = read_csv(ROOT / inputs["outputs"]["validation"]["path"])
    if len(train_rows) != 3584 or len(validation_rows) != 1024:
        raise RuntimeError("Unexpected W11 train/validation population")
    return lock, protocol_hash, train_rows, validation_rows


def build_real_cache(
    config: dict[str, Any],
    train_rows: list[dict[str, str]],
    validation_rows: list[dict[str, str]],
    device: torch.device,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    cache_path = CACHE_ROOT / "real_feature_cache.npz"
    identity = {
        "schema_version": 1,
        "config_sha256": sha256_file(CONFIG_PATH),
        "train_manifest_sha256": sha256_file(ROOT / "data" / "manifests" / "w09" / "train.csv"),
        "validation_manifest_sha256": sha256_file(ROOT / "data" / "manifests" / "w09" / "validation.csv"),
        "inception_weight_sha256": config["distribution_metrics"]["inception_fid_kid"]["weights_sha256"],
        "resnet_weight_sha256": config["memorization_screen"]["feature_weights_sha256"],
    }
    identity_hash = object_hash(identity)
    identity_path = CACHE_ROOT / "real_feature_cache_identity.json"
    if cache_path.exists() and identity_path.exists():
        stored = json.loads(identity_path.read_text(encoding="utf-8"))
        if stored.get("identity_sha256") == identity_hash:
            with np.load(cache_path, allow_pickle=False) as data:
                return {key: data[key] for key in data.files}, identity
        raise RuntimeError("Existing W11 real-feature cache has a different identity")
    train_uint8 = np.rint(image_tensor(train_rows).numpy()[:, 0] * 255).astype(np.uint8)
    validation_uint8 = np.rint(image_tensor(validation_rows).numpy()[:, 0] * 255).astype(np.uint8)
    train_morphology = morphology_matrix(train_uint8.astype(np.float32) / 255.0)
    validation_morphology = morphology_matrix(validation_uint8.astype(np.float32) / 255.0)
    center, scale, replaced = fit_scaler(train_morphology)
    inception_spec = config["distribution_metrics"]["inception_fid_kid"]
    inception = Inception768(ROOT / inception_spec["weights_path"], inception_spec["weights_sha256"], device)
    validation_inception = inception.embeddings(validation_uint8)
    inception.close()
    screen_spec = config["memorization_screen"]
    screen = ResNet18Screen(ROOT / screen_spec["feature_weights_path"], screen_spec["feature_weights_sha256"], device)
    train_screen = screen.embeddings(train_uint8)
    validation_screen = screen.embeddings(validation_uint8)
    validation_screen_flip = screen.embeddings(validation_uint8[:, :, ::-1].copy())
    screen.close()
    train_phash = np.asarray([phash(value) for value in train_uint8], dtype=np.uint64)
    cached = {
        "train_uint8": train_uint8,
        "validation_uint8": validation_uint8,
        "train_morphology": train_morphology,
        "validation_morphology": validation_morphology,
        "pffd_center": center,
        "pffd_scale": scale,
        "validation_inception": validation_inception,
        "train_screen": train_screen,
        "validation_screen": validation_screen,
        "validation_screen_flip": validation_screen_flip,
        "train_phash": train_phash,
        "train_ids": np.asarray([row["sample_id"] for row in train_rows]),
    }
    atomic_npz(cache_path, **cached)
    atomic_json(identity_path, {**identity, "identity_sha256": identity_hash, "cache_sha256": sha256_file(cache_path)})
    atomic_json(EVIDENCE_ROOT / "pffd10_scaler.json", {
        "schema_version": 1,
        "features": list(PFFD10_FEATURES),
        "center": center.tolist(),
        "scale": scale.tolist(),
        "replaced_scale_features": replaced,
        "fit_scope": "train_real_only",
        "n": len(train_rows),
        "config_sha256": sha256_file(CONFIG_PATH),
    })
    return cached, identity


def calibrate_screen(
    cache: dict[str, np.ndarray],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    bank_hashes = {pixel_hash(value) for value in cache["train_uint8"]}
    measured = nearest_similarity(
        cache["validation_uint8"],
        cache["validation_screen"],
        cache["validation_screen_flip"],
        cache["train_uint8"],
        cache["train_screen"],
        bank_hashes,
        cache["train_phash"],
    )
    feature_q01 = float(np.quantile(measured["feature_min"], 0.01, method="linear"))
    ssim_q99 = float(np.quantile(measured["nearest_reference_ssim"], 0.99, method="linear"))
    phash_q01 = int(np.quantile(measured["phash_min"], 0.01, method="lower"))
    suspicious = (
        measured["exact_original_or_flip"]
        | (measured["feature_min"] <= feature_q01)
        | (measured["nearest_reference_ssim"] >= ssim_q99)
    )
    lock = {
        "schema_version": 1,
        "status": "FROZEN_VALIDATION_CALIBRATION",
        "decision_rule": "exact_original_or_horizontal_flip OR feature_min_le_q01 OR nearest_reference_ssim_ge_q99",
        "thresholds": {"feature_q01": feature_q01, "nearest_reference_ssim_q99": ssim_q99},
        "phash_q01_diagnostic_only": phash_q01,
        "quantile_methods": {"feature": "linear", "ssim": "linear", "phash": "lower"},
        "calibration_rows": len(suspicious),
        "calibration_flag_count": int(suspicious.sum()),
        "calibration_flag_rate": float(suspicious.mean()),
        "reference_split": "train",
        "query_split": "validation",
        "test_rows_used": 0,
        "interpretation": "in_sample_validation_operating_rate_not_independent_false_positive_rate",
    }
    atomic_json(EVIDENCE_ROOT / "memorization_threshold_lock.json", lock)
    return lock, measured


def pool_array(seed: int, arm: str, expected_count: int) -> np.ndarray:
    path = RUN_ROOT / f"s{seed}" / f"quality_{arm}_images_uint8.npy"
    value = np.load(path, allow_pickle=False)
    if value.shape != (expected_count, 1, 64, 64) or value.dtype != np.uint8:
        raise RuntimeError(f"Invalid W11 quality array {path}")
    return value[:, 0]


def analyze() -> dict[str, Any]:
    config = load_config()
    lock, protocol_hash, train_rows, validation_rows = validate_inputs(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" or torch.cuda.get_device_name(device) != "NVIDIA GeForce RTX 5080":
        raise RuntimeError("Frozen W11 quality evaluation requires the local RTX 5080")
    cache, cache_identity = build_real_cache(config, train_rows, validation_rows, device)
    screen_lock, calibration = calibrate_screen(cache)
    bank_hashes = {pixel_hash(value) for value in cache["train_uint8"]}
    center, scale = cache["pffd_center"], cache["pffd_scale"]
    real_pffd = (cache["validation_morphology"] - center) / scale
    inception_spec = config["distribution_metrics"]["inception_fid_kid"]
    screen_spec = config["memorization_screen"]
    inception = Inception768(ROOT / inception_spec["weights_path"], inception_spec["weights_sha256"], device)
    screen = ResNet18Screen(ROOT / screen_spec["feature_weights_path"], screen_spec["feature_weights_sha256"], device)
    aggregate_rows: list[dict[str, Any]] = []
    per_image_rows: list[dict[str, Any]] = []
    try:
        for seed in config["seeds"]:
            for arm in config["arms"]:
                generated_uint8 = pool_array(seed, arm, len(validation_rows))
                generated = generated_uint8.astype(np.float64) / 255.0
                morphology = morphology_matrix(generated)
                pffd10 = frechet_distance(real_pffd, (morphology - center) / scale)
                inception_features = inception.embeddings(generated_uint8)
                fid = frechet_distance(cache["validation_inception"], inception_features)
                kid = polynomial_kid(cache["validation_inception"], inception_features)
                screen_original = screen.embeddings(generated_uint8)
                screen_flipped = screen.embeddings(generated_uint8[:, :, ::-1].copy())
                similarity = nearest_similarity(
                    generated_uint8,
                    screen_original,
                    screen_flipped,
                    cache["train_uint8"],
                    cache["train_screen"],
                    bank_hashes,
                    cache["train_phash"],
                )
                suspicious = (
                    similarity["exact_original_or_flip"]
                    | (similarity["feature_min"] <= screen_lock["thresholds"]["feature_q01"])
                    | (similarity["nearest_reference_ssim"] >= screen_lock["thresholds"]["nearest_reference_ssim_q99"])
                )
                paired = paired_measurements(generated, cache["validation_uint8"].astype(np.float64) / 255.0,
                                              morphology, validation_rows)
                row = {
                    "evidence_layer": "N2_NEW_STUDY",
                    "data_role": "validation",
                    "arm": arm,
                    "seed": seed,
                    "n_images": len(generated),
                    "PFFD10_train_scaled": pffd10,
                    "FID_InceptionV3_Mixed6e_768": fid,
                    "KID_InceptionV3_Mixed6e_768": kid,
                    "radial_log_power_l1_mean": float(paired["radial_log_power_l1"].mean()),
                    "laplacian_variance_absolute_difference_mean": float(paired["laplacian_variance_absolute_difference"].mean()),
                    "centroid_distance_to_inherited_bbox_center_mean": float(paired["centroid_distance_to_inherited_bbox_center"].mean()),
                    "bright_bbox_width_absolute_error_mean": float(paired["bright_bbox_width_absolute_error"].mean()),
                    "bright_bbox_height_absolute_error_mean": float(paired["bright_bbox_height_absolute_error"].mean()),
                    "memorization_screen_flag_count": int(suspicious.sum()),
                    "memorization_screen_flag_rate": float(suspicious.mean()),
                    "exact_train_or_flip_count": int(similarity["exact_original_or_flip"].sum()),
                    "feature_min_mean": float(similarity["feature_min"].mean()),
                    "nearest_reference_ssim_max": float(similarity["nearest_reference_ssim"].max()),
                    "phash_minimum": int(similarity["phash_min"].min()),
                    "source_array_sha256": sha256_file(RUN_ROOT / f"s{seed}" / f"quality_{arm}_images_uint8.npy"),
                    "test_payload_accessed": False,
                }
                aggregate_rows.append(row)
                for index, source in enumerate(validation_rows):
                    per_image_rows.append({
                        "data_role": "validation",
                        "arm": arm,
                        "seed": seed,
                        "target_sample_id": source["sample_id"],
                        "specimen": source["specimen"],
                        "view": source["view"],
                        **{name: float(values[index]) for name, values in paired.items()},
                        "feature_min": float(similarity["feature_min"][index]),
                        "nearest_reference_ssim": float(similarity["nearest_reference_ssim"][index]),
                        "exact_original_or_flip": bool(similarity["exact_original_or_flip"][index]),
                        "phash_min": int(similarity["phash_min"][index]),
                        "memorization_screen_flag": bool(suspicious[index]),
                        "nearest_train_id": str(cache["train_ids"][similarity["feature_reference_index"][index]]),
                        "nearest_orientation": "horizontal_flip" if similarity["feature_orientation_flip"][index] else "original",
                    })
                print(json.dumps({"W11_quality": "cell_complete", "arm": arm, "seed": seed,
                                  "PFFD10": pffd10, "FID768": fid, "KID768": kid}), flush=True)
    finally:
        inception.close()
        screen.close()
        torch.cuda.empty_cache()

    aggregate = pd.DataFrame(aggregate_rows).sort_values(["arm", "seed"]).reset_index(drop=True)
    if len(aggregate) != 50:
        raise RuntimeError("W11 aggregate grid must have 50 rows")
    metric_columns = [
        "PFFD10_train_scaled",
        "FID_InceptionV3_Mixed6e_768",
        "KID_InceptionV3_Mixed6e_768",
        "radial_log_power_l1_mean",
        "laplacian_variance_absolute_difference_mean",
        "centroid_distance_to_inherited_bbox_center_mean",
        "bright_bbox_width_absolute_error_mean",
        "bright_bbox_height_absolute_error_mean",
        "memorization_screen_flag_rate",
        "exact_train_or_flip_count",
        "feature_min_mean",
        "nearest_reference_ssim_max",
        "phash_minimum",
    ]
    summaries = []
    for arm, group in aggregate.groupby("arm", sort=True):
        summary: dict[str, Any] = {"arm": arm, "n_seeds": len(group), "images_per_seed": 1024}
        for metric in metric_columns:
            summary[f"{metric}_mean"] = float(group[metric].mean())
            summary[f"{metric}_sd"] = float(group[metric].std(ddof=1))
        summaries.append(summary)

    diversity_rows = []
    for arm in config["arms"]:
        arrays = [pool_array(seed, arm, len(validation_rows)).astype(np.float32) / 255.0 for seed in config["seeds"]]
        target_values = []
        for target_index in range(len(validation_rows)):
            pair_values = []
            for left in range(len(arrays)):
                for right in range(left + 1, len(arrays)):
                    pair_values.append(1.0 - structural_similarity(
                        arrays[left][target_index], arrays[right][target_index], data_range=1.0,
                        win_size=7, gaussian_weights=False, use_sample_covariance=True,
                    ))
            target_values.append(float(np.mean(pair_values)))
        diversity_rows.append({
            "arm": arm,
            "n_targets": len(target_values),
            "seed_pairs_per_target": 45,
            "within_target_one_minus_SSIM_mean": float(np.mean(target_values)),
            "within_target_one_minus_SSIM_sd_across_targets": float(np.std(target_values, ddof=1)),
        })

    aggregate_path = EVIDENCE_ROOT / "generation_metrics.parquet"
    per_image_path = EVIDENCE_ROOT / "generation_per_image.parquet"
    grid_path = EVIDENCE_ROOT / "quality_grid.csv"
    summary_path = EVIDENCE_ROOT / "quality_arm_summary.csv"
    diversity_path = EVIDENCE_ROOT / "within_target_diversity.csv"
    atomic_parquet(aggregate_path, aggregate)
    atomic_parquet(per_image_path, pd.DataFrame(per_image_rows))
    atomic_csv(grid_path, aggregate)
    atomic_csv(summary_path, pd.DataFrame(summaries))
    atomic_csv(diversity_path, pd.DataFrame(diversity_rows))

    label_validity = {
        "schema_version": 1,
        "status": "SCOPE_LIMITED_INHERITED_OPERATIONAL_TARGET_ONLY",
        "human_review": "EXCLUDED_BY_USER_SCOPE",
        "label_source": "inherited_target",
        "label_unit": "inherited_bbox_region",
        "physical_semantics_verified": False,
        "pool_manifest_rows": 10 * 5 * 3584,
        "quality_roster_rows_evaluated": len(aggregate) * 1024,
        "roster_alignment": "PASS",
        "permitted_claim": "Each generated byte array is aligned to the frozen inherited-box target roster.",
        "prohibited_claims": config["label_validity"]["prohibited_claims"],
        "detector_training_scope": "authorized_only_for_operational_inherited_bbox_region_with_explicit_manuscript_limitation",
        "test_payload_accessed": False,
    }
    atomic_json(EVIDENCE_ROOT / "NEW_POOL_LABEL_VALIDITY.json", label_validity)
    outputs = {
        str(path.relative_to(ROOT)).replace("\\", "/"): sha256_file(path)
        for path in (aggregate_path, per_image_path, grid_path, summary_path, diversity_path,
                     EVIDENCE_ROOT / "pffd10_scaler.json",
                     EVIDENCE_ROOT / "memorization_threshold_lock.json",
                     EVIDENCE_ROOT / "NEW_POOL_LABEL_VALIDITY.json")
    }
    receipt = {
        "schema_version": 1,
        "status": "PASS_WITH_SCOPE_LIMITATION",
        "protocol_sha256": protocol_hash,
        "quality_config_sha256": sha256_file(CONFIG_PATH),
        "real_cache_identity_sha256": object_hash(cache_identity),
        "cells_completed": len(aggregate),
        "cells_planned": 50,
        "common_quality_roster_images_per_cell": 1024,
        "distribution_population_pairing": "same_validation_target_roster",
        "memorization_calibration_flag_rate": screen_lock["calibration_flag_rate"],
        "label_validity_status": label_validity["status"],
        "human_review": "EXCLUDED_BY_USER_SCOPE",
        "test_payload_accessed": False,
        "outputs": outputs,
        "interpretive_limits": [
            "Inception metrics use ImageNet Inception-v3 Mixed_6e features and are descriptive for this grayscale domain.",
            "The automated similarity screen does not prove memorization or its absence.",
            "Operational geometry is relative to inherited boxes whose physical object semantics were not independently reviewed.",
        ],
    }
    atomic_json(EVIDENCE_ROOT / "W11_QUALITY_COMPLETENESS.json", receipt)
    return receipt


def preflight() -> dict[str, Any]:
    config = load_config()
    lock, protocol_hash = verify_protocol()
    inception = ROOT / config["distribution_metrics"]["inception_fid_kid"]["weights_path"]
    resnet = ROOT / config["memorization_screen"]["feature_weights_path"]
    result = {
        "status": "PASS",
        "protocol_sha256": protocol_hash,
        "quality_config_sha256": sha256_file(CONFIG_PATH),
        "seeds_match": config["seeds"] == lock["seed_rosters"]["primary_full_pipeline"],
        "inception_weight_hash_match": inception.is_file() and sha256_file(inception) == config["distribution_metrics"]["inception_fid_kid"]["weights_sha256"],
        "resnet_weight_hash_match": resnet.is_file() and sha256_file(resnet) == config["memorization_screen"]["feature_weights_sha256"],
        "test_access": "PROHIBITED",
    }
    if not all((result["seeds_match"], result["inception_weight_hash_match"], result["resnet_weight_hash_match"])):
        result["status"] = "FAIL"
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--preflight", action="store_true")
    action.add_argument("--run", action="store_true")
    args = parser.parse_args()
    result = preflight() if args.preflight else analyze()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] != "FAIL" else 1


if __name__ == "__main__":
    raise SystemExit(main())
