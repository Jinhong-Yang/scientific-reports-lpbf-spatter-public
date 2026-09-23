"""Analyze the complete W10 prior sweep and matched smoothing controls."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "vendor" / "parquet"))
import pandas as pd
from scipy.ndimage import gaussian_filter
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "generators"))
sys.path.insert(0, str(ROOT / "src" / "analysis"))
from run_w09_factorial import (  # noqa: E402
    atomic_json,
    condition_tensor,
    image_tensor,
    read_csv,
    sha256_file,
    verify_inputs,
    verify_protocol,
)
from run_w10_prior_sweep import RUN_ROOT, config, lambda_slug, roster  # noqa: E402
from materialize_w09_validation_arrays import build_model, reconstruct  # noqa: E402


OUTPUT_ROOT = ROOT / "evidence" / "prior_sweep"
ARRAY_ROOT = OUTPUT_ROOT / "validation_arrays"


def atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def finite_difference_diagnostics(images: np.ndarray, conditions: np.ndarray) -> dict[str, np.ndarray]:
    """Return per-image common optical residual, roughness and spectral metrics."""
    values = images[:, 0].astype(np.float64)
    height, width = values.shape[-2:]
    dy, dx = 2.0 / (height - 1), 2.0 / (width - 1)
    gy, gx = np.gradient(values, dy, dx, axis=(-2, -1), edge_order=1)
    gyy = np.gradient(gy, dy, axis=-2, edge_order=1)
    gxx = np.gradient(gx, dx, axis=-1, edge_order=1)
    laplacian = gxx + gyy
    area = np.expm1(conditions[:, 9] * math.log(90001.0)) / 90000.0
    base = np.clip(np.sqrt(np.maximum(area, 16 / 90000)), 0.018, 0.55)
    sigma_x = np.clip(base * np.exp(0.25 * conditions[:, 10]), 0.018, 0.60)
    sigma_y = np.clip(base * np.exp(-0.25 * conditions[:, 10]), 0.018, 0.60)
    diffusivity = 0.018 + 0.010 * conditions[:, 4] + 0.004 * conditions[:, 5]
    transport_x = (0.05 + 0.15 * conditions[:, 1]) * conditions[:, 12]
    transport_y = (0.05 + 0.15 * conditions[:, 1]) * conditions[:, 11]
    source = 0.08 + 0.42 * conditions[:, 2] + 0.05 * conditions[:, 0]
    decay = 0.65 + 0.35 * conditions[:, 1]
    y = np.linspace(-1, 1, height)[None, :, None]
    x = np.linspace(-1, 1, width)[None, None, :]
    gaussian = np.exp(-0.5 * (
        ((x - conditions[:, 7, None, None]) / sigma_x[:, None, None]) ** 2
        + ((y - conditions[:, 8, None, None]) / sigma_y[:, None, None]) ** 2
    ))
    residual = (
        -diffusivity[:, None, None] * laplacian
        + transport_x[:, None, None] * gx
        + transport_y[:, None, None] * gy
        + decay[:, None, None] * values
        - source[:, None, None] * gaussian
    )
    fft = np.fft.fftshift(np.fft.fft2(values, axes=(-2, -1)), axes=(-2, -1))
    power = np.abs(fft) ** 2
    yy, xx = np.ogrid[:height, :width]
    radius = np.sqrt((yy - (height - 1) / 2) ** 2 + (xx - (width - 1) / 2) ** 2)
    high = radius >= 0.35 * radius.max()
    spectral_fraction = power[:, high].sum(1) / np.maximum(power.sum(axis=(-2, -1)), 1e-12)
    return {
        "common_optical_residual_rms": np.sqrt(np.mean(residual ** 2, axis=(-2, -1))),
        "own_laplacian_rms": np.sqrt(np.mean(laplacian ** 2, axis=(-2, -1))),
        "gradient_rms": np.sqrt(np.mean(gx ** 2 + gy ** 2, axis=(-2, -1))),
        "laplacian_variance": np.var(laplacian, axis=(-2, -1)),
        "high_frequency_psd_fraction": spectral_fraction,
    }


def blur_images(images: np.ndarray, sigma: float) -> np.ndarray:
    if sigma == 0:
        return images.copy()
    return gaussian_filter(images, sigma=(0, 0, sigma, sigma), mode="reflect")


def mse_per_image(predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
    return np.mean((predictions.astype(np.float64) - targets.astype(np.float64)) ** 2, axis=(1, 2, 3))


def analyze() -> dict[str, Any]:
    lock, protocol_hash = verify_protocol()
    completeness_path = OUTPUT_ROOT / "W10_SWEEP_COMPLETENESS.json"
    completeness = json.loads(completeness_path.read_text(encoding="utf-8"))
    if completeness["status"] != "COMPLETE" or completeness["completed"] != 18:
        raise RuntimeError("W10 sweep is not complete")
    inputs = verify_inputs()
    validation_rows = read_csv(ROOT / inputs["outputs"]["validation"]["path"])
    targets_t, conditions_t = image_tensor(validation_rows), condition_tensor(validation_rows)
    targets, conditions = targets_t.numpy(), conditions_t.numpy()
    sample_ids = np.asarray([row["sample_id"] for row in validation_rows])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" or torch.cuda.get_device_name(device) != "NVIDIA GeForce RTX 5080":
        raise RuntimeError("Frozen local RTX 5080 is required")

    path_rows: list[dict[str, Any]] = []
    arrays: dict[tuple[float, int], np.ndarray] = {}
    mechanism_rows: list[dict[str, Any]] = []
    for weight, seed in roster():
        directory = RUN_ROOT / f"{lambda_slug(weight)}_s{seed}"
        summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
        checkpoint_path = directory / "best.pt"
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model = build_model(checkpoint, device)
        predictions, measured = reconstruct(model, targets_t, conditions_t, device)
        if abs(measured - float(summary["best_validation_mse"])) > max(1e-9, measured * 1e-5):
            raise RuntimeError(f"W10 validation drift: {weight}, {seed}")
        arrays[(weight, seed)] = predictions
        output = ARRAY_ROOT / f"{lambda_slug(weight)}_s{seed}.npz"
        atomic_npz(output, sample_id=sample_ids, reconstruction=predictions,
                   measured_validation_mse=np.asarray(measured, dtype=np.float64))
        path_rows.append({"proxy_lambda": weight, "seed": seed, "best_epoch": summary["best_epoch"],
                          "validation_mse": measured, "gpu_hours": summary["gpu_hours"],
                          "array_sha256": sha256_file(output), "checkpoint_sha256": sha256_file(checkpoint_path)})
        diagnostics = finite_difference_diagnostics(predictions, conditions)
        permuted = conditions.copy()
        permuted[:, :3] = np.roll(permuted[:, :3], 1, axis=0)
        permuted_residual = finite_difference_diagnostics(predictions, permuted)["common_optical_residual_rms"]
        for index, sample_id in enumerate(sample_ids):
            mechanism_rows.append({
                "family": "optical_prior_path", "proxy_lambda": weight, "seed": seed,
                "sample_id": sample_id, "validation_mse": mse_per_image(predictions[index:index + 1], targets[index:index + 1])[0],
                **{key: values[index] for key, values in diagnostics.items()},
                "permuted_condition_optical_residual_rms": permuted_residual[index],
            })
        del model, checkpoint
        torch.cuda.empty_cache()

    controls = config()["smoothing_controls"]
    control_rows: list[dict[str, Any]] = []
    for seed in config()["seeds"]:
        base = arrays[(0.0, seed)].astype(np.float32)
        for family, strengths in (
            ("gaussian_sigma_px", controls["gaussian_sigma_px"]),
            ("heat_tau_px2", controls["laplacian_heat_tau_px2"]),
        ):
            for strength in strengths:
                sigma = float(strength) if family == "gaussian_sigma_px" else math.sqrt(2 * float(strength))
                smoothed = blur_images(base, sigma)
                per_image_mse = mse_per_image(smoothed, targets)
                diagnostics = finite_difference_diagnostics(smoothed, conditions)
                control_rows.append({"control_family": family, "strength": strength, "seed": seed,
                                     "equivalent_gaussian_sigma_px": sigma,
                                     "validation_mse": float(per_image_mse.mean()),
                                     "common_optical_residual_rms": float(diagnostics["common_optical_residual_rms"].mean()),
                                     "own_laplacian_rms": float(diagnostics["own_laplacian_rms"].mean()),
                                     "gradient_rms": float(diagnostics["gradient_rms"].mean()),
                                     "high_frequency_psd_fraction": float(diagnostics["high_frequency_psd_fraction"].mean())})

    target = float(np.mean([row["validation_mse"] for row in path_rows if row["proxy_lambda"] == 0.0035]))
    matches: list[dict[str, Any]] = []
    for family in ("gaussian_sigma_px", "heat_tau_px2"):
        aggregate: list[dict[str, Any]] = []
        for strength in sorted({row["strength"] for row in control_rows if row["control_family"] == family}):
            values = [row["validation_mse"] for row in control_rows if row["control_family"] == family and row["strength"] == strength]
            mean = float(np.mean(values))
            aggregate.append({"control_family": family, "strength": strength, "mean_validation_mse": mean,
                              "target_np_lambda": 0.0035, "target_mean_validation_mse": target,
                              "absolute_relative_gap": abs(mean - target) / target})
        selected = min(aggregate, key=lambda row: (row["absolute_relative_gap"], row["strength"]))
        selected["match_tolerance"] = controls["relative_tolerance"]
        selected["match_status"] = "MATCHED" if selected["absolute_relative_gap"] <= controls["relative_tolerance"] else "UNMATCHED_CLOSEST_ONLY"
        matches.append(selected)

    aggregate_path: list[dict[str, Any]] = []
    for weight in config()["proxy_lambda_grid"]:
        values = [row["validation_mse"] for row in path_rows if row["proxy_lambda"] == weight]
        aggregate_path.append({"proxy_lambda": weight, "seeds": len(values), "validation_mse_mean": float(np.mean(values)),
                               "validation_mse_sd": float(np.std(values, ddof=1)), "validation_mse_min": float(np.min(values)),
                               "validation_mse_max": float(np.max(values))})

    paths = {
        "regularization_path.csv": path_rows,
        "regularization_path_aggregate.csv": aggregate_path,
        "smoothing_control_metrics.csv": control_rows,
        "smoothing_match.csv": matches,
    }
    for name, rows in paths.items():
        atomic_csv(OUTPUT_ROOT / name, rows)
    mechanism_path = OUTPUT_ROOT / "mechanism_diagnostics.parquet"
    pd.DataFrame(mechanism_rows).to_parquet(mechanism_path, index=False)
    report_path = ROOT / "docs" / "W10_OPERATOR_ANALYSIS.md"
    lines = [
        "# W10 optical-prior and smoothing analysis", "",
        f"Protocol SHA-256: `{protocol_hash}`", "",
        "The analysis uses validation reconstructions only; no held-out test payload was accessed.", "",
        "## Matched-control decision", "",
    ]
    for row in matches:
        lines.append(f"- {row['control_family']}: strength {row['strength']}, relative MSE gap {row['absolute_relative_gap']:.4f}, status `{row['match_status']}`.")
    lines += ["", "A control marked `UNMATCHED_CLOSEST_ONLY` is retained diagnostically and is not described as a matched comparator.", "",
              "Common optical residuals use one frozen finite-difference scoring operator for every family. Own-operator roughness is the Laplacian RMS. Condition-triplet diagnostics rotate the power, speed, and line-energy coefficients across the frozen validation roster; this is a specificity diagnostic, not an intervention on the physical process.", ""]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    artifacts = [OUTPUT_ROOT / name for name in paths] + [mechanism_path, report_path]
    receipt = {
        "schema_version": 1, "status": "PASS", "protocol_sha256": protocol_hash,
        "sweep_runs": len(path_rows), "validation_rows": len(validation_rows),
        "test_payload_accessed": False, "matched_control_status": matches,
        "interpretation": "Mechanism and smoothing results are controlled validation diagnostics; no physical-field validation or held-out detector claim is made.",
        "artifacts": {str(path.relative_to(ROOT)).replace("\\", "/"): {"bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in artifacts},
    }
    atomic_json(OUTPUT_ROOT / "W10_ANALYSIS.json", receipt)
    return receipt


def main() -> int:
    argparse.ArgumentParser(description=__doc__).parse_args()
    print(json.dumps(analyze(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
