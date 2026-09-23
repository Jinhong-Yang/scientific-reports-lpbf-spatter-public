"""Materialize W09 validation reconstructions from immutable best checkpoints."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
GENERATOR_DIR = ROOT / "src" / "generators"
sys.path.insert(0, str(GENERATOR_DIR))
from run_w09_factorial import (  # noqa: E402
    ConditionalVariationalField,
    ModelConfig,
    RUN_ROOT,
    atomic_json,
    condition_tensor,
    image_tensor,
    read_csv,
    sha256_file,
    verify_inputs,
    verify_protocol,
)
from metal_spatter_pinn.inference import render_field  # noqa: E402


OUTPUT_ROOT = ROOT / "evidence" / "generator_factorial" / "validation_arrays"
ARMS = ("F00", "F01", "F10", "F11")
SEEDS = tuple(range(61001, 61011))


def atomic_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def build_model(checkpoint: dict[str, Any], device: torch.device) -> ConditionalVariationalField:
    raw = checkpoint["model_config"]
    value = dict(raw)
    value["fourier_frequencies"] = tuple(value["fourier_frequencies"])
    model = ConditionalVariationalField(ModelConfig(**value), int(checkpoint["image_size"])).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


@torch.inference_mode()
def reconstruct(model: ConditionalVariationalField, images: torch.Tensor, conditions: torch.Tensor,
                device: torch.device, batch_size: int = 32) -> tuple[np.ndarray, float]:
    chunks: list[np.ndarray] = []
    squared_error = 0.0
    elements = 0
    for start in range(0, len(images), batch_size):
        image = images[start:start + batch_size].to(device)
        condition = conditions[start:start + batch_size].to(device)
        mu, _ = model.encode(image, condition)
        prediction = render_field(model, condition, mu, model.image_size)
        squared_error += float((prediction - image).square().sum())
        elements += image.numel()
        chunks.append(prediction.cpu().numpy().astype(np.float16))
    return np.concatenate(chunks), squared_error / elements


def materialize() -> dict[str, Any]:
    _, protocol_hash = verify_protocol()
    inputs = verify_inputs()
    validation_rows = read_csv(ROOT / inputs["outputs"]["validation"]["path"])
    images, conditions = image_tensor(validation_rows), condition_tensor(validation_rows)
    sample_ids = np.asarray([row["sample_id"] for row in validation_rows])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" or torch.cuda.get_device_name(device) != "NVIDIA GeForce RTX 5080":
        raise RuntimeError("Frozen local RTX 5080 is required")
    artifacts: list[dict[str, Any]] = []
    for arm in ARMS:
        for seed in SEEDS:
            directory = RUN_ROOT / f"{arm}_s{seed}"
            summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
            checkpoint_path = directory / "best.pt"
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            model = build_model(checkpoint, device)
            predictions, measured_mse = reconstruct(model, images, conditions, device)
            expected_mse = float(summary["best_validation_mse"])
            if abs(measured_mse - expected_mse) > max(1e-9, expected_mse * 1e-5):
                raise RuntimeError(f"Validation MSE drift for {arm}_s{seed}: {measured_mse} vs {expected_mse}")
            output = OUTPUT_ROOT / f"{arm}_s{seed}.npz"
            atomic_npz(output, sample_id=sample_ids, reconstruction=predictions,
                       measured_validation_mse=np.asarray(measured_mse, dtype=np.float64))
            artifacts.append({
                "arm": arm, "seed": seed,
                "path": str(output.relative_to(ROOT)).replace("\\", "/"),
                "sha256": sha256_file(output), "bytes": output.stat().st_size,
                "shape": list(predictions.shape), "dtype": str(predictions.dtype),
                "measured_validation_mse": measured_mse,
                "checkpoint_sha256": sha256_file(checkpoint_path),
            })
            del model, checkpoint
            torch.cuda.empty_cache()
    receipt = {
        "schema_version": 1, "status": "PASS", "protocol_sha256": protocol_hash,
        "data_role": "validation", "arrays": len(artifacts), "rows_per_array": len(validation_rows),
        "test_payload_accessed": False, "artifacts": artifacts,
    }
    atomic_json(OUTPUT_ROOT / "MANIFEST.json", receipt)
    return receipt


def main() -> int:
    argparse.ArgumentParser(description=__doc__).parse_args()
    print(json.dumps(materialize(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
