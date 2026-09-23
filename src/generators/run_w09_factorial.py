"""Frozen W09 train/validation-only generator-factorial campaign.

The module refuses test access, verifies the W08 protocol and every frozen
input hash, writes atomic resumable run artifacts, and preserves failed runs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import tempfile
import time
from typing import Any

import numpy as np
from PIL import Image
import torch
import yaml


ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = ROOT / "historical" / "metal_spatter_pinn"
PACKAGE_ROOT = SOURCE_ROOT / "src"
if str(PACKAGE_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(PACKAGE_ROOT))

from metal_spatter_pinn.data import condition_from_manifest, make_coordinate_grid  # noqa: E402
from metal_spatter_pinn.inference import render_field  # noqa: E402
from metal_spatter_pinn.model import (  # noqa: E402
    ConditionalVariationalField,
    ModelConfig,
    kl_divergence,
)
from metal_spatter_pinn.training import weighted_reconstruction  # noqa: E402


LOCK_PATH = ROOT / "configs" / "PROTOCOL_LOCK.yaml"
LOCK_RECEIPT_PATH = ROOT / "configs" / "PROTOCOL_LOCK.sha256"
SPEC_PATH = ROOT / "configs" / "generator_factorial.json"
FULL_MANIFEST_PATH = ROOT / "data" / "manifests" / "image_manifest.csv"
INPUT_DIR = ROOT / "data" / "manifests" / "w09"
INPUT_RECEIPT_PATH = INPUT_DIR / "input_receipt.json"
RUN_ROOT = ROOT / "runs" / "new_study" / "W09_generator_factorial"
EVIDENCE_ROOT = ROOT / "evidence" / "generator_factorial"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def object_hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_json(path: Path, payload: Any) -> None:
    atomic_bytes(path, (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8"))


def atomic_torch(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".pt", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def atomic_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def verify_protocol() -> tuple[dict[str, Any], str]:
    expected, filename = LOCK_RECEIPT_PATH.read_text(encoding="utf-8").strip().split(maxsplit=1)
    if filename != LOCK_PATH.name or sha256_file(LOCK_PATH) != expected:
        raise RuntimeError("Protocol lock receipt mismatch")
    lock = yaml.safe_load(LOCK_PATH.read_text(encoding="utf-8"))
    if lock["status"] != "FROZEN_PRE_N2_OUTCOME":
        raise RuntimeError("Protocol is not frozen")
    gates = lock["execution_gates"]
    if gates["full_training_authorized"] is not True or gates["test_data_access_authorized_now"] is not False:
        raise RuntimeError("Training authorization or test gate differs")
    for name, item in lock["frozen_inputs"].items():
        path = ROOT / item["path"]
        if not path.is_file() or sha256_file(path) != item["sha256"]:
            raise RuntimeError(f"Frozen input mismatch: {name}")
    return lock, expected


def spec() -> dict[str, Any]:
    value = read_json(SPEC_PATH)
    if value["status"] != "W08_APPROVED_PRE_TEST" or value["test_access"] != "PROHIBITED":
        raise RuntimeError("Generator specification is not approved")
    return value


def prepare_inputs() -> dict[str, Any]:
    lock, protocol_hash = verify_protocol()
    rows = read_csv(FULL_MANIFEST_PATH)
    if len(rows) != 5632:
        raise RuntimeError(f"Expected 5632 frozen manifest rows, found {len(rows)}")
    outputs: dict[str, Any] = {}
    for split, expected in (("train", 3584), ("validation", 1024)):
        selected = sorted((row for row in rows if row["split"] == split), key=lambda row: row["sample_id"])
        if len(selected) != expected or any(row["split"] != split for row in selected):
            raise RuntimeError(f"Unexpected {split} roster")
        if len({row["specimen"] for row in selected}) != (112 if split == "train" else 32):
            raise RuntimeError(f"Unexpected {split} specimen count")
        path = INPUT_DIR / f"{split}.csv"
        atomic_csv(path, selected, list(rows[0]))
        outputs[split] = {
            "path": str(path.relative_to(ROOT)).replace("\\", "/"),
            "rows": len(selected),
            "specimens": len({row["specimen"] for row in selected}),
            "sha256": sha256_file(path),
        }
    if set(outputs) != {"train", "validation"}:
        raise RuntimeError("Only train and validation outputs are allowed")
    receipt = {
        "schema_version": 1,
        "status": "PASS",
        "protocol_sha256": protocol_hash,
        "source_manifest_sha256": lock["frozen_inputs"]["derived_image_manifest"]["sha256"],
        "outputs": outputs,
        "test_rows_written": 0,
        "test_image_or_label_payload_accessed": False,
    }
    atomic_json(INPUT_RECEIPT_PATH, receipt)
    return receipt


def verify_inputs() -> dict[str, Any]:
    receipt = read_json(INPUT_RECEIPT_PATH)
    _, protocol_hash = verify_protocol()
    if receipt["protocol_sha256"] != protocol_hash or receipt["test_rows_written"] != 0:
        raise RuntimeError("W09 input receipt differs from protocol")
    for split, expected in (("train", 3584), ("validation", 1024)):
        item = receipt["outputs"][split]
        path = ROOT / item["path"]
        if sha256_file(path) != item["sha256"] or len(read_csv(path)) != expected:
            raise RuntimeError(f"W09 {split} input mismatch")
    return receipt


def stream_seed(seed: int, phase: str, step: int = 0) -> int:
    payload = f"SR-LPBF-N2-LOCK-20260914-A001|{seed}|{phase}|{step}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63 - 1)


def generator(seed: int, phase: str, step: int, device: torch.device | str = "cpu") -> torch.Generator:
    return torch.Generator(device=device).manual_seed(stream_seed(seed, phase, step))


def seed_everything(seed: int) -> None:
    random.seed(stream_seed(seed, "python"))
    np.random.seed(stream_seed(seed, "numpy") % (2**32 - 1))
    torch.manual_seed(stream_seed(seed, "torch"))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(stream_seed(seed, "cuda"))


def resolve_image(row: dict[str, str]) -> Path:
    path = SOURCE_ROOT / Path(row["image_path"].replace("\\", "/"))
    if not path.is_file() or sha256_file(path) != row["image_sha256"]:
        raise RuntimeError(f"Image hash mismatch: {row['sample_id']}")
    return path


def image_tensor(rows: list[dict[str, str]]) -> torch.Tensor:
    images: list[np.ndarray] = []
    for row in rows:
        with Image.open(resolve_image(row)) as image:
            array = np.asarray(
                image.convert("L").resize((64, 64), Image.Resampling.BILINEAR), dtype=np.float32
            ) / 255.0
        images.append(array[None])
    return torch.from_numpy(np.stack(images))


def condition_tensor(rows: list[dict[str, str]]) -> torch.Tensor:
    return torch.from_numpy(np.stack([condition_from_manifest(row).vector for row in rows])).float()


def coefficients(condition: torch.Tensor) -> dict[str, torch.Tensor]:
    area = torch.expm1(condition[:, 9:10] * math.log(90001)) / 90000
    base = area.clamp_min(16 / 90000).sqrt().clamp(0.018, 0.55)
    return {
        "diffusivity": 0.018 + 0.010 * condition[:, 4:5] + 0.004 * condition[:, 5:6],
        "transport_x": (0.05 + 0.15 * condition[:, 1:2]) * condition[:, 12:13],
        "transport_y": (0.05 + 0.15 * condition[:, 1:2]) * condition[:, 11:12],
        "source": 0.08 + 0.42 * condition[:, 2:3] + 0.05 * condition[:, 0:1],
        "decay": 0.65 + 0.35 * condition[:, 1:2],
        "center": condition[:, 7:9],
        "sigma": torch.cat(
            (
                (base * torch.exp(0.25 * condition[:, 10:11])).clamp(0.018, 0.60),
                (base * torch.exp(-0.25 * condition[:, 10:11])).clamp(0.018, 0.60),
            ),
            1,
        ),
    }


def regularizer_components(
    decoder: torch.nn.Module,
    condition: torch.Tensor,
    latent: torch.Tensor,
    coordinates: torch.Tensor,
    boundary: torch.Tensor,
    moment_grid: torch.Tensor,
) -> dict[str, torch.Tensor]:
    coordinates = coordinates.detach().clone().requires_grad_(True)
    field = decoder(coordinates, condition, latent)
    grad = torch.autograd.grad(field, coordinates, torch.ones_like(field), create_graph=True)[0]
    hxx = torch.autograd.grad(
        grad[..., :1], coordinates, torch.ones_like(grad[..., :1]), create_graph=True, retain_graph=True
    )[0][..., :1]
    hyy = torch.autograd.grad(
        grad[..., 1:], coordinates, torch.ones_like(grad[..., 1:]), create_graph=True, retain_graph=True
    )[0][..., 1:]
    laplacian = hxx + hyy
    values = coefficients(condition)
    gaussian = torch.exp(
        -0.5 * ((coordinates - values["center"][:, None]) / values["sigma"][:, None]).square().sum(-1, keepdim=True)
    )
    residual = (
        -values["diffusivity"][:, None] * laplacian
        + values["transport_x"][:, None] * grad[..., :1]
        + values["transport_y"][:, None] * grad[..., 1:]
        + values["decay"][:, None] * field
        - values["source"][:, None] * gaussian
    )
    magnitude = residual.detach().abs()
    attention = 0.2 + 0.8 * magnitude / magnitude.amax(1, keepdim=True).clamp_min(1e-6)
    boundary_field = decoder(boundary, condition, latent)
    moment_field = decoder(moment_grid, condition, latent).squeeze(-1)
    weights = moment_field.square() + 1e-6
    norm = weights.sum(1, keepdim=True)
    center = (weights[..., None] * moment_grid).sum(1) / norm
    variance = (weights[..., None] * (moment_grid - center[:, None]).square()).sum(1) / norm
    spread = variance.clamp_min(1e-6).sqrt() - values["sigma"]
    moment = (center - values["center"]).square().mean() + 0.25 * spread.square().mean()
    return {
        "proxy": (attention * residual.square()).mean(),
        "boundary": boundary_field.square().mean(),
        "moment": moment,
    }


def draw_inputs(
    seed: int,
    step: int,
    batch: int,
    latent_dim: int,
    pixel_samples: int,
    collocation_samples: int,
    boundary_samples: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    noise = torch.randn(batch, latent_dim, generator=generator(seed, "posterior", step, device), device=device)
    pixels = torch.randint(4096, (batch, pixel_samples), generator=generator(seed, "pixels", step, device), device=device)
    collocation = torch.rand(
        batch, collocation_samples, 2, generator=generator(seed, "collocation", step, device), device=device
    ) * 2 - 1
    boundary_rng = generator(seed, "boundary", step, device)
    t = torch.rand(batch, boundary_samples, 1, generator=boundary_rng, device=device) * 2 - 1
    edge = torch.randint(4, (batch, boundary_samples, 1), generator=boundary_rng, device=device)
    bx = torch.where(edge == 0, -torch.ones_like(t), torch.where(edge == 1, torch.ones_like(t), t))
    by = torch.where(edge == 2, -torch.ones_like(t), torch.where(edge == 3, torch.ones_like(t), t))
    return noise, pixels, collocation, torch.cat((bx, by), -1)


@torch.no_grad()
def validation_mse(
    model: ConditionalVariationalField,
    images: torch.Tensor,
    conditions: torch.Tensor,
    device: torch.device,
    batch_size: int = 32,
) -> float:
    model.eval()
    total = 0.0
    pixels = 0
    for start in range(0, len(images), batch_size):
        image = images[start : start + batch_size].to(device)
        condition = conditions[start : start + batch_size].to(device)
        mu, _ = model.encode(image, condition)
        prediction = render_field(model, condition, mu, 64)
        total += float((prediction - image).square().sum())
        pixels += image.numel()
    return total / pixels


def rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        torch.cuda.set_rng_state_all([item.cpu() for item in state["cuda"]])


def run_identity(arm: str, seed: int, protocol_hash: str, inputs: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "run_id": f"W09_{arm}_s{seed}",
        "attempt_id": 1,
        "evidence_layer": "N2_NEW_STUDY",
        "arm": arm,
        "generator_seed": seed,
        "protocol_sha256": protocol_hash,
        "config_sha256": sha256_file(SPEC_PATH),
        "dataset_sha256": object_hash({key: value["sha256"] for key, value in inputs["outputs"].items()}),
        "label_policy_sha256": sha256_file(ROOT / "configs" / "LABEL_POLICY_LOCK.yaml"),
        "environment_sha256": sha256_file(ROOT / "ENVIRONMENT_LOCK.json"),
        "source_sha256": sha256_file(Path(__file__)),
        "data_roles": ["train", "validation"],
        "test_access": "PROHIBITED",
    }


def run_one(arm: str, seed: int) -> dict[str, Any]:
    lock, protocol_hash = verify_protocol()
    configuration = spec()
    if arm not in configuration["arms"] or seed not in configuration["seeds"]:
        raise ValueError("Run is outside the frozen W09 roster")
    inputs = verify_inputs()
    identity = run_identity(arm, seed, protocol_hash, inputs)
    directory = RUN_ROOT / f"{arm}_s{seed}"
    summary_path = directory / "summary.json"
    if summary_path.exists():
        summary = read_json(summary_path)
        if summary["identity_sha256"] != object_hash(identity):
            raise RuntimeError("Completed run identity mismatch")
        for relative, expected in summary["artifact_hashes"].items():
            if sha256_file(directory / relative) != expected:
                raise RuntimeError(f"Completed artifact mismatch: {relative}")
        return summary

    directory.mkdir(parents=True, exist_ok=True)
    atomic_json(directory / "identity.json", identity)
    resolved = {
        "protocol_id": lock["protocol_id"],
        "arm": arm,
        "seed": seed,
        "weights": configuration["arms"][arm],
        "model": configuration["model"],
        "training": configuration["training"],
        "checkpoint": configuration["checkpoint"],
    }
    atomic_json(directory / "resolved_config.json", resolved)

    train_rows = read_csv(ROOT / inputs["outputs"]["train"]["path"])
    validation_rows = read_csv(ROOT / inputs["outputs"]["validation"]["path"])
    images, conditions = image_tensor(train_rows), condition_tensor(train_rows)
    validation_images = image_tensor(validation_rows)
    validation_conditions = condition_tensor(validation_rows)
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" or torch.cuda.get_device_name(device) != "NVIDIA GeForce RTX 5080":
        raise RuntimeError("Frozen local RTX 5080 is required")

    model_config = ModelConfig(
        condition_dim=configuration["condition_dimension"],
        latent_dim=configuration["model"]["latent_dimension"],
        hidden_dim=configuration["model"]["hidden_dimension"],
        fourier_frequencies=tuple(configuration["model"]["fourier_frequencies"]),
        residual_blocks=configuration["model"]["residual_blocks"],
        decoder_type=configuration["model"]["decoder_type"],
        conditional_prior=configuration["model"]["conditional_prior"],
    )
    model = ConditionalVariationalField(model_config, configuration["image_size"]).to(device)
    training = configuration["training"]
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=training["learning_rate"], weight_decay=training["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=training["epochs"], eta_min=training["scheduler_eta_min"]
    )
    grid = make_coordinate_grid(64, device)
    moment_grid = make_coordinate_grid(training["moment_grid_size"], device)
    batch_size = training["batch_size"]
    steps_per_epoch = len(images) // batch_size
    resume_path = directory / "resume.pt"
    history: list[dict[str, Any]] = []
    epoch = 0
    best_epoch = -1
    best_mse = float("inf")
    elapsed_prior = 0.0
    if resume_path.exists():
        saved = torch.load(resume_path, map_location=device, weights_only=False)
        if saved["identity_sha256"] != object_hash(identity):
            raise RuntimeError("Resume identity mismatch")
        model.load_state_dict(saved["model_state"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        history = saved["history"]
        epoch = saved["epoch"]
        best_epoch = saved["best_epoch"]
        best_mse = saved["best_mse"]
        elapsed_prior = saved["elapsed_seconds"]
        restore_rng(saved["rng"])

    started = time.perf_counter()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    def save_resume() -> None:
        atomic_torch(
            resume_path,
            {
                **model.checkpoint_payload(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "rng": rng_state(),
                "epoch": epoch,
                "history": history,
                "best_epoch": best_epoch,
                "best_mse": best_mse,
                "elapsed_seconds": elapsed_prior + time.perf_counter() - started,
                "identity_sha256": object_hash(identity),
            },
        )

    if not resume_path.exists():
        save_resume()
    failure_path = directory / "failure.json"
    try:
        while epoch < training["epochs"]:
            model.train()
            order = torch.randperm(len(images), generator=generator(seed, "image_order", epoch))
            totals = {key: 0.0 for key in ("loss", "reconstruction", "kl", "proxy", "boundary", "moment")}
            for batch_index in range(steps_per_epoch):
                indices = order[batch_index * batch_size : (batch_index + 1) * batch_size]
                image = images[indices].to(device)
                condition = conditions[indices].to(device)
                step = epoch * steps_per_epoch + batch_index
                noise, pixels, collocation, boundary = draw_inputs(
                    seed,
                    step,
                    batch_size,
                    model_config.latent_dim,
                    training["pixel_samples"],
                    training["collocation_samples"],
                    training["boundary_samples"],
                    device,
                )
                mu, logvar = model.encode(image, condition)
                latent = mu + torch.exp(0.5 * logvar) * noise
                coordinates = grid[pixels]
                prediction = model.decode(coordinates, condition, latent)
                targets = torch.gather(image[:, 0].reshape(batch_size, -1, 1), 1, pixels[..., None])
                reconstruction = weighted_reconstruction(prediction, targets)
                kl = kl_divergence(mu, logvar)
                regularizers = regularizer_components(
                    model.decoder,
                    condition,
                    latent,
                    collocation,
                    boundary,
                    moment_grid[None].expand(batch_size, -1, -1),
                )
                warmup = min(1.0, (epoch + 1) / training["physics_warmup_epochs"])
                weights = configuration["arms"][arm]
                loss = reconstruction + training["kl_weight"] * kl + warmup * sum(
                    weights[name] * regularizers[name] for name in weights
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Nonfinite loss at epoch {epoch + 1}, step {step}")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), training["gradient_clip_norm"], error_if_nonfinite=True
                )
                optimizer.step()
                values = {"loss": loss, "reconstruction": reconstruction, "kl": kl, **regularizers}
                for key, value in values.items():
                    totals[key] += float(value.detach())
            scheduler.step()
            value = validation_mse(model, validation_images, validation_conditions, device)
            record = {
                "epoch": epoch + 1,
                **{key: total / steps_per_epoch for key, total in totals.items()},
                "validation_mse": value,
            }
            history.append(record)
            if value < best_mse:
                best_mse = value
                best_epoch = epoch + 1
                atomic_torch(
                    directory / "best.pt",
                    {
                        **model.checkpoint_payload(),
                        "epoch": best_epoch,
                        "validation_mse": best_mse,
                        "arm": arm,
                        "seed": seed,
                        "identity_sha256": object_hash(identity),
                    },
                )
            epoch += 1
            save_resume()
            print(json.dumps({"arm": arm, "seed": seed, **record}), flush=True)
        torch.cuda.synchronize()
        atomic_csv(directory / "history.csv", history, list(history[0]))
        elapsed = elapsed_prior + time.perf_counter() - started
        artifacts = ["identity.json", "resolved_config.json", "history.csv", "best.pt", "resume.pt"]
        summary = {
            "schema_version": 1,
            "status": "COMPLETED",
            "arm": arm,
            "seed": seed,
            "epochs": epoch,
            "optimizer_updates": epoch * steps_per_epoch,
            "best_epoch": best_epoch,
            "best_validation_mse": best_mse,
            "elapsed_seconds": elapsed,
            "gpu_hours": elapsed / 3600,
            "peak_allocated_cuda_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_cuda_bytes": torch.cuda.max_memory_reserved(),
            "identity_sha256": object_hash(identity),
            "test_payload_accessed": False,
            "artifact_hashes": {name: sha256_file(directory / name) for name in artifacts},
        }
        atomic_json(summary_path, summary)
        failure_path.unlink(missing_ok=True)
        return summary
    except BaseException as exc:
        atomic_json(
            failure_path,
            {
                "status": "FAILED_OR_INTERRUPTED",
                "arm": arm,
                "seed": seed,
                "epoch_last_atomically_completed": epoch,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "resume_sha256": sha256_file(resume_path) if resume_path.exists() else None,
                "test_payload_accessed": False,
            },
        )
        raise
    finally:
        del model, optimizer
        torch.cuda.empty_cache()


def roster() -> list[tuple[str, int]]:
    configuration = spec()
    return [(arm, seed) for arm in configuration["arms"] for seed in configuration["seeds"]]


def summarize() -> dict[str, Any]:
    _, protocol_hash = verify_protocol()
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    missing: list[str] = []
    for arm, seed in roster():
        directory = RUN_ROOT / f"{arm}_s{seed}"
        if (directory / "summary.json").exists():
            rows.append(read_json(directory / "summary.json"))
        elif (directory / "failure.json").exists():
            failures.append(read_json(directory / "failure.json"))
        else:
            missing.append(f"{arm}_s{seed}")
    EVIDENCE_ROOT.mkdir(parents=True, exist_ok=True)
    metrics = [
        {
            "arm": row["arm"],
            "seed": row["seed"],
            "best_epoch": row["best_epoch"],
            "best_validation_mse": row["best_validation_mse"],
            "optimizer_updates": row["optimizer_updates"],
            "elapsed_seconds": row["elapsed_seconds"],
            "gpu_hours": row["gpu_hours"],
            "summary_sha256": sha256_file(RUN_ROOT / f"{row['arm']}_s{row['seed']}" / "summary.json"),
        }
        for row in rows
    ]
    if metrics:
        atomic_csv(EVIDENCE_ROOT / "factorial_seed_metrics.csv", metrics, list(metrics[0]))
    receipt = {
        "schema_version": 1,
        "status": "COMPLETE" if len(rows) == 40 and not failures and not missing else "INCOMPLETE",
        "protocol_sha256": protocol_hash,
        "planned": 40,
        "completed": len(rows),
        "failed_or_interrupted": len(failures),
        "missing": len(missing),
        "failure_records": failures,
        "missing_run_ids": missing,
        "cumulative_generator_gpu_hours": sum(row["gpu_hours"] for row in rows),
        "test_payload_accessed": False,
    }
    atomic_json(EVIDENCE_ROOT / "W09_COMPLETENESS.json", receipt)
    return receipt


def preflight() -> dict[str, Any]:
    lock, protocol_hash = verify_protocol()
    inputs = verify_inputs()
    configuration = spec()
    checks = {
        "protocol_receipt_valid": True,
        "frozen_inputs_valid": True,
        "test_access_disabled": lock["execution_gates"]["test_data_access_authorized_now"] is False,
        "train_rows": inputs["outputs"]["train"]["rows"] == 3584,
        "validation_rows": inputs["outputs"]["validation"]["rows"] == 1024,
        "test_rows_written_zero": inputs["test_rows_written"] == 0,
        "factorial_cells": len(configuration["arms"]) == 4,
        "paired_seeds": len(configuration["seeds"]) == 10,
        "planned_runs": len(roster()) == 40,
        "cuda_available": torch.cuda.is_available(),
        "approved_device": torch.cuda.is_available() and torch.cuda.get_device_name(0) == "NVIDIA GeForce RTX 5080",
    }
    receipt = {
        "schema_version": 1,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "protocol_sha256": protocol_hash,
        "checks": checks,
        "planned_run_count": len(roster()),
        "test_payload_accessed": False,
    }
    EVIDENCE_ROOT.mkdir(parents=True, exist_ok=True)
    atomic_json(EVIDENCE_ROOT / "W09_PREFLIGHT.json", receipt)
    if receipt["status"] != "PASS":
        raise RuntimeError("W09 preflight failed")
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--prepare", action="store_true")
    action.add_argument("--preflight", action="store_true")
    action.add_argument("--run-one", action="store_true")
    action.add_argument("--run-all", action="store_true")
    action.add_argument("--summarize", action="store_true")
    parser.add_argument("--arm", choices=["F00", "F01", "F10", "F11"])
    parser.add_argument("--seed", type=int)
    args = parser.parse_args(argv)
    if args.prepare:
        result = prepare_inputs()
    elif args.preflight:
        result = preflight()
    elif args.run_one:
        if args.arm is None or args.seed is None:
            parser.error("--run-one requires --arm and --seed")
        result = run_one(args.arm, args.seed)
    elif args.run_all:
        preflight()
        for arm, seed in roster():
            current = summarize()
            if current["cumulative_generator_gpu_hours"] >= 130:
                raise RuntimeError("Approved cumulative GPU-hour ceiling reached")
            run_one(arm, seed)
        result = summarize()
    else:
        result = summarize()
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

