"""Frozen W10 train/validation-only optical-prior strength sweep.

The runner derives its grid and seeds from the W08 protocol, reuses the W09
training contract, refuses test access, and writes resumable per-cell evidence.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

import torch

from run_w09_factorial import (
    ROOT,
    ConditionalVariationalField,
    ModelConfig,
    atomic_json,
    atomic_torch,
    condition_tensor,
    draw_inputs,
    generator,
    image_tensor,
    kl_divergence,
    make_coordinate_grid,
    object_hash,
    read_csv,
    regularizer_components,
    restore_rng,
    rng_state,
    seed_everything,
    sha256_file,
    spec as w09_spec,
    validation_mse,
    verify_inputs,
    verify_protocol,
    weighted_reconstruction,
)


CONFIG_PATH = ROOT / "configs" / "w10_prior_sweep.json"
RUN_ROOT = ROOT / "runs" / "new_study" / "W10_prior_sweep"
EVIDENCE_ROOT = ROOT / "evidence" / "prior_sweep"


def config() -> dict[str, Any]:
    value = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    lock, _ = verify_protocol()
    frozen = lock["generator"]
    if value["status"] != "DERIVED_FROM_FROZEN_PROTOCOL":
        raise RuntimeError("W10 configuration is not protocol-derived")
    if value["test_access"] != "PROHIBITED" or value["data_roles"] != ["train", "validation"]:
        raise RuntimeError("W10 data roles differ from the frozen pretest contract")
    if value["proxy_lambda_grid"] != frozen["proxy_lambda_sensitivity_grid"]:
        raise RuntimeError("W10 lambda grid differs from the protocol")
    if value["seeds"] != lock["seed_rosters"]["lambda_development"]:
        raise RuntimeError("W10 seed roster differs from the protocol")
    controls = value["smoothing_controls"]
    expected = frozen["non_pde_controls"]
    if controls["gaussian_sigma_px"] != expected["blob_gaussian_sigma_px"]:
        raise RuntimeError("Gaussian control grid differs from the protocol")
    if controls["laplacian_heat_tau_px2"] != expected["laplacian_heat_tau_px2"]:
        raise RuntimeError("Heat control grid differs from the protocol")
    return value


def lambda_slug(value: float) -> str:
    return f"L{value:.5f}".replace(".", "p")


def roster() -> list[tuple[float, int]]:
    value = config()
    return [(float(weight), int(seed)) for weight in value["proxy_lambda_grid"] for seed in value["seeds"]]


def run_identity(weight: float, seed: int, protocol_hash: str, inputs: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "run_id": f"W10_{lambda_slug(weight)}_s{seed}",
        "evidence_layer": "N2_NEW_STUDY",
        "proxy_lambda": weight,
        "seed": seed,
        "protocol_sha256": protocol_hash,
        "config_sha256": sha256_file(CONFIG_PATH),
        "training_config_sha256": sha256_file(ROOT / "configs" / "generator_factorial.json"),
        "w09_training_dependency_sha256": sha256_file(ROOT / "src" / "generators" / "run_w09_factorial.py"),
        "dataset_sha256": object_hash({key: item["sha256"] for key, item in inputs["outputs"].items()}),
        "source_sha256": sha256_file(Path(__file__)),
        "data_roles": ["train", "validation"],
        "test_access": "PROHIBITED",
    }


def run_one(weight: float, seed: int) -> dict[str, Any]:
    value = config()
    if (weight, seed) not in roster():
        raise ValueError("Run is outside the frozen W10 roster")
    _, protocol_hash = verify_protocol()
    inputs = verify_inputs()
    identity = run_identity(weight, seed, protocol_hash, inputs)
    directory = RUN_ROOT / f"{lambda_slug(weight)}_s{seed}"
    summary_path = directory / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary["identity_sha256"] != object_hash(identity):
            raise RuntimeError("Completed W10 run identity mismatch")
        return summary

    directory.mkdir(parents=True, exist_ok=True)
    atomic_json(directory / "identity.json", identity)
    base = w09_spec()
    training = base["training"]
    resolved = {
        "proxy_lambda": weight,
        "boundary_weight": value["boundary_weight"],
        "moment_weight": value["moment_weight"],
        "model": base["model"],
        "training": training,
        "checkpoint_selection": value["checkpoint_selection"],
    }
    atomic_json(directory / "resolved_config.json", resolved)

    train_rows = read_csv(ROOT / inputs["outputs"]["train"]["path"])
    validation_rows = read_csv(ROOT / inputs["outputs"]["validation"]["path"])
    images, conditions = image_tensor(train_rows), condition_tensor(train_rows)
    validation_images, validation_conditions = image_tensor(validation_rows), condition_tensor(validation_rows)
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" or torch.cuda.get_device_name(device) != "NVIDIA GeForce RTX 5080":
        raise RuntimeError("Frozen local RTX 5080 is required")

    model_cfg = base["model"]
    model = ConditionalVariationalField(
        ModelConfig(
            condition_dim=base["condition_dimension"],
            latent_dim=model_cfg["latent_dimension"],
            hidden_dim=model_cfg["hidden_dimension"],
            fourier_frequencies=tuple(model_cfg["fourier_frequencies"]),
            residual_blocks=model_cfg["residual_blocks"],
            decoder_type=model_cfg["decoder_type"],
            conditional_prior=model_cfg["conditional_prior"],
        ),
        base["image_size"],
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=training["learning_rate"], weight_decay=training["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=training["epochs"], eta_min=training["scheduler_eta_min"]
    )
    grid = make_coordinate_grid(base["image_size"], device)
    moment_grid = make_coordinate_grid(training["moment_grid_size"], device)
    batch_size = training["batch_size"]
    steps_per_epoch = len(images) // batch_size
    resume_path = directory / "resume.pt"
    history: list[dict[str, Any]] = []
    epoch, best_epoch, best_mse, elapsed_prior = 0, -1, float("inf"), 0.0
    identity_hash = object_hash(identity)
    if resume_path.exists():
        saved = torch.load(resume_path, map_location=device, weights_only=False)
        if saved["identity_sha256"] != identity_hash:
            raise RuntimeError("Resume identity mismatch")
        model.load_state_dict(saved["model_state"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        history, epoch = saved["history"], saved["epoch"]
        best_epoch, best_mse = saved["best_epoch"], saved["best_mse"]
        elapsed_prior = saved["elapsed_seconds"]
        restore_rng(saved["rng"])

    started = time.perf_counter()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    def save_resume() -> None:
        atomic_torch(directory / "resume.pt", {
            **model.checkpoint_payload(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "rng": rng_state(), "epoch": epoch,
            "history": history, "best_epoch": best_epoch, "best_mse": best_mse,
            "elapsed_seconds": elapsed_prior + time.perf_counter() - started,
            "identity_sha256": identity_hash,
        })

    if not resume_path.exists():
        save_resume()
    failure_path = directory / "failure.json"
    try:
        while epoch < training["epochs"]:
            model.train()
            order = torch.randperm(len(images), generator=generator(seed, "image_order", epoch))
            totals = {key: 0.0 for key in ("loss", "reconstruction", "kl", "proxy", "boundary", "moment")}
            for batch_index in range(steps_per_epoch):
                indices = order[batch_index * batch_size:(batch_index + 1) * batch_size]
                image, condition = images[indices].to(device), conditions[indices].to(device)
                step = epoch * steps_per_epoch + batch_index
                noise, pixels, collocation, boundary = draw_inputs(
                    seed, step, batch_size, model_cfg["latent_dimension"], training["pixel_samples"],
                    training["collocation_samples"], training["boundary_samples"], device,
                )
                mu, logvar = model.encode(image, condition)
                latent = mu + (0.5 * logvar).exp() * noise
                coordinates = grid[pixels]
                target = image.flatten(2).transpose(1, 2).gather(1, pixels[..., None])
                prediction = model.decode(coordinates, condition, latent)
                reconstruction = weighted_reconstruction(prediction, target)
                kl = kl_divergence(mu, logvar)
                regularizers = regularizer_components(
                    model.decoder, condition, latent, collocation, boundary,
                    moment_grid[None].expand(batch_size, -1, -1),
                )
                warmup = min(1.0, (epoch + 1) / training["physics_warmup_epochs"])
                weights = {"proxy": weight, "boundary": value["boundary_weight"], "moment": value["moment_weight"]}
                loss = reconstruction + training["kl_weight"] * kl + warmup * sum(weights[k] * regularizers[k] for k in weights)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Nonfinite loss at epoch {epoch + 1}, step {step}")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), training["gradient_clip_norm"], error_if_nonfinite=True)
                optimizer.step()
                for key, metric in {"loss": loss, "reconstruction": reconstruction, "kl": kl, **regularizers}.items():
                    totals[key] += float(metric.detach())
            scheduler.step()
            measured = validation_mse(model, validation_images, validation_conditions, device)
            record = {"epoch": epoch + 1, **{key: total / steps_per_epoch for key, total in totals.items()}, "validation_mse": measured}
            history.append(record)
            if measured < best_mse:
                best_mse, best_epoch = measured, epoch + 1
                atomic_torch(directory / "best.pt", {**model.checkpoint_payload(), "epoch": best_epoch,
                    "validation_mse": best_mse, "proxy_lambda": weight, "seed": seed,
                    "identity_sha256": identity_hash})
            epoch += 1
            save_resume()
            print(json.dumps({"proxy_lambda": weight, "seed": seed, **record}), flush=True)
        torch.cuda.synchronize()
        from run_w09_factorial import atomic_csv
        atomic_csv(directory / "history.csv", history, list(history[0]))
        elapsed = elapsed_prior + time.perf_counter() - started
        artifacts = ["identity.json", "resolved_config.json", "history.csv", "best.pt", "resume.pt"]
        summary = {
            "schema_version": 1, "status": "COMPLETED", "proxy_lambda": weight,
            "seed": seed, "epochs": epoch, "optimizer_updates": epoch * steps_per_epoch,
            "best_epoch": best_epoch, "best_validation_mse": best_mse,
            "elapsed_seconds": elapsed, "gpu_hours": elapsed / 3600,
            "peak_allocated_cuda_bytes": torch.cuda.max_memory_allocated(),
            "identity_sha256": identity_hash, "test_payload_accessed": False,
            "artifact_hashes": {name: sha256_file(directory / name) for name in artifacts},
        }
        atomic_json(summary_path, summary)
        failure_path.unlink(missing_ok=True)
        return summary
    except BaseException as exc:
        atomic_json(failure_path, {"status": "FAILED_OR_INTERRUPTED", "proxy_lambda": weight,
            "seed": seed, "epoch_last_atomically_completed": epoch,
            "error_type": type(exc).__name__, "error": str(exc), "test_payload_accessed": False})
        raise
    finally:
        del model, optimizer
        torch.cuda.empty_cache()


def summarize() -> dict[str, Any]:
    _, protocol_hash = verify_protocol()
    completed, failures, missing = [], [], []
    for weight, seed in roster():
        directory = RUN_ROOT / f"{lambda_slug(weight)}_s{seed}"
        if (directory / "summary.json").exists():
            completed.append(json.loads((directory / "summary.json").read_text(encoding="utf-8")))
        elif (directory / "failure.json").exists():
            failures.append(json.loads((directory / "failure.json").read_text(encoding="utf-8")))
        else:
            missing.append(f"{lambda_slug(weight)}_s{seed}")
    receipt = {
        "schema_version": 1,
        "status": "COMPLETE" if len(completed) == 18 and not failures and not missing else "INCOMPLETE",
        "protocol_sha256": protocol_hash, "planned": 18, "completed": len(completed),
        "failed_or_interrupted": len(failures), "missing": len(missing),
        "missing_run_ids": missing, "failure_records": failures,
        "cumulative_gpu_hours": sum(item["gpu_hours"] for item in completed),
        "test_payload_accessed": False,
    }
    atomic_json(EVIDENCE_ROOT / "W10_SWEEP_COMPLETENESS.json", receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--preflight", action="store_true")
    action.add_argument("--run-all", action="store_true")
    action.add_argument("--run-one", action="store_true")
    action.add_argument("--summarize", action="store_true")
    parser.add_argument("--lambda-weight", type=float)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    if args.preflight:
        result = {"status": "PASS", "planned": len(roster()), "test_access": "PROHIBITED"}
    elif args.run_one:
        if args.lambda_weight is None or args.seed is None:
            parser.error("--run-one requires --lambda-weight and --seed")
        result = run_one(args.lambda_weight, args.seed)
    elif args.run_all:
        for weight, seed in roster():
            if summarize()["cumulative_gpu_hours"] >= 130:
                raise RuntimeError("Approved cumulative GPU-hour ceiling reached")
            run_one(weight, seed)
        result = summarize()
    else:
        result = summarize()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
