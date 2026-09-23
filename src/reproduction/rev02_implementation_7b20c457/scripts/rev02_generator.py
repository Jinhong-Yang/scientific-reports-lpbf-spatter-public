"""Frozen REV02 generator campaign. Explicit split access; no historical runner calls.

All production paths/provenance/access gates are supplied by rev02_common. Pure
numerical functions below also support synthetic-unit tests without corpus access.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import importlib
import itertools
import json
import math
from pathlib import Path
import random
import sys
import time
from typing import Any

import numpy as np
from PIL import Image
from scipy import ndimage
from skimage.metrics import structural_similarity
import torch
import yaml
from torch.profiler import ProfilerActivity, profile

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))
from metal_spatter_pinn.data import condition_from_manifest, make_coordinate_grid
from metal_spatter_pinn.evaluation import morphology_features
from metal_spatter_pinn.inference import load_model, render_field
from metal_spatter_pinn.model import ConditionalVariationalField, ModelConfig, kl_divergence
from metal_spatter_pinn.training import weighted_reconstruction

PRIMARY = ("fresh_collocation_proxy_RMS", "reconstruction_PSNR", "PFFD_std",
           "within_condition_one_minus_SSIM", "boundary_energy", "moment_centroid_error",
           "moment_spread_error")
BASE_ARMS = ("G0", "F00", "F01", "F10", "F11", "G2N")
SENSITIVITY = {
    "F11_kappa_low": ("diffusivity_intercept", .009),
    "F11_kappa_high": ("diffusivity_intercept", .027),
    "F11_source_low": ("source_line_energy_coefficient", .21),
    "F11_source_high": ("source_line_energy_coefficient", .63),
    "F11_decay_low": ("decay_intercept", .325),
    "F11_decay_high": ("decay_intercept", .975),
}
COEFFICIENT_DEFAULTS = {"diffusivity_intercept": .018,
                         "source_line_energy_coefficient": .42, "decay_intercept": .65}
FEATURES = tuple(morphology_features(np.zeros((8, 8))).keys())


def common():
    return importlib.import_module("rev02_common")


def canonical_arm(arm: str) -> str:
    arm = {"G1": "F00", "G2": "F11"}.get(arm, arm)
    if arm not in BASE_ARMS and arm not in SENSITIVITY:
        raise ValueError(f"Unplanned generator arm: {arm}")
    return arm


def object_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def stream_seed(seed: int, phase: str, step: int = 0) -> int:
    return int.from_bytes(hashlib.sha256(f"REV02|{seed}|{phase}|{step}".encode()).digest()[:8], "big") % (2**63-1)


def rng(seed: int, phase: str, step: int, device="cpu") -> torch.Generator:
    return torch.Generator(device=device).manual_seed(stream_seed(seed, phase, step))


def fixed_boundary(points_per_edge=8, device="cpu") -> torch.Tensor:
    t = torch.linspace(-1, 1, points_per_edge, device=device)
    return torch.cat([torch.stack((t*0-1, t), 1), torch.stack((t*0+1, t), 1),
                      torch.stack((t, t*0-1), 1), torch.stack((t, t*0+1), 1)])


def draw_training_inputs(seed, step, batch, pixels, pixel_samples, latent_dim,
                         collocation_count, boundary_count, device):
    posterior = torch.randn(batch, latent_dim, generator=rng(seed, "posterior", step, device), device=device)
    indices = torch.randint(pixels, (batch, pixel_samples), generator=rng(seed, "pixels", step, device), device=device)
    coords = torch.rand(batch, collocation_count, 2, generator=rng(seed, "collocation", step, device), device=device)*2-1
    brng = rng(seed, "boundary", step, device)
    t = torch.rand(batch, boundary_count, 1, generator=brng, device=device)*2-1
    edge = torch.randint(4, (batch, boundary_count, 1), generator=brng, device=device)
    bx = torch.where(edge == 0, -torch.ones_like(t), torch.where(edge == 1, torch.ones_like(t), t))
    by = torch.where(edge == 2, -torch.ones_like(t), torch.where(edge == 3, torch.ones_like(t), t))
    return posterior, indices, coords, torch.cat((bx, by), -1)


def coefficients(condition, overrides=None):
    values = dict(COEFFICIENT_DEFAULTS)
    if overrides:
        if set(overrides)-set(values):
            raise ValueError("Unknown coefficient override")
        values.update(overrides)
    area = torch.expm1(condition[:, 9:10]*math.log(90001))/90000
    base = area.clamp_min(16/90000).sqrt().clamp(.018, .55)
    return {
        "diffusivity": values["diffusivity_intercept"]+.010*condition[:, 4:5]+.004*condition[:, 5:6],
        "transport_x": (.05+.15*condition[:, 1:2])*condition[:, 12:13],
        "transport_y": (.05+.15*condition[:, 1:2])*condition[:, 11:12],
        "source": .08+values["source_line_energy_coefficient"]*condition[:, 2:3]+.05*condition[:, 0:1],
        "decay": values["decay_intercept"]+.35*condition[:, 1:2],
        "center": condition[:, 7:9],
        "sigma": torch.cat(((base*torch.exp(.25*condition[:, 10:11])).clamp(.018, .60),
                            (base*torch.exp(-.25*condition[:, 10:11])).clamp(.018, .60)), 1),
    }


def field_derivatives(decoder, condition, latent, coordinates):
    coordinates = coordinates.detach().clone().requires_grad_(True)
    field = decoder(coordinates, condition, latent)
    grad = torch.autograd.grad(field, coordinates, torch.ones_like(field), create_graph=True)[0]
    hxx = torch.autograd.grad(grad[..., :1], coordinates, torch.ones_like(grad[..., :1]), create_graph=True, retain_graph=True)[0][..., :1]
    hyy = torch.autograd.grad(grad[..., 1:], coordinates, torch.ones_like(grad[..., 1:]), create_graph=True, retain_graph=True)[0][..., 1:]
    return field, grad, hxx+hyy, coordinates


def residual_from_derivatives(field, grad, laplacian, coordinates, condition, overrides=None):
    coeff = coefficients(condition, overrides)
    gaussian = torch.exp(-.5*((coordinates-coeff["center"][:, None])/coeff["sigma"][:, None]).square().sum(-1, keepdim=True))
    residual = (-coeff["diffusivity"][:, None]*laplacian+coeff["transport_x"][:, None]*grad[..., :1]
                +coeff["transport_y"][:, None]*grad[..., 1:]+coeff["decay"][:, None]*field-coeff["source"][:, None]*gaussian)
    return residual, gaussian


def moment_terms(field, coordinates, condition):
    weights = field.squeeze(-1).square()+1e-6
    norm = weights.sum(1, keepdim=True)
    center = (weights[..., None]*coordinates).sum(1)/norm
    variance = (weights[..., None]*(coordinates-center[:, None]).square()).sum(1)/norm
    target = coefficients(condition)
    spread_delta = variance.clamp_min(1e-6).sqrt()-target["sigma"]
    loss = (center-target["center"]).square().mean()+.25*spread_delta.square().mean()
    return loss, torch.linalg.vector_norm(center-target["center"], dim=1), spread_delta.square().mean(1).sqrt()


def regularizer_components(decoder, condition, latent, coordinates, boundary, moment_grid, overrides=None):
    field, grad, lap, coords = field_derivatives(decoder, condition, latent, coordinates)
    residual, gaussian = residual_from_derivatives(field, grad, lap, coords, condition, overrides)
    magnitude = residual.detach().abs()
    attention = .2+.8*magnitude/magnitude.amax(1, keepdim=True).clamp_min(1e-6)
    bfield = decoder(boundary, condition, latent)
    mfield = decoder(moment_grid, condition, latent)
    moment, _, _ = moment_terms(mfield, moment_grid, condition)
    return {"proxy": (attention*residual.square()).mean(), "boundary": bfield.square().mean(),
            "moment": moment, "blob": (field-gaussian).square().mean(), "laplacian": lap.square().mean()}


def objective_weights(arm):
    arm = canonical_arm(arm)
    weights = {"proxy": 0., "boundary": 0., "moment": 0., "blob": 0., "laplacian": 0.}
    if arm in ("F10", "F11") or arm in SENSITIVITY:
        weights["proxy"] = .035
    if arm in ("F01", "F11", "G2N") or arm in SENSITIVITY:
        weights.update(boundary=.01, moment=.02)
    if arm == "G2N":
        weights.update(blob=.035, laplacian=.035)
    return weights


def condition_tensor(rows, device="cpu"):
    return torch.from_numpy(np.stack([condition_from_manifest(r).vector for r in rows])).to(device)


def image_tensor(rows):
    c = common()
    images = []
    for row in rows:
        expected = row.get("image_sha256")
        if expected and c.sha256_file(c.resolve_path(row["image_path"])) != expected:
            raise RuntimeError(f"Image changed after split lock: {row['sample_id']}")
        with Image.open(c.resolve_path(row["image_path"])) as image:
            images.append(np.asarray(image.convert("L").resize((64, 64), Image.Resampling.BILINEAR), dtype=np.float32)/255.)
    return torch.from_numpy(np.stack(images)[:, None])


def sorted_rows(split):
    if split not in ("train", "validation", "test"):
        raise ValueError("Explicit valid split required")
    c = common()
    if split == "test":
        c.require_test_unlocked()
    rows = sorted(c.load_rows(split), key=lambda r: r["sample_id"])
    if not rows or any(r.get("source_kind") != "real" or r["split"] != split for r in rows):
        raise RuntimeError("Invalid real split manifest")
    if len({r["sample_id"] for r in rows}) != len(rows):
        raise RuntimeError("Duplicate sample IDs")
    return rows


def atomic_torch_save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+".tmp")
    torch.save(value, temp)
    temp.replace(path)


def atomic_numpy_save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+".tmp")
    with temp.open("wb") as handle:
        np.save(handle, value, allow_pickle=False)
    temp.replace(path)


def check_completed(directory, identity):
    c = common()
    receipt = directory/"summary.json"
    if not receipt.exists():
        return None
    value = c.load_json(receipt)
    if value.get("status") != "completed":
        return None
    if value.get("identity_sha256") != object_hash(identity):
        raise RuntimeError(f"Completed artifact identity differs; replacement forbidden: {directory}")
    for relative, expected in value.get("artifact_hashes", {}).items():
        if c.sha256_file(directory/relative) != expected:
            raise RuntimeError(f"Completed artifact corrupted: {directory/relative}")
    if not value.get("artifact_hashes"):
        raise RuntimeError("Completed artifact has no integrity ledger")
    return value


def identity_for(task, config, inputs):
    c = common()
    return {"config": config, "provenance": c.run_provenance(task, config, inputs)}


def save_identity(directory, identity):
    c = common()
    path = directory/"identity.json"
    if path.exists():
        if c.load_json(path) != identity:
            raise RuntimeError(f"Run identity mismatch: {directory}")
    else:
        c.write_json(path, identity)


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


def fitted_config(arm, seed):
    c = common()
    p = c.load_protocol("T1")
    arm = canonical_arm(arm)
    allowed = c.load_protocol("T3C")["design"]["seeds"] if arm in SENSITIVITY else p["fresh_training"]["seeds"]
    if seed not in allowed:
        raise ValueError(f"Seed {seed} is not frozen for {arm}")
    return {"arm": arm, "seed": seed, "base": p["generator_base"], "weights": objective_weights(arm),
            "coefficients": dict([SENSITIVITY[arm]]) if arm in SENSITIVITY else {},
            "protocols": ["T1", "T3A", "T3C"], "runtime_schema": 1}


@torch.no_grad()
def validation_mse(model, images, conditions, device, batch_size=32):
    model.eval()
    total, pixels = 0., 0
    for start in range(0, len(images), batch_size):
        im, cond = images[start:start+batch_size].to(device), conditions[start:start+batch_size].to(device)
        mu, _ = model.encode(im, cond)
        prediction = render_field(model, cond, mu, 64)
        total += float((prediction-im).square().sum())
        pixels += im.numel()
    return total/pixels


def train(arm, seed):
    c = common()
    if (c.REV_ROOT/"T7_UNLOCK.json").exists():
        raise PermissionError("No fitting after T7 unlock")
    config = fitted_config(arm, seed)
    arm = config["arm"]
    task = "T3C" if arm in SENSITIVITY else "T3A" if arm == "G2N" else "T1"
    c.task_dir(task)
    identity = identity_for(task, config, {"train_manifest": c.manifest_path("train"), "validation_manifest": c.manifest_path("validation")})
    directory = c.generator_run_dir(arm, seed)
    complete = check_completed(directory, identity)
    if complete:
        return complete
    directory.mkdir(parents=True, exist_ok=True)
    save_identity(directory, identity)
    config_file = directory/"resolved_config.yaml"
    if not config_file.exists():
        c.atomic_text(config_file, yaml.safe_dump(config, sort_keys=False))
    c.seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base = config["base"]
    tr, va = sorted_rows("train"), sorted_rows("validation")
    images, val_images = image_tensor(tr), image_tensor(va)
    conditions, val_conditions = condition_tensor(tr), condition_tensor(va)
    model_raw = dict(base["model"])
    model_raw["fourier_frequencies"] = tuple(model_raw["fourier_frequencies"])
    model_raw["decoder_type"] = "mlp" if arm == "G0" else "pirate"
    model = ConditionalVariationalField(ModelConfig(**model_raw), 64).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=base["optimizer"]["learning_rate"], weight_decay=base["optimizer"]["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=base["epochs"], eta_min=base["scheduler"]["eta_min"])
    grid, mg = make_coordinate_grid(64, device), make_coordinate_grid(base["moment_grid_size"], device)
    batch_size = base["batch_size"]
    steps_epoch = len(images)//batch_size
    history, totals = [], {}
    epoch, batch_index, best_epoch, best_mse, elapsed_prior = 0, 0, -1, float("inf"), 0.
    prior_peak_allocated, prior_peak_reserved = 0, 0
    saved_rng = None
    resume_path = directory/"resume.pt"
    if resume_path.exists():
        saved = torch.load(resume_path, map_location=device, weights_only=False)
        if saved["identity_sha256"] != object_hash(identity):
            raise RuntimeError("Resume identity mismatch")
        model.load_state_dict(saved["model_state"])
        opt.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        epoch, batch_index = saved["epoch"], saved["batch_index"]
        history, totals, best_mse, best_epoch = saved["history"], saved["totals"], saved["best_mse"], saved["best_epoch"]
        elapsed_prior = saved["elapsed_seconds"]
        prior_peak_allocated = saved.get("peak_allocated_cuda_bytes", 0)
        prior_peak_reserved = saved.get("peak_reserved_cuda_bytes", 0)
        if saved.get("best_checkpoint_sha256") is not None:
            if not (directory/"best.pt").exists() or c.sha256_file(directory/"best.pt") != saved["best_checkpoint_sha256"]:
                raise RuntimeError("Best/resume checkpoint transaction differs; technical recovery required")
        saved_rng = saved["rng"]
        # CPU RNG state is serialized as CPU bytes even when weights load onto CUDA.
        saved_rng["torch"] = saved_rng["torch"].cpu()
        saved_rng["cuda"] = [state.cpu() for state in saved_rng["cuda"]]
        restore_rng(saved_rng)
        c.record_event("generator_resume", {"arm": arm, "seed": seed, "epoch": epoch, "batch_index": batch_index})
    elif (directory/"best.pt").exists():
        raise RuntimeError("Partial checkpoint without full resume state; technical recovery required")
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    c.record_event("generator_train_start", {"arm": arm, "seed": seed, "identity_sha256": object_hash(identity)})

    def save_resume():
        atomic_torch_save(resume_path, {**model.checkpoint_payload(), "optimizer": opt.state_dict(),
            "scheduler": scheduler.state_dict(), "rng": rng_state(), "epoch": epoch, "batch_index": batch_index,
            "history": history, "totals": totals, "best_mse": best_mse, "best_epoch": best_epoch,
            "elapsed_seconds": elapsed_prior+time.perf_counter()-started, "identity_sha256": object_hash(identity),
            "best_checkpoint_sha256": c.sha256_file(directory/"best.pt") if (directory/"best.pt").exists() else None,
            "peak_allocated_cuda_bytes": max(prior_peak_allocated, torch.cuda.max_memory_allocated()) if device.type == "cuda" else 0,
            "peak_reserved_cuda_bytes": max(prior_peak_reserved, torch.cuda.max_memory_reserved()) if device.type == "cuda" else 0,
            "sampler_rule": "epoch permutation from per-seed independent stream; next batch_index is stored"})

    if not resume_path.exists():
        save_resume()
    try:
        while epoch < base["epochs"]:
            model.train()
            order = torch.randperm(len(images), generator=rng(seed, "image_order", epoch))
            while batch_index < steps_epoch:
                idx = order[batch_index*batch_size:(batch_index+1)*batch_size]
                im, cond = images[idx].to(device), conditions[idx].to(device)
                step = epoch*steps_epoch+batch_index
                noise, pixels, colloc, boundary = draw_training_inputs(seed, step, batch_size, 4096, base["pixel_samples"],
                    model.config.latent_dim, base["collocation_samples"], base["boundary_samples"], device)
                mu, logvar = model.encode(im, cond)
                latent = mu+torch.exp(.5*logvar)*noise
                coordinates = grid[pixels]
                prediction = model.decode(coordinates, cond, latent)
                targets = torch.gather(im[:, 0].reshape(batch_size, -1, 1), 1, pixels[..., None])
                reconstruction = weighted_reconstruction(prediction, targets)
                kl = kl_divergence(mu, logvar)
                regularizers = regularizer_components(model.decoder, cond, latent, colloc, boundary,
                    mg[None].expand(batch_size, -1, -1), config["coefficients"])
                loss = reconstruction+base["kl_weight"]*kl+min(1., (epoch+1)/base["physics_warmup_epochs"])*sum(
                    config["weights"][name]*value for name, value in regularizers.items())
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite planned training loss")
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), base["gradient_clip"], error_if_nonfinite=True)
                opt.step()
                metrics = {"loss": loss, "reconstruction": reconstruction, "kl": kl, **regularizers}
                for key, value in metrics.items():
                    totals[key] = totals.get(key, 0.)+float(value.detach())
                batch_index += 1
                if batch_index % 32 == 0:
                    save_resume()
            scheduler.step()
            val = validation_mse(model, val_images, val_conditions, device)
            record = {"epoch": epoch+1, **{key: value/steps_epoch for key, value in totals.items()}, "validation_mse": val}
            history.append(record)
            if val < best_mse:
                best_mse, best_epoch = val, epoch+1
                atomic_torch_save(directory/"best.pt", {**model.checkpoint_payload(), "epoch": best_epoch,
                    "validation": {"mse": best_mse}, "seed": seed, "arm": arm,
                    "decoder_type": model.config.decoder_type, "use_physics": any(config["weights"].values()),
                    "identity_sha256": object_hash(identity)})
            epoch, batch_index, totals = epoch+1, 0, {}
            save_resume()
            print(json.dumps({"arm": arm, "seed": seed, **record}), flush=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
        c.write_csv(directory/"history.csv", history, overwrite=True)
        summary = {"status": "completed", "arm": arm, "seed": seed, "best_epoch": best_epoch,
            "best_validation_mse": best_mse, "epochs": epoch, "steps": epoch*steps_epoch,
            "elapsed_seconds": elapsed_prior+time.perf_counter()-started,
            "peak_allocated_cuda_bytes": max(prior_peak_allocated, torch.cuda.max_memory_allocated()) if device.type == "cuda" else None,
            "peak_reserved_cuda_bytes": max(prior_peak_reserved, torch.cuda.max_memory_reserved()) if device.type == "cuda" else None,
            "identity_sha256": object_hash(identity), "finished_utc": c.utc_now(),
            "artifact_hashes": {name: c.sha256_file(directory/name) for name in ("best.pt", "resume.pt", "history.csv", "identity.json", "resolved_config.yaml")}}
        c.write_json(directory/"summary.json", summary)
        c.append_run_log(task, f"Completed generator {arm}, seed {seed}; checkpoint epoch {best_epoch}; full identity and artifacts hashed.")
        c.record_event("generator_train_complete", {"arm": arm, "seed": seed, "summary_sha256": c.sha256_file(directory/"summary.json")})
        return summary
    except BaseException as exc:
        # Resume only from the last atomically persisted fully successful step;
        # never serialize potentially partially-updated optimizer state here.
        c.record_event("generator_train_failed", {"arm": arm, "seed": seed, "error": repr(exc),
                       "resume_state_sha256": c.sha256_file(resume_path) if resume_path.exists() else None})
        raise
    finally:
        del model, opt
        if device.type == "cuda":
            torch.cuda.empty_cache()


def pixel_blend(left, right, alpha):
    if not 0 <= alpha <= 1:
        raise ValueError("Alpha outside convex range")
    return alpha*np.asarray(left, dtype=np.float32)+(1-alpha)*np.asarray(right, dtype=np.float32)


def output_pixels(image):
    # Float-image resize precedes clip/round for BOTH neural and pixel arms.
    value = np.asarray(Image.fromarray(np.asarray(image, dtype=np.float32)).resize((300, 300), Image.Resampling.BILINEAR))
    return np.clip(np.rint(np.clip(value, 0., 1.)*255.), 0, 255).astype(np.uint8)


def synthetic_label(entry, pool_arm, image_path):
    target = entry["target"]
    label = {"synthetic.generated": True, "synthetic.pool": pool_arm, "synthetic.plan_index": entry["plan_index"],
             "synthetic.donor_a": entry["donor_a"], "synthetic.donor_b": entry["donor_b"],
             "synthetic.alpha": entry["alpha"], "synthetic.generator_seed": entry["generator_seed"],
             "synthetic.target_sample_id": entry["target_sample_id"], "image.path": str(image_path),
             "image.width": 300, "image.height": 300,
             "material.name": target["material"], "condition.laser.power": float(target["laser_power_w"]),
             "condition.scan.speed": float(target["scan_speed_mm_s"])}
    for short, column in (("x", "bbox_x_px"), ("y", "bbox_y_px"), ("width", "bbox_width_px"), ("height", "bbox_height_px")):
        label[f"anotation.bbox.{short}"] = float(target[column])
    return label


def verify_pool_rows(rows, entries):
    c = common()
    if len(rows) != len(entries):
        raise RuntimeError("Pool count mismatch")
    for row, entry in zip(rows, entries):
        if row["source_kind"] != "synthetic" or row["split"] != "train":
            raise RuntimeError("Pool split/source mismatch")
        if (row["donor_a"], row["donor_b"], float(row["interpolation_weight"])) != (entry["donor_a"], entry["donor_b"], entry["alpha"]):
            raise RuntimeError("Donor triple mismatch")
        label = c.load_json(c.resolve_path(row["label_path"]))
        for short, column in (("x", "bbox_x_px"), ("y", "bbox_y_px"), ("width", "bbox_width_px"), ("height", "bbox_height_px")):
            if float(row[column]) != float(label[f"anotation.bbox.{short}"]) or float(row[column]) != float(entry["target"][column]):
                raise RuntimeError("JSON/manifest/canonical-target box mismatch")
        for key in ("image", "label"):
            if c.sha256_file(c.resolve_path(row[f"{key}_path"])) != row[f"{key}_sha256"]:
                raise RuntimeError(f"Pool {key} hash mismatch")
    return {"rows": len(rows), "donor_triple_mismatches": 0, "box_mismatches": 0}


def make_pool(arm):
    if arm not in ("G1", "G2", "D5"):
        raise ValueError("Unplanned pool arm")
    c = common()
    if (c.REV_ROOT/"T7_UNLOCK.json").exists():
        raise PermissionError("No pool construction after T7 unlock")
    protocol = c.load_protocol("T3B")
    c.task_dir("T3B")
    plan_path = plan_root()/"pool_plan.json"
    expected_plan_identity = identity_for("T3B", {"kind": "pool_plan", "pool": protocol["pool"], "schema": 1},
                                         {"train_manifest": c.manifest_path("train")})
    plan = verified_plan(plan_path, expected_plan_identity)
    entries = plan["entries"]
    tr = sorted_rows("train")
    if plan["train_sample_ids"] != [r["sample_id"] for r in tr] or len(entries) != 7168:
        raise RuntimeError("Pool plan does not match frozen train roster")
    inputs = {"plan": plan_path, "train_manifest": c.manifest_path("train")}
    if arm != "D5":
        for seed in c.load_protocol("T1")["fresh_training"]["seeds"]:
            inputs[f"checkpoint_{seed}"] = c.generator_run_dir(canonical_arm(arm), seed)/"best.pt"
    identity = identity_for("T3B", {"kind": "pool", "arm": arm, "schema": 1}, inputs)
    directory = c.pool_dir(arm)
    existing = check_completed(directory, identity)
    if existing:
        verify_pool_rows(c.read_csv(directory/"manifest.csv"), entries)
        return existing
    directory.mkdir(parents=True, exist_ok=True)
    save_identity(directory, identity)
    (directory/"images").mkdir(exist_ok=True)
    (directory/"labels").mkdir(exist_ok=True)
    c.record_event("generator_pool_start", {"arm": arm, "plan_sha256": c.sha256_file(plan_path)})
    train_images, train_conditions = image_tensor(tr), condition_tensor(tr)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaded_seed, model, bank = None, None, None
    rows = []
    started = time.perf_counter()
    try:
        for start in range(0, len(entries), 16):
            current = entries[start:start+16]
            # Seed-allocation boundaries can fall inside a batch: render each
            # consecutive seed subgroup, preserving the exact original order.
            for seed, iterator in itertools.groupby(current, key=lambda e: e["generator_seed"]):
                batch = list(iterator)
                if arm == "D5":
                    arrays = [pixel_blend(train_images[e["donor_a_index"], 0].numpy(), train_images[e["donor_b_index"], 0].numpy(), e["alpha"]) for e in batch]
                else:
                    if loaded_seed != seed:
                        if model is not None:
                            del model, bank
                        model, _ = load_fitted(arm, seed, device)
                        bank = encode_bank(model, train_images, train_conditions, device)
                        loaded_seed = seed
                    arrays = render_field(model, condition_tensor([e["target"] for e in batch], device), plan_latent(batch, bank), 64)[:, 0].cpu().numpy()
                for entry, array in zip(batch, arrays):
                    name = f"{arm}_{entry['plan_index']:06d}"
                    ipath, lpath = directory/"images"/f"{name}.png", directory/"labels"/f"{name}.json"
                    pixels = output_pixels(array)
                    if ipath.exists():
                        with Image.open(ipath) as old_image:
                            if not np.array_equal(np.asarray(old_image), pixels):
                                raise RuntimeError("Partial pool output differs; technical recovery required")
                    else:
                        tmp = ipath.with_suffix(".png.tmp")
                        Image.fromarray(pixels).save(tmp, format="PNG")
                        tmp.replace(ipath)
                    label = synthetic_label(entry, arm, ipath)
                    if lpath.exists():
                        if c.load_json(lpath) != label:
                            raise RuntimeError("Partial pool label differs")
                    else:
                        c.write_json(lpath, label)
                    target = entry["target"]
                    row = {**target, "sample_id": name, "synthetic_id": name, "source_kind": "synthetic", "split": "train",
                        "image_path": str(ipath), "label_path": str(lpath), "image_sha256": c.sha256_file(ipath),
                        "label_sha256": c.sha256_file(lpath), "pool_arm": arm, "seed": seed,
                        "generator_seed": seed, "plan_index": entry["plan_index"],
                        "conditioning_image_path": target["image_path"], "conditioning_label_path": target["label_path"],
                        "donor_a": entry["donor_a"], "donor_b": entry["donor_b"], "interpolation_weight": entry["alpha"],
                        "target_sample_id": entry["target_sample_id"], "latent_mode": "pixel_blend" if arm == "D5" else "train_posterior_interpolation"}
                    rows.append(row)
        c.write_csv(directory/"manifest.csv", rows, overwrite=True)
        integrity = verify_pool_rows(c.read_csv(directory/"manifest.csv"), entries)
        result = {"status": "completed", "arm": arm, "source_scope": "REV02_train_only", **integrity,
            "ordered_triples_sha256": object_hash([[e["donor_a"], e["donor_b"], e["alpha"]] for e in entries]),
            "self_pairs": sum(e["donor_a_id"] == e["donor_b_id"] for e in entries),
            "elapsed_seconds": time.perf_counter()-started, "finished_utc": c.utc_now(), "identity_sha256": object_hash(identity),
            "artifact_hashes": {name: c.sha256_file(directory/name) for name in ("manifest.csv", "identity.json")}}
        c.write_json(directory/"summary.json", result, overwrite=True)
        c.append_run_log("T3B", f"Completed {arm} pool; {len(rows)} train-only rows; zero donor/box mismatches.")
        c.record_event("generator_pool_complete", {"arm": arm, "summary_sha256": c.sha256_file(directory/"summary.json")})
        return result
    except BaseException as exc:
        c.record_event("generator_pool_failed", {"arm": arm, "error": repr(exc)})
        raise
    finally:
        if model is not None:
            del model, bank
        if device.type == "cuda":
            torch.cuda.empty_cache()


def discrete_proxy_rms(field, condition):
    height, width = field.shape
    dy, dx = 2/(height-1), 2/(width-1)
    gy, gx = np.gradient(np.asarray(field, dtype=np.float64), dy, dx)
    lap = np.gradient(gx, dx, axis=1)+np.gradient(gy, dy, axis=0)
    p = coefficients(torch.as_tensor(condition, dtype=torch.float64)[None])
    yy, xx = np.meshgrid(np.linspace(-1, 1, height), np.linspace(-1, 1, width), indexing="ij")
    center, sigma = p["center"][0].numpy(), p["sigma"][0].numpy()
    gaussian = np.exp(-.5*(((xx-center[0])/sigma[0])**2+((yy-center[1])/sigma[1])**2))
    residual = (-float(p["diffusivity"][0, 0])*lap+float(p["transport_x"][0, 0])*gx
                +float(p["transport_y"][0, 0])*gy+float(p["decay"][0, 0])*field-float(p["source"][0, 0])*gaussian)
    return float(np.sqrt(np.mean(residual**2)))


def stratified_halves(rows, seed):
    groups = defaultdict(list)
    for index, row in enumerate(rows):
        groups[(row["stratum"], row["view"])].append(index)
    left, right = [], []
    gen = np.random.default_rng(seed)
    for key in sorted(groups):
        indices = gen.permutation(groups[key])
        if len(indices) < 2 or len(indices) % 2:
            raise RuntimeError("Real-real strata must support exact equal halves")
        mid = len(indices)//2
        left.extend(indices[:mid].tolist())
        right.extend(indices[mid:].tolist())
    return np.asarray(left), np.asarray(right)


def controls(split):
    c = common()
    if split == "test":
        c.require_test_unlocked()
        for seed in c.load_protocol("MASTER")["seeds"]["generator"]:
            verify_test_roster_entry("F00", seed)
    elif split != "validation":
        raise ValueError("Explicit validation/test control split required")
    c.load_protocol("T1")
    seeds = c.load_protocol("MASTER")["seeds"]["generator"]
    directory = c.ARTIFACT_ROOT/"evaluation"/"generator_controls"/split
    inputs = {"manifest": c.manifest_path(split), "scaler": plan_root()/"feature_scaler.json"}
    for seed in seeds:
        evaluation = evaluation_dir("F00", seed, split)
        summary = verify_current_artifact(evaluation)
        if (summary["arm"], summary["seed"], summary["split"]) != ("F00", seed, split):
            raise RuntimeError("Blur source generator evaluation identity differs")
        inputs[f"G1_generated_{seed}"] = evaluation/"generated64.npy"
    identity = identity_for("T1", {"kind": "blur_and_real_real_controls", "split": split, "schema": 1}, inputs)
    existing = check_completed(directory, identity)
    if existing:
        return existing
    if split == "test" and (directory/"identity.json").exists():
        raise RuntimeError("Test controls already started; documented recovery required")
    directory.mkdir(parents=True, exist_ok=True)
    save_identity(directory, identity)
    c.record_event("generator_controls_start", {"split": split})
    rows = sorted_rows(split)
    real = image_tensor(rows)[:, 0].numpy()
    cond = condition_tensor(rows).numpy()
    scaler = ensure_scaler()
    real_matrix = standardized(features(real), scaler)
    floor_rows = []
    for seed in range(61006, 61016):
        left, right = stratified_halves(rows, seed)
        floor_rows.append({"split": split, "seed": seed, "n_left": len(left), "n_right": len(right),
                           "PFFD_std": frechet_distance(real_matrix[left], real_matrix[right]),
                           "left_ids_sha256": object_hash([rows[i]["sample_id"] for i in left]),
                           "right_ids_sha256": object_hash([rows[i]["sample_id"] for i in right])})
    aggregate, per_image = [], []
    sources = [("real", None, real)]+[("G1", seed, None) for seed in seeds]
    for arm, seed, source in sources:
        if source is None:
            source = np.load(inputs[f"G1_generated_{seed}"], allow_pickle=False)
        if source.shape != real.shape:
            raise RuntimeError("Blur source shape differs")
        for sigma in (0., .5, 1., 2.):
            current = np.stack([ndimage.gaussian_filter(im, sigma=sigma, mode="reflect", truncate=4.) if sigma else im for im in source])
            rms = [discrete_proxy_rms(im, co) for im, co in zip(current, cond)]
            mse = np.mean((current-real)**2, axis=(1, 2))
            similarities = [ssim(a, b) for a, b in zip(current, real)]
            value = {"split": split, "arm": arm, "seed": seed, "sigma_px": sigma,
                     "discrete_proxy_RMS": float(np.mean(rms)),
                     "PFFD_std": frechet_distance(real_matrix, standardized(features(current), scaler)),
                     "matched_target_PSNR": -10*math.log10(max(float(np.mean(mse)), 1e-12)),
                     "matched_target_SSIM": float(np.mean(similarities)), "n_images": len(real)}
            aggregate.append(value)
            for i, row in enumerate(rows):
                per_image.append({"split": split, "arm": arm, "seed": seed, "sigma_px": sigma,
                    "sample_id": row["sample_id"], "specimen": row["specimen"], "discrete_proxy_RMS": rms[i],
                    "matched_target_mse": float(mse[i]), "matched_target_SSIM": similarities[i]})
    c.write_csv(directory/"blur_summary.csv", aggregate, overwrite=True)
    c.write_csv(directory/"blur_per_image.csv", per_image, overwrite=True)
    c.write_csv(directory/"real_real_floor.csv", floor_rows, overwrite=True)
    floor_values = [row["PFFD_std"] for row in floor_rows]
    result = {"status": "completed", "split": split, "blur_cells": len(aggregate),
        "real_real_floor_mean": float(np.mean(floor_values)), "real_real_floor_sd": float(np.std(floor_values, ddof=1)),
        "identity_sha256": object_hash(identity), "finished_utc": c.utc_now(),
        "artifact_hashes": {name: c.sha256_file(directory/name) for name in ("blur_summary.csv", "blur_per_image.csv", "real_real_floor.csv", "identity.json")},
        "scope": "Finite-difference proxy RMS is not numerically interchangeable with decoder-autograd RMS."}
    c.write_json(directory/"summary.json", result, overwrite=True)
    c.record_event("generator_controls_complete", {"split": split, "summary_sha256": c.sha256_file(directory/"summary.json")})
    return result


def psd_eigh(matrix):
    values, vectors = np.linalg.eigh((matrix+matrix.T)*.5)
    if values.min() < -1e-8:
        raise FloatingPointError(f"PSD eigenvalue defect {values.min()}")
    return np.maximum(values, 0.), vectors


def frechet_distance(real, generated):
    real, generated = np.asarray(real, dtype=np.float64), np.asarray(generated, dtype=np.float64)
    if real.ndim != 2 or generated.ndim != 2 or real.shape[1] != generated.shape[1] or min(len(real), len(generated)) < 2:
        raise ValueError("Frechet needs two finite feature matrices with n>=2")
    if not np.isfinite(real).all() or not np.isfinite(generated).all():
        raise ValueError("Nonfinite Frechet input")
    cr = np.cov(real, rowvar=False)+np.eye(real.shape[1])*1e-8
    cg = np.cov(generated, rowvar=False)+np.eye(real.shape[1])*1e-8
    vals, vecs = psd_eigh(cr)
    root = (vecs*np.sqrt(vals))@vecs.T
    inner, _ = psd_eigh(root@cg@root)
    value = float(np.sum((real.mean(0)-generated.mean(0))**2)+np.trace(cr)+np.trace(cg)-2*np.sqrt(inner).sum())
    if value < -1e-8:
        raise FloatingPointError(f"Negative Frechet distance {value}")
    return max(0., value)


def features(images):
    return np.asarray([[record[key] for key in FEATURES] for record in map(morphology_features, images)], dtype=np.float64)


def fit_scaler(matrix):
    center, scale = matrix.mean(0), matrix.std(0, ddof=0)
    replaced = np.flatnonzero(scale < 1e-6)
    scale[replaced] = 1.
    return {"features": list(FEATURES), "center": center.tolist(), "scale": scale.tolist(),
            "replaced_scale_features": [FEATURES[i] for i in replaced], "fit_scope": "train_only", "n": len(matrix)}


def standardized(matrix, scaler):
    return (matrix-np.asarray(scaler["center"]))/np.asarray(scaler["scale"])


def ssim(left, right):
    return float(structural_similarity(left, right, data_range=1., win_size=7,
        gaussian_weights=False, use_sample_covariance=True, channel_axis=None, K1=.01, K2=.03))


def select_diversity_rows(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["stratum"], row["view"])].append(row)
    selected = []
    for key in sorted(groups):
        group = sorted(groups[key], key=lambda r: r["sample_id"])
        if len(group) < 2:
            raise RuntimeError(f"Insufficient diversity-group rows: {key}")
        selected.extend(group[:2])
    return selected


def create_donor_plan(train_rows, target_rows, seed, allocations=None, evaluation=False):
    """Model-independent float64 donor selection, deterministic ties, explicit epsilon."""
    train_rows = sorted(train_rows, key=lambda r: r["sample_id"])
    bank = condition_tensor(train_rows).numpy().astype(np.float64)
    targets = condition_tensor(target_rows).numpy().astype(np.float64)
    gen = np.random.default_rng(seed)
    dims = [0, 1, 2, 7, 8, 9, 10]
    generator_seeds = [seed for seed, count in (allocations or []) for _ in range(count)]
    if allocations and len(generator_seeds) != len(target_rows):
        raise ValueError("Pool allocation does not match targets")
    entries = []
    for index, (row, target) in enumerate(zip(target_rows, targets)):
        candidates = np.asarray([j for j, r in enumerate(train_rows)
                                 if r["material"] == row["material"] and r["view"] == row["view"]], dtype=int)
        if not len(candidates):
            raise RuntimeError("No matching train donor; cross-material fallback prohibited")
        distances = np.mean((bank[candidates][:, dims]-target[dims])**2, axis=1)
        nearest = candidates[np.lexsort((candidates, distances))[:min(64, len(candidates))]]
        a, b = gen.choice(nearest, size=2, replace=True)
        entry = {"plan_index": index, "target_sample_id": row["sample_id"],
                 "donor_a_index": int(a), "donor_b_index": int(b),
                 "donor_a": train_rows[a]["image_path"], "donor_b": train_rows[b]["image_path"],
                 "donor_a_id": train_rows[a]["sample_id"], "donor_b_id": train_rows[b]["sample_id"],
                 "donor_a_sha256": train_rows[a].get("image_sha256", train_rows[a].get("sha256_image")),
                 "donor_b_sha256": train_rows[b].get("image_sha256", train_rows[b].get("sha256_image")),
                 "alpha": float(.2+.6*gen.random()), "epsilon": gen.standard_normal(16).tolist(),
                 "generator_seed": generator_seeds[index] if allocations else None,
                 "target": dict(row)}
        entries.append(entry)
    if evaluation:
        fresh_rng = np.random.default_rng(stream_seed(seed, "fresh_evaluation_coordinates"))
        for entry in entries:
            entry["collocation"] = fresh_rng.uniform(-1, 1, (32, 2)).tolist()
    return entries


def plan_root():
    root = common().ARTIFACT_ROOT/"plans"
    root.mkdir(parents=True, exist_ok=True)
    return root


def verified_plan(path, expected_identity=None):
    c = common()
    payload = c.load_json(path)
    digest = payload["payload_sha256"]
    body = {key: value for key, value in payload.items() if key != "payload_sha256"}
    if object_hash(body) != digest:
        raise RuntimeError(f"Plan hash mismatch: {path}")
    if expected_identity is not None and body["identity"] != expected_identity:
        raise RuntimeError(f"Plan identity mismatch: {path}")
    return body


def freeze_plan(path, body):
    c = common()
    if path.exists():
        existing = verified_plan(path, body["identity"])
        if existing != body:
            raise RuntimeError("Frozen plan differs; overwrite prohibited")
        return existing
    c.write_json(path, {**body, "payload_sha256": object_hash(body)})
    c.record_event("generator_plan_frozen", {"path": str(path), "sha256": c.sha256_file(path)})
    return body


def ensure_scaler(train_rows=None, train_images=None):
    c = common()
    path = plan_root()/"feature_scaler.json"
    identity = identity_for("T1", {"kind": "train_feature_scaler", "schema": 1}, {"train_manifest": c.manifest_path("train")})
    if path.exists():
        return verified_plan(path, identity)["scaler"]
    if (c.REV_ROOT/"T7_UNLOCK.json").exists():
        raise PermissionError("Missing scaler cannot be fitted after T7")
    rows = train_rows if train_rows is not None else sorted_rows("train")
    images = train_images if train_images is not None else image_tensor(rows)
    matrix = features(images[:, 0].numpy())
    body = {"identity": identity, "scaler": fit_scaler(matrix), "sample_ids": [r["sample_id"] for r in rows]}
    freeze_plan(path, body)
    return body["scaler"]


def ensure_evaluation_plan(split):
    c = common()
    if split not in ("validation", "test"):
        raise ValueError("Evaluation split must be explicit")
    if split == "test":
        c.require_test_unlocked()
    path = plan_root()/f"{split}_plan.json"
    seed = c.load_protocol("MASTER")["seeds"]["generator_evaluation"]
    identity = identity_for("T1", {"kind": "generator_evaluation_plan", "split": split, "seed": seed, "schema": 1},
                            {"train_manifest": c.manifest_path("train"), "target_manifest": c.manifest_path(split)})
    if path.exists():
        return verified_plan(path, identity)
    tr, target = sorted_rows("train"), sorted_rows(split)
    diversity = select_diversity_rows(target)
    if len(target) != 1024 or len(diversity) != 128:
        raise RuntimeError("Frozen evaluation sample counts differ")
    entries = create_donor_plan(tr, target, seed, evaluation=True)
    diversity_targets = [r for r in diversity for _ in range(4)]
    diversity_entries = create_donor_plan(tr, diversity_targets, stream_seed(seed, "diversity"))
    body = {"identity": identity, "split": split, "entries": entries, "diversity_entries": diversity_entries,
            "diversity_ids": [r["sample_id"] for r in diversity], "group_count": 64,
            "train_sample_ids": [r["sample_id"] for r in tr]}
    return freeze_plan(path, body)


def make_plans():
    c = common()
    if (c.REV_ROOT/"T7_UNLOCK.json").exists():
        raise PermissionError("Training pool plans cannot be created after T7")
    protocol = c.load_protocol("T3B")
    tr = sorted_rows("train")
    if len(tr) != 3584:
        raise RuntimeError("Unexpected train size")
    path = plan_root()/"pool_plan.json"
    identity = identity_for("T3B", {"kind": "pool_plan", "pool": protocol["pool"], "schema": 1}, {"train_manifest": c.manifest_path("train")})
    if path.exists():
        verified_plan(path, identity)
    else:
        for row in tr:
            actual = c.sha256_file(c.resolve_path(row["image_path"]))
            expected = row.get("image_sha256", row.get("sha256_image"))
            if expected and expected != actual:
                raise RuntimeError("Train source image hash mismatch")
            row["image_sha256"] = actual
        allocations = list(zip(protocol["pool"]["generator_seeds"], protocol["pool"]["allocations"]))
        entries = create_donor_plan(tr, [r for r in tr for _ in range(2)], protocol["pool"]["donor_plan_seed"], allocations)
        freeze_plan(path, {"identity": identity, "entries": entries, "train_sample_ids": [r["sample_id"] for r in tr]})
    ensure_scaler(tr)
    ensure_evaluation_plan("validation")
    return {"status": "completed", "plans": {p.name: c.sha256_file(p) for p in (path, plan_root()/"feature_scaler.json", plan_root()/"validation_plan.json")}}


def load_fitted(arm, seed, device):
    c = common()
    arm = canonical_arm(arm)
    directory = c.generator_run_dir(arm, seed)
    config = fitted_config(arm, seed)
    task = "T3C" if arm in SENSITIVITY else "T3A" if arm == "G2N" else "T1"
    identity = identity_for(task, config, {"train_manifest": c.manifest_path("train"), "validation_manifest": c.manifest_path("validation")})
    if not check_completed(directory, identity):
        raise RuntimeError(f"Generator not completed: {arm}/{seed}")
    return load_model(directory/"best.pt", device)


@torch.no_grad()
def encode_bank(model, images, conditions, device):
    bank = []
    for start in range(0, len(images), 32):
        mu, _ = model.encode(images[start:start+32].to(device), conditions[start:start+32].to(device))
        bank.append(mu)
    return torch.cat(bank)


def plan_latent(entries, bank):
    a = torch.tensor([r["donor_a_index"] for r in entries], device=bank.device)
    b = torch.tensor([r["donor_b_index"] for r in entries], device=bank.device)
    alpha = torch.tensor([r["alpha"] for r in entries], device=bank.device, dtype=bank.dtype)[:, None]
    noise = torch.tensor([r["epsilon"] for r in entries], device=bank.device, dtype=bank.dtype)
    return alpha*bank[a]+(1-alpha)*bank[b]+.08*noise


def profile_model(model, device):
    inputs = torch.zeros(1, 1, 64, 64, device=device)
    cond = torch.zeros(1, 13, device=device)
    grid = make_coordinate_grid(64, device)[None]
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    with torch.no_grad(), profile(activities=[ProfilerActivity.CPU]+([ProfilerActivity.CUDA] if device.type == "cuda" else []), with_flops=True) as trace:
        mu, _ = model.encode(inputs, cond)
        model.decode(grid, cond, mu)
        if device.type == "cuda":
            torch.cuda.synchronize()
    return {"parameters_total": sum(p.numel() for p in model.parameters()),
            "parameters_trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "profiled_forward_flops": sum(int(e.flops or 0) for e in trace.key_averages()),
            "inference_peak_allocated_cuda_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else None,
            "profile_input": "one64x64_encode_plus4096_coordinate_decode", "not_training_memory": True}


def evaluation_dir(arm, seed, split):
    if split == "test":
        common().require_test_unlocked()
    return common().ARTIFACT_ROOT/"evaluation"/"generators"/split/f"{canonical_arm(arm)}_s{seed}"


def verify_test_roster_entry(arm, seed):
    """Bind model identity before even preparing a test-data plan."""
    c = common()
    c.require_test_unlocked()
    arm = canonical_arm(arm)
    freeze = c.load_json(c.REV_ROOT/"PRETEST_FREEZE.json")
    roster = freeze["test_roster"]["generators"]
    matches = [row for row in roster if row["arm"] == arm and int(row["seed"]) == seed]
    if len(matches) != 1:
        raise PermissionError(f"Generator absent or duplicated in frozen test roster: {arm}/{seed}")
    entry = matches[0]
    current = c.generator_run_dir(arm, seed)/"best.pt"
    if Path(entry["checkpoint"]).resolve() != current.resolve():
        raise PermissionError("Frozen test checkpoint path differs")
    if c.sha256_file(current) != entry["checkpoint_sha256"]:
        raise PermissionError("Frozen test checkpoint hash differs")
    return entry


def evaluate(arm, seed, split):
    c = common()
    config = fitted_config(arm, seed)
    arm = config["arm"]
    if split == "test":
        verify_test_roster_entry(arm, seed)
    plan = ensure_evaluation_plan(split)
    scaler = ensure_scaler()
    task = "T3C" if arm in SENSITIVITY else "T3A" if arm == "G2N" else "T1"
    directory = evaluation_dir(arm, seed, split)
    ckpt = c.generator_run_dir(arm, seed)/"best.pt"
    identity = identity_for(task, {"arm": arm, "seed": seed, "split": split, "kind": "generator_evaluation", "schema": 1},
        {"checkpoint": ckpt, "plan": plan_root()/f"{split}_plan.json", "scaler": plan_root()/"feature_scaler.json"})
    existing = check_completed(directory, identity)
    if existing:
        return existing
    # A started but unfinished test evaluation is not silently repeated.
    if split == "test" and (directory/"identity.json").exists():
        raise RuntimeError("Test evaluation was already started; documented T7 technical recovery required")
    directory.mkdir(parents=True, exist_ok=True)
    save_identity(directory, identity)
    c.record_event("generator_evaluation_start", {"arm": arm, "seed": seed, "split": split})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_fitted(arm, seed, device)
    tr, target = sorted_rows("train"), sorted_rows(split)
    if [r["sample_id"] for r in tr] != plan["train_sample_ids"]:
        raise RuntimeError("Donor bank order mismatch")
    target_images = image_tensor(target)
    bank = encode_bank(model, image_tensor(tr), condition_tensor(tr), device)
    records, generated_arrays, diversity_records = [], [], []
    start_time = time.perf_counter()
    boundary, moment = fixed_boundary(device=device), make_coordinate_grid(16, device)
    try:
        for start in range(0, len(target), 12):
            entries = plan["entries"][start:start+12]
            rows = target[start:start+12]
            if [e["target_sample_id"] for e in entries] != [r["sample_id"] for r in rows]:
                raise RuntimeError("Target plan order mismatch")
            cond, real = condition_tensor(rows, device), target_images[start:start+12].to(device)
            latent = plan_latent(entries, bank)
            with torch.no_grad():
                mu, _ = model.encode(real, cond)
                recon = render_field(model, cond, mu, 64)
                generated = render_field(model, cond, latent, 64)
                benergy = model.decode(boundary[None].expand(len(rows), -1, -1), cond, latent).square().mean((1, 2))
                mcoords = moment[None].expand(len(rows), -1, -1)
                _, centroid, spread = moment_terms(model.decode(mcoords, cond, latent), mcoords, cond)
            fresh = torch.tensor([e["collocation"] for e in entries], device=device, dtype=torch.float32)
            field, grad, lap, coords = field_derivatives(model.decoder, cond, latent, fresh)
            common_res, _ = residual_from_derivatives(field, grad, lap, coords, cond)
            native_res, _ = residual_from_derivatives(field, grad, lap, coords, cond, config["coefficients"])
            rms = common_res.detach().square().mean((1, 2)).sqrt().cpu().numpy()
            native = native_res.detach().square().mean((1, 2)).sqrt().cpu().numpy()
            real_np, recon_np, gen_np = (x.detach().cpu().numpy()[:, 0] for x in (real, recon, generated))
            generated_arrays.extend(gen_np)
            for i, row in enumerate(rows):
                records.append({"arm": arm, "seed": seed, "split": split, "sample_id": row["sample_id"],
                    "specimen": row["specimen"], "stratum": row["stratum"], "view": row["view"],
                    "reconstruction_mse": float(np.mean((real_np[i]-recon_np[i])**2)),
                    "reconstruction_SSIM": ssim(real_np[i], recon_np[i]), "fresh_collocation_proxy_RMS": float(rms[i]),
                    "fresh_proxy_RMS_variant_native_coefficients": float(native[i]),
                    "boundary_energy": float(benergy[i].cpu()), "moment_centroid_error": float(centroid[i].cpu()),
                    "moment_spread_error": float(spread[i].cpu())})
            del field, grad, lap, common_res, native_res
        with torch.no_grad():
            for start in range(0, len(plan["diversity_entries"]), 4):
                entries = plan["diversity_entries"][start:start+4]
                cond = condition_tensor([e["target"] for e in entries], device)
                images = render_field(model, cond, plan_latent(entries, bank), 64)[:, 0].cpu().numpy()
                value = float(np.mean([1-ssim(images[a], images[b]) for a, b in itertools.combinations(range(4), 2)]))
                diversity_records.append({"sample_id": entries[0]["target_sample_id"], "stratum": entries[0]["target"]["stratum"],
                                          "view": entries[0]["target"]["view"], "within_condition_one_minus_SSIM": value})
        generated_arrays = np.asarray(generated_arrays, dtype=np.float32)
        real_features, gen_features = features(target_images[:, 0].numpy()), features(generated_arrays)
        pffd = frechet_distance(standardized(real_features, scaler), standardized(gen_features, scaler))
        metrics = {name: float(np.mean([r[name] for r in records])) for name in (
            "fresh_collocation_proxy_RMS", "boundary_energy", "moment_centroid_error", "moment_spread_error",
            "reconstruction_SSIM", "reconstruction_mse", "fresh_proxy_RMS_variant_native_coefficients")}
        metrics.update(reconstruction_PSNR=-10*math.log10(max(metrics["reconstruction_mse"], 1e-12)), PFFD_std=pffd,
            within_condition_one_minus_SSIM=float(np.mean([r["within_condition_one_minus_SSIM"] for r in diversity_records])),
            fresh_proxy_RMS_common_baseline_coefficients=metrics["fresh_collocation_proxy_RMS"])
        c.write_csv(directory/"per_image_metrics.csv", records, overwrite=True)
        c.write_csv(directory/"diversity_per_condition.csv", diversity_records, overwrite=True)
        atomic_numpy_save(directory/"generated64.npy", generated_arrays)
        profile_row = profile_model(model, device) if seed == 41001 and arm in ("F11", "G2N") else None
        if profile_row:
            c.write_json(directory/"capacity_profile.json", profile_row, overwrite=True)
        artifacts = ["per_image_metrics.csv", "diversity_per_condition.csv", "generated64.npy", "identity.json"]
        if profile_row:
            artifacts.append("capacity_profile.json")
        result = {"status": "completed", "arm": arm, "seed": seed, "split": split, "n_images": len(records),
            "n_diversity_conditions": len(diversity_records), **metrics, "profile": profile_row,
            "elapsed_seconds": time.perf_counter()-start_time, "finished_utc": c.utc_now(),
            "identity_sha256": object_hash(identity), "artifact_hashes": {name: c.sha256_file(directory/name) for name in artifacts}}
        c.write_json(directory/"summary.json", result, overwrite=True)
        c.record_event("generator_evaluation_complete", {"arm": arm, "seed": seed, "split": split, "summary_sha256": c.sha256_file(directory/"summary.json")})
        return result
    except BaseException as exc:
        c.record_event("generator_evaluation_failed", {"arm": arm, "seed": seed, "split": split, "error": repr(exc)})
        raise
    finally:
        del model, bank
        if device.type == "cuda":
            torch.cuda.empty_cache()


def planned_runs():
    c = common()
    seeds = c.load_protocol("T1")["fresh_training"]["seeds"]
    short = c.load_protocol("T3C")["design"]["seeds"]
    return [(arm, seed) for arm in BASE_ARMS for seed in seeds]+[(arm, seed) for arm in SENSITIVITY for seed in short]


def verify_current_artifact(directory):
    """Rebind an artifact to CURRENT protocol/code/environment and input bytes."""
    c = common()
    saved = c.load_json(directory/"identity.json")
    inputs = {name: item["path"] for name, item in saved["provenance"]["input_hashes"].items()}
    expected = identity_for(saved["provenance"]["task_id"], saved["config"], inputs)
    result = check_completed(directory, expected)
    if not result:
        raise RuntimeError(f"Incomplete artifact: {directory}")
    return result


def descriptive_G2N(rows):
    by_key = {(r["arm"], int(r["seed"]), r["split"]): r for r in rows}
    outputs = []
    for split in sorted({r["split"] for r in rows}):
        for endpoint in PRIMARY:
            differences = [float(by_key[("G2N", seed, split)][endpoint])-float(by_key[("F11", seed, split)][endpoint])
                           for seed in range(41001, 41011)]
            mean, sd = float(np.mean(differences)), float(np.std(differences, ddof=1))
            outputs.append({"split": split, "endpoint": endpoint, "n": len(differences),
                            "mean_paired_difference_G2N_minus_G2": mean, "sample_SD_of_paired_differences": sd,
                            "descriptive_within_seed_SD": abs(mean) <= sd,
                            "equivalence_claim": False})
    return outputs


def summarize(split="validation"):
    c = common()
    if split not in ("validation", "all"):
        raise ValueError("Summary split must be validation or gated all")
    splits = ["validation"]
    if split == "all":
        c.require_test_unlocked()
        splits.append("test")
    rows, missing = [], []
    for partition in splits:
        for arm, seed in planned_runs():
            directory = evaluation_dir(arm, seed, partition)
            if not (directory/"summary.json").exists():
                missing.append(f"{partition}/{arm}/{seed}")
                continue
            summary = verify_current_artifact(directory)
            rows.append({"arm": arm, "seed": seed, "split": partition,
                **{endpoint: summary[endpoint] for endpoint in (*PRIMARY, "reconstruction_SSIM")},
                "fresh_proxy_RMS_variant_native_coefficients": summary["fresh_proxy_RMS_variant_native_coefficients"],
                "summary_sha256": c.sha256_file(directory/"summary.json")})
    t1 = c.task_dir("T1")
    if rows:
        c.write_csv(t1/"raw"/"generator_per_seed_endpoints.csv", rows, overwrite=True)
    if missing:
        c.append_run_log("T1", f"Generator summary incomplete; missing {len(missing)} fixed cells: {missing}")
        raise RuntimeError(f"Generator summary incomplete: {missing}")
    control_rows = [r for r in rows if r["arm"] in ("F11", "G2N")]
    descriptive = descriptive_G2N(control_rows)
    t3a = c.task_dir("T3A")
    c.write_csv(t3a/"raw"/"per_seed_endpoints.csv", control_rows, overwrite=True)
    c.write_csv(t3a/"raw"/"paired_differences.csv", descriptive, overwrite=True)
    profile_rows = []
    for arm in ("F11", "G2N"):
        row = c.load_json(evaluation_dir(arm, 41001, "validation")/"capacity_profile.json")
        profile_rows.append({"arm": arm, **row})
    if any(profile_rows[0][key] != profile_rows[1][key] for key in ("parameters_total", "parameters_trainable", "profiled_forward_flops")):
        raise RuntimeError("G2/G2N inference-capacity matching failed")
    c.write_csv(t3a/"raw"/"capacity_profile.csv", profile_rows, overwrite=True)
    match = {partition: all(r["descriptive_within_seed_SD"] for r in descriptive if r["split"] == partition) for partition in splits}
    c.write_task_summary("T3A", {"status": "pretest_complete" if split == "validation" else "completed", "splits": splits,
        "n_seeds": 10, "descriptive_match_by_split": match, "capacity_match": True,
        "rule_is_not_equivalence": True, "figures": "pending_root_publication_rendering"})
    c.append_run_log("T3A", f"All paired endpoint cells verified; descriptive matching {match}; no equivalence conclusion.")
    sensitivity_rows = [r for r in rows if r["arm"] in SENSITIVITY]
    baseline = {(int(r["seed"]), r["split"]): r for r in rows if r["arm"] == "F11"}
    changes = []
    for row in sensitivity_rows:
        parameter, value = SENSITIVITY[row["arm"]]
        base = baseline[(int(row["seed"]), row["split"])]
        for endpoint in ("fresh_collocation_proxy_RMS", "PFFD_std"):
            delta = row[endpoint]-base[endpoint]
            changes.append({"arm": row["arm"], "seed": row["seed"], "split": row["split"],
                "parameter": parameter, "coefficient_value": value, "endpoint": endpoint,
                "value": row[endpoint], "baseline_value": base[endpoint], "paired_difference": delta,
                "relative_difference": delta/base[endpoint] if base[endpoint] != 0 else None})
    t3c = c.task_dir("T3C")
    c.write_csv(t3c/"raw"/"per_seed_sensitivity.csv", sensitivity_rows, overwrite=True)
    c.write_csv(t3c/"raw"/"paired_baseline_changes.csv", changes, overwrite=True)
    c.write_task_summary("T3C", {"status": "pretest_complete" if split == "validation" else "completed", "splits": splits,
        "n_training_cells": 18, "n_endpoint_rows": len(sensitivity_rows), "primary_RMS_definition": "common_baseline_coefficients",
        "all_cells_reported": True, "figures": "pending_root_publication_rendering"})
    c.append_run_log("T3C", "Verified all fixed coefficient sensitivity cells; no cells or seeds replaced.")
    pool_rows = []
    for pool_arm in ("G1", "G2", "D5"):
        pool_summary = verify_current_artifact(c.pool_dir(pool_arm))
        pool_rows.append({key: pool_summary[key] for key in ("arm", "status", "rows", "box_mismatches", "donor_triple_mismatches", "ordered_triples_sha256", "self_pairs", "finished_utc")})
    t3b = c.task_dir("T3B")
    c.write_csv(t3b/"raw"/"pool_integrity.csv", pool_rows, overwrite=True)
    c.write_task_summary("T3B", {"status": "generator_pools_completed_detector_evidence_owned_by_coordinator",
        "n_pools": 3, "pool_rows_each": 7168, "all_boxes_match": True, "all_donor_triples_match": len({r["ordered_triples_sha256"] for r in pool_rows}) == 1})
    for partition in splits:
        controls_dir = c.ARTIFACT_ROOT/"evaluation"/"generator_controls"/partition
        if not (controls_dir/"summary.json").exists():
            raise RuntimeError(f"Missing generator controls: {partition}")
        verify_current_artifact(controls_dir)
        for name in ("blur_summary.csv", "real_real_floor.csv"):
            c.write_csv(t1/"raw"/f"{partition}_{name}", c.read_csv(controls_dir/name), overwrite=True)
    return {"status": "completed", "splits": splits, "endpoint_rows": len(rows), "missing_units": []}


def pretest_evidence():
    """Read-only, fail-closed evidence for the coordinator's global T7 gate."""
    c = common()
    grouped = {key: [] for key in ("T1", "T3A", "T3B", "T3C")}
    units = {key: [] for key in grouped}
    times = {key: [] for key in grouped}
    for arm, seed in planned_runs():
        config = fitted_config(arm, seed)
        task = "T3C" if arm in SENSITIVITY else "T3A" if arm == "G2N" else "T1"
        expected = identity_for(task, config, {"train_manifest": c.manifest_path("train"), "validation_manifest": c.manifest_path("validation")})
        training_dir = c.generator_run_dir(arm, seed)
        training = check_completed(training_dir, expected)
        if not training:
            raise RuntimeError(f"Pretest generator training missing: {arm}/{seed}")
        evaluation = evaluation_dir(arm, seed, "validation")
        validated = verify_current_artifact(evaluation)
        if (validated["arm"], validated["seed"], validated["split"], validated["n_images"], validated["n_diversity_conditions"]) != (arm, seed, "validation", 1024, 128):
            raise RuntimeError("Validation identity/count mismatch")
        if not all(math.isfinite(float(validated[key])) for key in (*PRIMARY, "reconstruction_SSIM")):
            raise RuntimeError("Nonfinite endpoint")
        assigned = ["T1"] if arm in BASE_ARMS else ["T3C"]
        if arm == "G2N":
            assigned.append("T3A")
        for key in assigned:
            units[key].append(f"{arm}/{seed}:train_and_validation")
            grouped[key].extend([str(training_dir/"summary.json"), str(evaluation/"summary.json")])
            times[key].extend([training["finished_utc"], validated["finished_utc"]])
    plan = verified_plan(plan_root()/"pool_plan.json")
    triple_hashes = []
    for arm in ("G1", "G2", "D5"):
        directory = c.pool_dir(arm)
        summary = verify_current_artifact(directory)
        check = verify_pool_rows(c.read_csv(directory/"manifest.csv"), plan["entries"])
        if check["rows"] != 7168 or summary["box_mismatches"] != 0:
            raise RuntimeError("Pool integrity failed")
        triple_hashes.append(summary["ordered_triples_sha256"])
        units["T3B"].append(arm)
        times["T3B"].append(summary["finished_utc"])
        grouped["T3B"].extend([str(directory/"summary.json"), str(directory/"manifest.csv")])
    if len(set(triple_hashes)) != 1:
        raise RuntimeError("Shared donor triples differ across pools")
    controls_dir = c.ARTIFACT_ROOT/"evaluation"/"generator_controls"/"validation"
    control_summary = verify_current_artifact(controls_dir)
    if control_summary["blur_cells"] != 44 or len(c.read_csv(controls_dir/"real_real_floor.csv")) != 10:
        raise RuntimeError("Blur/real-real control grid incomplete")
    units["T1"].append("validation_blur_and_real_real_controls")
    grouped["T1"].append(str(controls_dir/"summary.json"))
    times["T1"].append(control_summary["finished_utc"])
    profiles = [c.load_json(evaluation_dir(arm, 41001, "validation")/"capacity_profile.json") for arm in ("F11", "G2N")]
    if any(profiles[0][key] != profiles[1][key] for key in ("parameters_total", "parameters_trainable", "profiled_forward_flops")):
        raise RuntimeError("Capacity matching failed")
    expected_counts = {"T1": 61, "T3A": 10, "T3B": 3, "T3C": 18}
    result = {}
    for task in grouped:
        if len(units[task]) != expected_counts[task]:
            raise RuntimeError("Pretest unit count mismatch")
        result[task] = {"task_id": task, "status": "pretest_complete", "expected_units": expected_counts[task],
            "completed_units": len(units[task]), "missing_units": [], "integrity_checks_passed": True,
            "unit_definition": "one declared seed fit plus its full validation; T1 adds complete controls, T3B counts pools",
            "last_training_validation_end_utc": max(times[task]), "output_paths": sorted(set(grouped[task]))}
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("train", "evaluate"):
        current = sub.add_parser(name)
        current.add_argument("--arm", choices=[*BASE_ARMS, *SENSITIVITY, "G1", "G2"], required=True)
        current.add_argument("--seed", type=int, required=True)
        if name == "evaluate":
            current.add_argument("--split", choices=["validation", "test"], required=True)
    sub.add_parser("make-plans")
    sub.add_parser("make-pool").add_argument("--arm", choices=["G1", "G2", "D5"], required=True)
    sub.add_parser("controls").add_argument("--split", choices=["validation", "test"], required=True)
    sub.add_parser("summarize").add_argument("--split", choices=["validation", "all"], default="validation")
    sub.add_parser("pretest-evidence")
    args = parser.parse_args(argv)
    if args.command == "train":
        result = train(args.arm, args.seed)
    elif args.command == "evaluate":
        result = evaluate(args.arm, args.seed, args.split)
    elif args.command == "make-plans":
        result = make_plans()
    elif args.command == "make-pool":
        result = make_pool(args.arm)
    elif args.command == "controls":
        result = controls(args.split)
    elif args.command == "summarize":
        result = summarize(args.split)
    else:
        result = pretest_evidence()
    print(json.dumps(result, allow_nan=False), flush=True)
    return result


if __name__ == "__main__":
    main()
