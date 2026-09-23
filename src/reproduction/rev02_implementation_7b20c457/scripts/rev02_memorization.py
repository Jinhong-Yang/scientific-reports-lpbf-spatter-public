"""REV02 canonical-resolution flip-aware audit with immutable calibration.

Commands are train/validation only. Validation posterior means are encoded only
to construct audit positives, never to supply augmentation-pool latents.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from PIL import Image, ImageEnhance
from scipy.fft import dctn
from skimage.metrics import structural_similarity

RULE = "exact_original_or_flip OR feature_min_le_q01_feature OR ssim_max_ge_q99_ssim"
ALPHAS = (0.99, 0.95, 0.90, 0.80)
BASE_CLASSES = ("exact", "noise_sigma2", "crop1_resize", "gamma_brightness", "horizontal_flip")
CLASSES = BASE_CLASSES + tuple(f"interpolation_alpha_{alpha:.2f}" for alpha in ALPHAS)
WEIGHT_SHA256 = "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec"
_BIT_COUNTS = np.asarray([int(i).bit_count() for i in range(256)], dtype=np.uint8)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def canonical_pixels(image: str | Path | Image.Image | np.ndarray) -> np.ndarray:
    if isinstance(image, (str, Path)):
        with Image.open(image) as opened:
            result = opened.convert("L").resize((64, 64), Image.Resampling.BILINEAR)
    elif isinstance(image, Image.Image):
        result = image.convert("L").resize((64, 64), Image.Resampling.BILINEAR)
    else:
        arr = np.asarray(image)
        _require(arr.dtype == np.uint8, "Canonical array input must be uint8, not unspecified float units")
        result = Image.fromarray(arr).convert("L").resize((64, 64), Image.Resampling.BILINEAR)
    return np.asarray(result, dtype=np.uint8).copy()


def pixel_hash(pixels: np.ndarray) -> str:
    arr = np.ascontiguousarray(pixels, dtype=np.uint8)
    _require(arr.shape == (64, 64), "Hash requires canonical 64x64 image")
    return hashlib.sha256(b"uint8:L:64:64:" + arr.tobytes()).hexdigest()


def phash(pixels: np.ndarray) -> np.uint64:
    small = Image.fromarray(pixels).resize((32, 32), Image.Resampling.LANCZOS)
    coefficient = dctn(np.asarray(small, dtype=np.float32), axes=(0, 1), norm="ortho")[:8, :8]
    bits = coefficient > np.median(coefficient[1:, 1:])
    bits[0, 0] = False
    value = 0
    for bit in bits.flat:
        value = (value << 1) | int(bit)
    return np.uint64(value)


def hamming_matrix(queries: np.ndarray, references: np.ndarray) -> np.ndarray:
    xor = np.bitwise_xor(np.asarray(queries, dtype=np.uint64)[:, None], np.asarray(references, dtype=np.uint64)[None, :])
    return _BIT_COUNTS[xor.view(np.uint8).reshape(xor.shape + (8,))].sum(axis=2, dtype=np.uint16)


def cosine_distances(query: np.ndarray, reference: np.ndarray) -> np.ndarray:
    distance = 1.0 - np.asarray(query) @ np.asarray(reference).T
    _require(bool(np.isfinite(distance).all()), "Nonfinite feature distance")
    _require(bool(np.all(distance >= -1e-6) and np.all(distance <= 2 + 1e-6)), "Cosine distance outside frozen roundoff tolerance")
    return np.clip(distance, 0, 2)


def ssim(left: np.ndarray, right: np.ndarray) -> float:
    _require(left.shape == right.shape == (64, 64), "SSIM requires matching canonical images")
    return float(structural_similarity(left.astype(np.float32) / 255, right.astype(np.float32) / 255,
                                     data_range=1.0, win_size=7, gaussian_weights=False,
                                     use_sample_covariance=True, channel_axis=None))


def apply_rule(row: Mapping[str, Any], lock: Mapping[str, Any]) -> dict[str, bool]:
    _require(lock.get("decision_rule") == RULE, "Unexpected audit rule")
    thresholds = lock["thresholds"]
    exact_value = row["exact_original_or_flip"]
    exact = exact_value if isinstance(exact_value, bool) else str(exact_value).lower() in ("true", "1")
    feature = float(row["feature_min"]) <= float(thresholds["feature_q01"])
    structural = float(row["ssim_max"]) >= float(thresholds["ssim_q99"])
    return {"flag_exact": exact, "flag_feature": feature, "flag_ssim": structural,
            "suspicious": bool(exact or feature or structural)}


def calibrate_thresholds(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    _require(len(rows) > 0, "Empty calibration rows")
    _require(all(r["query_split"] == "validation" and r["reference_split"] == "train" for r in rows), "Calibration restricted to validation-to-train")
    feature = np.asarray([r["feature_min"] for r in rows], dtype=np.float64)
    structural = np.asarray([r["ssim_max"] for r in rows], dtype=np.float64)
    hamming = np.asarray([r["phash_min"] for r in rows], dtype=np.float64)
    _require(bool(np.isfinite(feature).all() and np.all((feature >= 0) & (feature <= 2))), "Invalid calibrated feature distances")
    _require(bool(np.isfinite(structural).all() and np.all((structural >= -1) & (structural <= 1))), "Invalid calibrated SSIM")
    _require(bool(np.isfinite(hamming).all() and np.all((hamming >= 0) & (hamming <= 64)) and np.all(hamming == np.floor(hamming))), "Invalid pHash distance")
    return {"status": "FROZEN_VALIDATION_CALIBRATION", "decision_rule": RULE,
            "thresholds": {"feature_q01": float(np.quantile(feature, 0.01, method="linear")),
                           "ssim_q99": float(np.quantile(structural, 0.99, method="linear"))},
            "phash_q01_diagnostic_only": int(np.quantile(hamming, 0.01, method="lower")),
            "quantile_methods": {"feature": "linear", "ssim": "linear", "phash_diagnostic": "lower"},
            "test_rows_used": 0, "calibration_rows": len(rows)}


def wilson_interval(detected: int, n: int) -> dict[str, float | int]:
    _require(isinstance(detected, (int, np.integer)) and isinstance(n, (int, np.integer)) and 0 <= detected <= n and n > 0, "Invalid Wilson counts")
    z = 1.959963984540054
    proportion = detected / n
    denominator = 1 + z * z / n
    center = (proportion + z * z / (2 * n)) / denominator
    half = z * np.sqrt(proportion * (1 - proportion) / n + z * z / (4 * n * n)) / denominator
    return {"detected_count": int(detected), "total_count": int(n), "observed_TPR": proportion,
            "Wilson_95_CI_low": 0.0 if detected == 0 else max(0.0, float(center - half)),
            "Wilson_95_CI_high": 1.0 if detected == n else min(1.0, float(center + half))}


def summarize_controls(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    _require(len(rows) == 450 and set(str(r["control_class"]) for r in rows) == set(CLASSES), "Exactly nine classes/450 controls required")
    classes = {}
    for label in CLASSES:
        subset = [r for r in rows if r["control_class"] == label]
        _require(len(subset) == 50, f"Expected 50 controls for {label}")
        count = sum(str(r["suspicious"]).lower() in ("true", "1") for r in subset)
        classes[label] = wilson_interval(count, 50)
    demoted = any(classes[f"interpolation_alpha_{a:.2f}"]["observed_TPR"] < 0.5 for a in (0.95, 0.90, 0.80))
    return {"classes": classes, "pooled": wilson_interval(sum(v["detected_count"] for v in classes.values()), 450),
            "audit_contribution_removed": demoted,
            "zero_flag_placement": "Limitations_audit_design_limitation" if demoted else "calibrated_audit_result_not_proof_of_absence",
            "pooled_interval_caveat": "descriptive_repeated_donors_across_alpha_not_independent_trials"}


def make_control_plan(rows: Sequence[Mapping[str, Any]], conditions: np.ndarray, *, seed: int = 61004) -> list[dict[str, Any]]:
    _require(len(rows) >= 50 and conditions.shape == (len(rows), 13), "At least 50 validation rows and 13-D conditions required")
    _require(all(r.get("split") == "validation" and r.get("source_kind") == "real" for r in rows), "Validation real donor rows only")
    order = sorted(range(len(rows)), key=lambda i: str(rows[i]["sample_id"]))
    _require(len({str(r["sample_id"]) for r in rows}) == len(rows), "Duplicate donor IDs")
    anchor_rng, donor_rng, epsilon_rng = [np.random.default_rng(np.random.SeedSequence([seed, index])) for index in range(3)]
    selected = sorted(anchor_rng.choice(np.asarray(order), size=50, replace=False).tolist(), key=lambda i: str(rows[i]["sample_id"]))
    plan = []
    dims = [0, 1, 2, 7, 8, 9, 10]
    for rank, a in enumerate(selected):
        eligible = [b for b in order if b != a and np.argmax(conditions[b, 3:5]) == np.argmax(conditions[a, 3:5]) and np.argmax(conditions[b, 5:7]) == np.argmax(conditions[a, 5:7])]
        _require(bool(eligible), f"No eligible distinct donor for {rows[a]['sample_id']}")
        candidate = sorted(eligible, key=lambda b: (float(np.mean((conditions[b, dims].astype(np.float64) - conditions[a, dims]) ** 2)), str(rows[b]["sample_id"])))[:64]
        b = candidate[int(donor_rng.integers(0, len(candidate)))]
        epsilon = epsilon_rng.standard_normal(16).tolist()
        plan.append({"control_index": rank, "donor_a_id": str(rows[a]["sample_id"]), "donor_b_id": str(rows[b]["sample_id"]),
                     "donor_a_path": str(rows[a]["image_path"]), "donor_b_path": str(rows[b]["image_path"]),
                     "condition": conditions[a].astype(float).tolist(), "standard_normal_epsilon": epsilon,
                     "epsilon_sha256": hashlib.sha256(np.asarray(epsilon, dtype=np.float64).tobytes()).hexdigest()})
    return plan


class FeatureEngine:
    def __init__(self, weight_path: Path, device: str | None = None):
        import torch
        from torchvision.models import ResNet18_Weights, resnet18
        _require(weight_path.is_file(), "Frozen ResNet18 weights must already be staged; no implicit network download")
        actual = hashlib.sha256(weight_path.read_bytes()).hexdigest()
        _require(actual == WEIGHT_SHA256, "ResNet18 weight hash mismatch")
        self.weight_sha256 = actual
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = resnet18(weights=None)
        self.model.load_state_dict(torch.load(weight_path, map_location="cpu", weights_only=True))
        self.model.fc = torch.nn.Identity()
        self.model.eval().to(self.device)
        self.transform = ResNet18_Weights.IMAGENET1K_V1.transforms()

    def embeddings(self, arrays: Sequence[np.ndarray]) -> np.ndarray:
        import torch
        output = []
        with torch.no_grad():
            for start in range(0, len(arrays), 32):
                batch = torch.stack([self.transform(Image.fromarray(image).convert("RGB")) for image in arrays[start:start + 32]])
                output.append(self.model(batch.to(self.device)).cpu().numpy())
        result = np.concatenate(output)
        return result / np.maximum(np.linalg.norm(result, axis=1, keepdims=True), 1e-12)


def make_bank(rows: Sequence[Mapping[str, Any]], resolve: Callable, engine: Any) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda r: str(r["sample_id"]))
    _require(len({str(r["sample_id"]) for r in ordered}) == len(ordered) and bool(ordered), "Nonempty unique reference IDs required")
    pixels = np.stack([canonical_pixels(resolve(r["image_path"])) for r in ordered])
    return {"ids": np.asarray([str(r["sample_id"]) for r in ordered]),
            "paths": np.asarray([str(r["image_path"]) for r in ordered]),
            "pixels": pixels, "features": engine.embeddings(pixels),
            "phash": np.asarray([phash(p) for p in pixels], dtype=np.uint64),
            "pixel_hashes": np.asarray([pixel_hash(p) for p in pixels])}


def measure_queries(rows: Sequence[Mapping[str, Any]], bank: Mapping[str, Any], resolve: Callable, engine: Any,
                    *, query_split: str, reference_split: str, progress: Callable[[int, int], None] | None = None) -> list[dict[str, Any]]:
    _require(len(rows) > 0, "Empty audit queries")
    _require(list(bank["ids"]) == sorted(bank["ids"]), "Reference bank must use stable-ID ordering")
    lookup = {}
    for i, digest in enumerate(bank["pixel_hashes"]):
        lookup.setdefault(str(digest), i)
    path_to_index = {str(p).replace("\\", "/").lower(): k for k, p in enumerate(bank["paths"])}
    output = []
    for start in range(0, len(rows), 32):
        block = rows[start:start + 32]
        canonical = [canonical_pixels(resolve(r["image_path"])) for r in block]
        oriented = [candidate for pixels in canonical for candidate in (pixels, pixels[:, ::-1].copy())]
        query_feature = engine.embeddings(oriented)
        distance = cosine_distances(query_feature, bank["features"])
        nearest = np.argmin(distance, axis=1)
        hashes = np.asarray([phash(p) for p in oriented], dtype=np.uint64)
        hd = hamming_matrix(hashes, bank["phash"])
        hn = np.argmin(hd, axis=1)
        for i, row in enumerate(block):
            q0, q1 = 2 * i, 2 * i + 1
            feature_choice = min((q0, q1), key=lambda q: (float(distance[q, nearest[q]]), str(bank["ids"][nearest[q]]), q % 2))
            structural = {q: ssim(oriented[q], bank["pixels"][nearest[q]]) for q in (q0, q1)}
            ssim_choice = min((q0, q1), key=lambda q: (-structural[q], str(bank["ids"][nearest[q]]), q % 2))
            phash_choice = min((q0, q1), key=lambda q: (int(hd[q, hn[q]]), str(bank["ids"][hn[q]]), q % 2))
            exact_candidates = [(lookup[pixel_hash(oriented[q])], q) for q in (q0, q1) if pixel_hash(oriented[q]) in lookup]
            exact_pair = min(exact_candidates, key=lambda pair: (str(bank["ids"][pair[0]]), pair[1] % 2)) if exact_candidates else None
            record = {"query_id": str(row["sample_id"]), "query_path": str(row["image_path"]),
                      "query_split": query_split, "reference_split": reference_split,
                      "feature_min": float(distance[feature_choice, nearest[feature_choice]]),
                      "ssim_max": structural[ssim_choice], "phash_min": int(hd[phash_choice, hn[phash_choice]]),
                      "exact_original_or_flip": exact_pair is not None}
            for metric, q, ri in (("feature", feature_choice, int(nearest[feature_choice])),
                                  ("ssim", ssim_choice, int(nearest[ssim_choice])),
                                  ("phash", phash_choice, int(hn[phash_choice]))):
                record.update({f"{metric}_orientation": "original" if q % 2 == 0 else "horizontal_flip",
                               f"{metric}_reference_id": str(bank["ids"][ri]), f"{metric}_reference_path": str(bank["paths"][ri])})
            record["exact_orientation"] = ("original" if exact_pair[1] % 2 == 0 else "horizontal_flip") if exact_pair else ""
            record["exact_reference_id"] = str(bank["ids"][exact_pair[0]]) if exact_pair else ""
            record["exact_reference_path"] = str(bank["paths"][exact_pair[0]]) if exact_pair else ""
            # Donor distances are descriptive and never alter the decision rule.
            donor_distances = []
            for donor_key in ("donor_a", "donor_b"):
                donor = str(row.get(donor_key, "")).replace("\\", "/").lower()
                if donor in path_to_index:
                    donor_distances.append(float(distance[[q0, q1], path_to_index[donor]].min()))
            record["donor_min_feature_distance"] = min(donor_distances) if donor_distances else None
            output.append(record)
        if progress:
            progress(min(start + len(block), len(rows)), len(rows))
    return output


def _common():
    import rev02_common
    return rev02_common


def _progress(done: int, total: int) -> None:
    if done % 256 == 0 or done == total:
        print(json.dumps({"audit_progress": done, "total": total}), flush=True)


def _weight_path(c) -> Path:
    return c.ARTIFACT_ROOT / "torch_cache" / "hub" / "checkpoints" / "resnet18-f37072fd.pth"


def _input_files(c, rows: Sequence[Mapping[str, Any]], prefix: str) -> dict[str, Path]:
    result = {}
    for row in rows:
        path = c.resolve_path(row["image_path"])
        expected = row.get("image_sha256") or row.get("raw_image_sha256")
        _require(isinstance(expected, str) and len(expected) == 64, "Frozen source image SHA256 missing")
        _require(c.sha256_file(path) == expected, f"Frozen image hash mismatch for {row['sample_id']}")
        result[f"{prefix}_image_{row['sample_id']}"] = path
    return result


def _verify_stage(c, directory: Path) -> dict[str, Any]:
    receipt = c.load_json(directory / "completed_receipt.json")
    c.verify_implementation_lock()
    provenance = receipt["provenance"]
    _require(bool(provenance.get("input_hashes")) and bool(receipt.get("outputs")), "Empty audit receipt input/output hash maps")
    _require(c.sha256_file(c.REV_ROOT / "IMPLEMENTATION_LOCK.json") == provenance["implementation_lock_sha256"], "Changed implementation lock")
    for name, digest in provenance["environment_hashes"].items():
        _require(c.sha256_file(c.REV_ROOT / "env" / name) == digest, "Changed environment lock")
    for value in provenance["input_hashes"].values():
        _require(c.sha256_file(Path(value["path"])) == value["sha256"], "Changed audit-stage input")
    for relative, digest in receipt["outputs"].items():
        _require(c.sha256_file(directory / relative) == digest, f"Changed audit output {relative}")
    return receipt


def _run_stage(stage: str, inputs: Mapping[str, Path], action: Callable[[Any, Path, dict], dict]) -> dict:
    c = _common()
    c.load_protocol("T4")
    config = {"stage": stage}
    provenance = c.run_provenance("T4", config, dict(inputs))
    out = c.task_dir("T4") / stage
    if (out / "completed_receipt.json").exists():
        receipt = _verify_stage(c, out)
        _require(receipt["provenance"] == provenance, "Audit-stage provenance differs from completed run")
        return c.load_json(out / "summary.json")
    _require(not out.exists(), f"Incomplete audit stage {stage} preserved; documented recovery required")
    _require(not (c.REV_ROOT / "T7_UNLOCK.json").exists(), "New audit stages are forbidden after T7 unlock")
    out.mkdir(parents=True)
    started_at = c.utc_now()
    c.write_json(out / "provenance.json", provenance)
    c.record_event("audit_stage_started", {"stage": stage, "inputs": len(inputs)})
    try:
        summary = action(c, out, provenance)
        summary["finished_utc"] = c.utc_now()
        c.write_json(out / "summary.json", summary)
        outputs = {p.relative_to(out).as_posix(): c.sha256_file(p) for p in sorted(out.rglob("*")) if p.is_file()}
        c.write_json(out / "completed_receipt.json", {"stage": stage, "started_at": started_at, "completed_at": c.utc_now(), "provenance": provenance, "outputs": outputs})
        c.append_run_log("T4", f"Completed {stage}; immutable receipt and {len(outputs)} artifact hashes recorded.")
        c.record_event("audit_stage_completed", {"stage": stage, "receipt_sha256": c.sha256_file(out / "completed_receipt.json")})
        return summary
    except Exception as exc:
        failure = {"stage": stage, "timestamp": c.utc_now(), "error_type": type(exc).__name__, "error": str(exc), "partial_artifacts_preserved": True}
        c.write_json(out / "failure.json", failure)
        c.record_event("audit_stage_failed", failure)
        c.append_run_log("T4", f"Failed {stage}: {type(exc).__name__}: {exc}; partial outputs preserved.")
        raise


def _save_bank(path: Path, bank: Mapping[str, Any]) -> None:
    with path.open("xb") as handle:
        np.savez_compressed(handle, **bank)


def _load_bank(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name] for name in data.files}


def calibrate() -> dict:
    c = _common()
    c.load_protocol("T4")
    train, validation = c.load_rows("train"), c.load_rows("validation")
    _require(len(train) == 3584 and len(validation) == 1024, "Frozen calibration population must be 3584 train / 1024 validation images")
    inputs = {"train_manifest": c.manifest_path("train"), "validation_manifest": c.manifest_path("validation"), "feature_weights": _weight_path(c),
              **_input_files(c, train, "train"), **_input_files(c, validation, "validation")}
    def action(c, out, provenance):
        engine = FeatureEngine(_weight_path(c))
        bank = make_bank(train, c.resolve_path, engine)
        validation_bank = make_bank(validation, c.resolve_path, engine)
        rows = measure_queries(validation, bank, c.resolve_path, engine, query_split="validation", reference_split="train", progress=_progress)
        c.write_csv(out / "validation_to_train_continuous.csv", rows)
        _save_bank(out / "train_bank.npz", bank)
        _save_bank(out / "validation_bank.npz", validation_bank)
        lock = calibrate_thresholds(rows)
        lock.update(locked_at=c.utc_now(), metric_engine=c.load_protocol("T4")["metric_engine"],
                    train_count=len(train), validation_count=len(validation), feature_weight_sha256=engine.weight_sha256,
                    calibration_csv_sha256=c.sha256_file(out / "validation_to_train_continuous.csv"),
                    train_bank_sha256=c.sha256_file(out / "train_bank.npz"), validation_bank_sha256=c.sha256_file(out / "validation_bank.npz"),
                    provenance_sha256=c.sha256_file(out / "provenance.json"))
        c.write_json(out / "threshold_lock.json", lock)
        count = sum(apply_rule(r, lock)["suspicious"] for r in rows)
        return {"status": "calibration_locked", "threshold_lock_sha256": c.sha256_file(out / "threshold_lock.json"),
                "calibration_flag_count": count, "calibration_count": len(rows), "calibration_flag_rate": count / len(rows),
                "interpretation": "in_sample_validation_operating_rate_not_independent_FPR", "phash_q01_diagnostic_only": lock["phash_q01_diagnostic_only"]}
    return _run_stage("calibration", inputs, action)


def _calibration(c):
    path = c.task_dir("T4") / "calibration"
    _verify_stage(c, path)
    lock = c.load_json(path / "threshold_lock.json")
    _require(lock.get("status") == "FROZEN_VALIDATION_CALIBRATION" and lock.get("test_rows_used") == 0, "Invalid calibration lock")
    return path, lock


def freeze_controls() -> dict:
    c = _common()
    calibration, _ = _calibration(c)
    validation = c.load_rows("validation")
    checkpoint = c.generator_run_dir("F11", 41001) / "best.pt"
    import rev02_generator as generator
    fitted_identity = generator.identity_for("T1", generator.fitted_config("F11", 41001),
        {"train_manifest": c.manifest_path("train"), "validation_manifest": c.manifest_path("validation")})
    _require(generator.check_completed(checkpoint.parent, fitted_identity) is not None, "Canonical G2 checkpoint is not hash-verified complete")
    inputs = {"validation_manifest": c.manifest_path("validation"), "checkpoint": checkpoint,
              "threshold_lock": calibration / "threshold_lock.json", **_input_files(c, validation, "validation")}
    def action(c, out, provenance):
        from metal_spatter_pinn.data import condition_from_manifest
        conditions = np.stack([condition_from_manifest(r).vector for r in validation])
        plan = make_control_plan(validation, conditions)
        c.write_json(out / "control_plan.json", {"seed": 61004, "checkpoint_sha256": c.sha256_file(checkpoint), "checkpoint_arm": "F11_G2", "checkpoint_seed": 41001,
                                                 "frozen_at": c.utc_now(), "rows": plan})
        c.write_csv(out / "control_donor_plan.csv", [{k: v for k, v in row.items() if k not in ("condition", "standard_normal_epsilon")} for row in plan])
        c.write_csv(out / "control_donor_noise_vectors.csv", [{"control_index": row["control_index"], **{f"epsilon_{i}": value for i, value in enumerate(row["standard_normal_epsilon"])}} for row in plan])
        return {"status": "control_plan_frozen_before_images", "donor_pairs": 50, "latent_dimension": 16,
                "control_plan_sha256": c.sha256_file(out / "control_plan.json")}
    return _run_stage("control_plan", inputs, action)


def _base_control_images(anchor: np.ndarray, noise_rng: np.random.Generator) -> dict[str, np.ndarray]:
    image = Image.fromarray(anchor)
    noise = np.clip(np.rint(anchor.astype(np.float32) + noise_rng.normal(0, 2, anchor.shape)), 0, 255).astype(np.uint8)
    gamma = np.clip(np.rint(255 * (anchor.astype(np.float32) / 255) ** 1.05), 0, 255).astype(np.uint8)
    return {"exact": anchor.copy(), "noise_sigma2": noise,
            "crop1_resize": np.asarray(image.crop((1, 1, 63, 63)).resize((64, 64), Image.Resampling.BILINEAR)),
            "gamma_brightness": np.asarray(ImageEnhance.Brightness(Image.fromarray(gamma)).enhance(1.02)),
            "horizontal_flip": anchor[:, ::-1].copy()}


def controls() -> dict:
    c = _common()
    calibration, lock = _calibration(c)
    freeze_controls()
    plan_dir = c.task_dir("T4") / "control_plan"
    _verify_stage(c, plan_dir)
    plan = c.load_json(plan_dir / "control_plan.json")["rows"]
    checkpoint = c.generator_run_dir("F11", 41001) / "best.pt"
    validation = c.load_rows("validation")
    inputs = {"plan": plan_dir / "control_plan.json", "threshold_lock": calibration / "threshold_lock.json", "checkpoint": checkpoint,
              "validation_bank": calibration / "validation_bank.npz", "feature_weights": _weight_path(c),
              "validation_manifest": c.manifest_path("validation"), **_input_files(c, validation, "validation")}
    def action(c, out, provenance):
        import torch
        from metal_spatter_pinn.data import condition_from_manifest
        from metal_spatter_pinn.inference import render_field
        from rev02_generator import load_fitted
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, payload = load_fitted("F11", 41001, device)
        _require(model.config.latent_dim == 16, "Canonical control model must have 16-D latent")
        _require(payload.get("arm") == "F11" and payload.get("seed") == 41001, "Control checkpoint must be frozen F11/G2 seed 41001")
        by_id = {r["sample_id"]: r for r in validation}
        needed = sorted({row[key] for row in plan for key in ("donor_a_id", "donor_b_id")})
        posterior = {}
        with torch.no_grad():
            for start in range(0, len(needed), 32):
                ids = needed[start:start + 32]
                images = torch.from_numpy(np.stack([canonical_pixels(c.resolve_path(by_id[i]["image_path"])) for i in ids]).astype(np.float32) / 255)[:, None].to(device)
                conditions = torch.from_numpy(np.stack([condition_from_manifest(by_id[i]).vector for i in ids])).to(device)
                mu, _ = model.encode(images, conditions)
                posterior.update({identifier: mu[index] for index, identifier in enumerate(ids)})
        noise_rng = np.random.default_rng(np.random.SeedSequence([61004, 3]))
        (out / "images").mkdir()
        queries = []
        for row in plan:
            anchor = canonical_pixels(c.resolve_path(row["donor_a_path"]))
            images = _base_control_images(anchor, noise_rng)
            condition = torch.tensor(row["condition"], dtype=torch.float32, device=device)[None]
            epsilon = torch.tensor(row["standard_normal_epsilon"], dtype=torch.float32, device=device)
            with torch.no_grad():
                for alpha in ALPHAS:
                    latent = alpha * posterior[row["donor_a_id"]] + (1 - alpha) * posterior[row["donor_b_id"]] + 0.08 * epsilon
                    generated = render_field(model, condition, latent[None], 64)[0, 0].cpu().numpy()
                    _require(bool(np.isfinite(generated).all()), "Nonfinite interpolation control")
                    images[f"interpolation_alpha_{alpha:.2f}"] = np.clip(np.rint(255 * generated), 0, 255).astype(np.uint8)
            for label, pixels in images.items():
                query_id = f"{label}_{int(row['control_index']):03d}"
                path = out / "images" / f"{query_id}.bmp"
                _require(not path.exists(), "Refusing to overwrite control image")
                Image.fromarray(pixels).save(path)
                queries.append({"sample_id": query_id, "image_path": str(path), "control_class": label,
                                "donor_a_id": row["donor_a_id"], "donor_b_id": row["donor_b_id"],
                                "donor_a": row["donor_a_path"], "donor_b": row["donor_b_path"], "epsilon_sha256": row["epsilon_sha256"]})
        del model, posterior
        if device.type == "cuda":
            torch.cuda.empty_cache()
        c.write_csv(out / "control_image_manifest.csv", queries)
        engine = FeatureEngine(_weight_path(c))
        bank = _load_bank(calibration / "validation_bank.npz")
        continuous = measure_queries(queries, bank, c.resolve_path, engine, query_split="control", reference_split="validation", progress=_progress)
        query_index = {q["sample_id"]: q for q in queries}
        records = [{**r, "control_class": query_index[r["query_id"]]["control_class"],
                    "donor_a_id": query_index[r["query_id"]]["donor_a_id"], "donor_b_id": query_index[r["query_id"]]["donor_b_id"],
                    **apply_rule(r, lock), "threshold_lock_sha256": c.sha256_file(calibration / "threshold_lock.json")} for r in continuous]
        c.write_csv(out / "per_control.csv", records)
        summary = summarize_controls(records)
        c.append_run_log("T4", f"Frozen alpha criterion decision: audit_contribution_removed={summary['audit_contribution_removed']}; zero_flag_placement={summary['zero_flag_placement']}.")
        return {"status": "controls_complete", **summary, "threshold_locked_at": lock["locked_at"],
                "control_scope": "canonical_64px_donor_derived_latent_interpolation", "checkpoint_sha256": c.sha256_file(checkpoint)}
    return _run_stage("controls", inputs, action)


def audit_pool(arm: str) -> dict:
    c = _common()
    _require(arm in ("G1", "G2", "D5"), "Unknown audit pool")
    calibration, lock = _calibration(c)
    manifest = c.pool_dir(arm) / "manifest.csv"
    import rev02_generator as generator
    plan_path = generator.plan_root() / "pool_plan.json"
    plan = generator.verified_plan(plan_path)
    source_inputs = {"plan": plan_path, "train_manifest": c.manifest_path("train")}
    if arm != "D5":
        for seed in c.load_protocol("T1")["fresh_training"]["seeds"]:
            source_inputs[f"checkpoint_{seed}"] = c.generator_run_dir(generator.canonical_arm(arm), seed) / "best.pt"
    identity = generator.identity_for("T3B", {"kind": "pool", "arm": arm, "schema": 1}, source_inputs)
    _require(generator.check_completed(manifest.parent, identity) is not None, "Synthetic pool lacks valid completion identity")
    rows = c.read_csv(manifest)
    generator.verify_pool_rows(rows, plan["entries"])
    _require(len(rows) == 7168, "Every pool must contain exactly 7168 images")
    _require(all(r.get("source_kind") == "synthetic" for r in rows), "Pool provenance must explicitly be synthetic")
    normalized = [{**r, "sample_id": r.get("sample_id") or r.get("synthetic_id")} for r in rows]
    _require(all(r["sample_id"] for r in normalized) and len({r["sample_id"] for r in normalized}) == 7168, "Unique synthetic IDs required")
    train = c.load_rows("train")
    train_paths = {str(c.resolve_path(r["image_path"])).replace("\\", "/").lower() for r in train}
    for r in normalized:
        _require(all(r.get(key) and str(c.resolve_path(r[key])).replace("\\", "/").lower() in train_paths for key in ("donor_a", "donor_b")), "Nontraining pool donor")
    inputs = {"pool_manifest": manifest, "pool_completion": manifest.parent / "summary.json", "pool_plan": plan_path,
              "threshold_lock": calibration / "threshold_lock.json", "train_bank": calibration / "train_bank.npz",
              "feature_weights": _weight_path(c), **_input_files(c, normalized, arm)}
    def action(c, out, provenance):
        engine = FeatureEngine(_weight_path(c))
        bank = _load_bank(calibration / "train_bank.npz")
        continuous = measure_queries(normalized, bank, c.resolve_path, engine, query_split="synthetic", reference_split="train", progress=_progress)
        records = [{"pool": arm, **r, **apply_rule(r, lock), "threshold_lock_sha256": c.sha256_file(calibration / "threshold_lock.json")} for r in continuous]
        c.write_csv(out / "per_synthetic_audit.csv", records)
        flagged = sum(r["suspicious"] for r in records)
        return {"status": "pool_audit_complete", "pool": arm, "n": len(records), "flagged_count": flagged,
                "flagged_rate": flagged / len(records), "exact_original_or_flip_count": sum(r["flag_exact"] for r in records),
                "images_excluded": 0, "threshold_locked_at": lock["locked_at"]}
    return _run_stage(f"pool_{arm}", inputs, action)


def summarize() -> dict:
    c = _common()
    stages = ("calibration", "control_plan", "controls", "pool_G1", "pool_G2", "pool_D5")
    task = c.task_dir("T4")
    inputs = {}
    for stage in stages:
        _verify_stage(c, task / stage)
        inputs[stage] = task / stage / "completed_receipt.json"
    def action(c, out, provenance):
        calibration = c.load_json(task / "calibration" / "summary.json")
        control = c.load_json(task / "controls" / "summary.json")
        pools = [c.load_json(task / f"pool_{arm}" / "summary.json") for arm in ("G1", "G2", "D5")]
        all_pool_rows = [row for arm in ("G1", "G2", "D5") for row in c.read_csv(task / f"pool_{arm}" / "per_synthetic_audit.csv")]
        c.write_csv(out / "per_synthetic_audit.csv", all_pool_rows)
        aliases = {
            "threshold_lock.json": task / "calibration" / "threshold_lock.json",
            "raw/validation_to_train_continuous.csv": task / "calibration" / "validation_to_train_continuous.csv",
            "raw/control_donor_plan.csv": task / "control_plan" / "control_donor_plan.csv",
            "raw/control_donor_noise_vectors.csv": task / "control_plan" / "control_donor_noise_vectors.csv",
            "raw/per_control.csv": task / "controls" / "per_control.csv",
            "raw/per_synthetic_audit.csv": out / "per_synthetic_audit.csv",
        }
        for relative, source in aliases.items():
            target = task / relative
            if target.exists():
                _require(c.sha256_file(target) == c.sha256_file(source), "Changed public audit alias")
            else:
                c.atomic_text(target, source.read_text(encoding="utf-8"))
        return {"task_id": "T4", "status": "pretest_complete", "calibration": calibration, "controls": control,
                "pools": pools, "total_audited_pool_images": len(all_pool_rows), "test_rows_used": 0,
                "artifact_index": {relative: {"path": str(task / relative), "sha256": c.sha256_file(task / relative)} for relative in aliases},
                "figures_status": "pending_root_manuscript_renderer"}
    result = _run_stage("final_summary", inputs, action)
    c.write_task_summary("T4", result)
    return result


def pretest_evidence() -> dict:
    """Verify only public train/validation audit stages and immutable receipts."""
    from datetime import datetime
    c = _common()
    task = c.task_dir("T4")
    stages = ("calibration", "control_plan", "controls", "pool_G1", "pool_G2", "pool_D5", "final_summary")
    receipts = {stage: _verify_stage(c, task / stage) for stage in stages}
    summary = c.load_json(task / "final_summary" / "summary.json")
    _require(summary["status"] == "pretest_complete" and summary["test_rows_used"] == 0, "Audit is not pretest complete")
    _require(summary["calibration"]["calibration_count"] == 1024, "Incomplete validation calibration population")
    for artifact in summary["artifact_index"].values():
        _require(c.sha256_file(Path(artifact["path"])) == artifact["sha256"], "Changed public audit artifact")
    lock = c.load_json(task / "calibration" / "threshold_lock.json")
    locked_at = datetime.fromisoformat(lock["locked_at"])
    _require(locked_at <= datetime.fromisoformat(receipts["calibration"]["completed_at"]), "Invalid threshold chronology")
    for stage in ("controls", "pool_G1", "pool_G2", "pool_D5"):
        _require(locked_at < datetime.fromisoformat(receipts[stage]["started_at"]), "Audit/control assessment precedes threshold lock")
    per_control = c.read_csv(task / "controls" / "per_control.csv")
    checked_controls = summarize_controls(per_control)
    _require(checked_controls["audit_contribution_removed"] == summary["controls"]["audit_contribution_removed"], "Control decision changed")
    plan = c.load_json(task / "control_plan" / "control_plan.json")
    _require(datetime.fromisoformat(plan["frozen_at"]) < datetime.fromisoformat(receipts["controls"]["started_at"]), "Control plan was not frozen before image generation")
    _require(len(plan["rows"]) == 50 and all(r["donor_a_id"] != r["donor_b_id"] and len(r["standard_normal_epsilon"]) == 16 for r in plan["rows"]), "Incomplete donor/noise plan")
    for arm in ("G1", "G2", "D5"):
        pool_rows = c.read_csv(task / f"pool_{arm}" / "per_synthetic_audit.csv")
        _require(len(pool_rows) == 7168 and len({r["query_id"] for r in pool_rows}) == 7168, f"Incomplete {arm} audit")
        threshold_sha = c.sha256_file(task / "calibration" / "threshold_lock.json")
        _require(all(r["threshold_lock_sha256"] == threshold_sha for r in pool_rows), "Threshold changed within pool")
    return {"T4": {"task_id": "T4", "status": "pretest_complete", "expected_units": 21954, "completed_units": 21954,
                   "unit_definition": "450 controls plus 3 times 7168 pool queries", "missing_units": [],
                   "integrity_checks_passed": True,
                   "last_training_validation_end_utc": max(r["completed_at"] for r in receipts.values()),
                   "output_paths": sorted(set([str(task / stage / "completed_receipt.json") for stage in stages] +
                       [str(task / stage / relative) for stage, receipt in receipts.items() for relative in receipt["outputs"]] +
                       [x["path"] for x in summary["artifact_index"].values()]))}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("calibrate", "freeze-controls", "controls", "pools", "summarize"):
        sub.add_parser(command)
    one = sub.add_parser("audit-pool")
    one.add_argument("--arm", choices=("G1", "G2", "D5"), required=True)
    args = parser.parse_args(argv)
    if args.command == "calibrate":
        result = calibrate()
    elif args.command == "freeze-controls":
        result = freeze_controls()
    elif args.command == "controls":
        result = controls()
    elif args.command == "pools":
        result = {arm: audit_pool(arm) for arm in ("G1", "G2", "D5")}
    elif args.command == "audit-pool":
        result = audit_pool(args.arm)
    else:
        result = summarize()
    print(json.dumps(result, indent=2, allow_nan=False))
    return result


if __name__ == "__main__":
    main()
