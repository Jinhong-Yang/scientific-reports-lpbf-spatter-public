"""One-time REV02 metadata steward and environment/source freeze commands."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import random
import shutil
import subprocess
import sys

from rev02_common import (
    ARTIFACT_ROOT, PROJECT_ROOT, PROTOCOL_COMMIT, REV_ROOT, append_run_log,
    atomic_text, load_json, load_protocol, read_csv, record_event, sha256_file,
    steward_metadata_access, task_dir, utc_now, verify_event_chain,
    verify_protocols, write_csv, write_json, write_task_summary,
)

GIT = Path("C:/Users/WIN/.cache/codex-runtimes/codex-primary-runtime/dependencies/native/git/cmd/git.exe")


def git(*args: str) -> str:
    # The sandbox and approved D-drive runner use different Windows accounts.
    # Trust only this newly-created, scoped revision repository for this call.
    return subprocess.check_output([str(GIT), "-c", f"safe.directory={REV_ROOT.as_posix()}", *args],
                                   cwd=REV_ROOT, text=True, encoding="utf-8").strip()


def canonical_box(row: dict) -> tuple[dict, bool]:
    names = ("bbox_x_px", "bbox_y_px", "bbox_width_px", "bbox_height_px", "image_width_px", "image_height_px")
    values = [float(row[name]) for name in names]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Nonfinite source geometry")
    x, y, w, h, width, height = values
    if width < 4 or height < 4 or w < 0 or h < 0:
        raise ValueError("Invalid source geometry")
    new_w, new_h = min(max(w, 4.0), width), min(max(h, 4.0), height)
    new_x, new_y = min(max(x, 0.0), width - new_w), min(max(y, 0.0), height - new_h)
    canonical = dict(zip(names[:4], (new_x, new_y, new_w, new_h)))
    canonical["bbox_area_px2"] = new_w * new_h
    return canonical, (x, y, w, h) != (new_x, new_y, new_w, new_h)


def split_assignments(properties: list[dict], seed: int) -> dict[str, str]:
    groups = defaultdict(list)
    for row in properties:
        groups[row["stratum"]].append(row["specimen"])
    if len(groups) != 32 or sum(map(len, groups.values())) != 176:
        raise RuntimeError("Unexpected specimen/stratum roster")
    assignments = {}
    for group in sorted(groups):
        specimens = sorted(groups[group])
        if len(set(specimens)) != len(specimens) or len(specimens) < 3:
            raise RuntimeError("Invalid stratum specimen membership")
        entropy = int.from_bytes(hashlib.sha256(f"{seed}|{group}".encode()).digest()[:8], "big")
        random.Random(entropy).shuffle(specimens)
        for index, specimen in enumerate(specimens):
            assignments[specimen] = "test" if index == 0 else "validation" if index == 1 else "train"
    if dict(Counter(assignments.values())) != {"train": 112, "validation": 32, "test": 32}:
        raise RuntimeError("Unexpected split sizes")
    return assignments


def freeze_environment() -> None:
    output = REV_ROOT / "env"
    output.mkdir(parents=True, exist_ok=True)
    if (output / "versions.json").exists():
        return
    import torch
    import torchvision
    freeze = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True, encoding="utf-8")
    atomic_text(output / "pip_freeze.txt", freeze)
    driver = subprocess.check_output(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"], text=True).strip()
    versions = {
        "created_utc": utc_now(), "python": sys.version, "executable": sys.executable,
        "platform": platform.platform(), "torch": torch.__version__,
        "torchvision": torchvision.__version__, "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(), "gpu": driver,
        "pycocotools": importlib.metadata.version("pycocotools"),
        "protocol_commit": PROTOCOL_COMMIT,
        "determinism_scope": "Seeded single-host runtime; warn-only nondeterministic operations logged, no cross-device bitwise guarantee.",
        "pip_freeze_sha256": sha256_file(output / "pip_freeze.txt"),
    }
    write_json(output / "versions.json", versions)
    record_event("environment_frozen", {"versions_sha256": sha256_file(output / "versions.json")})


def initialize() -> None:
    verify_protocols()
    commit_info = git("show", "-s", "--format=%H%n%cI%n%s", PROTOCOL_COMMIT).splitlines()
    if commit_info[0] != PROTOCOL_COMMIT:
        raise RuntimeError("Protocol commit missing")
    events = verify_event_chain()
    if not any(event["event"] == "protocol_frozen" for event in events):
        record_event("protocol_frozen", {
            "commit": commit_info[0], "git_commit_time": commit_info[1],
            "hashes_sha256": sha256_file(REV_ROOT / "protocol" / "hashes.json"),
            "recording_time_is_not_commit_time": True,
        })
    freeze_environment()
    for task in ("T1", "T2", "T3A", "T3B", "T3C", "T4", "T5", "T6A", "T6B", "T7", "T8"):
        folder = task_dir(task)
        if not (folder / "summary.json").exists():
            write_task_summary(task, {"task_id": task, "status": "pending", "test_status": "locked_pending_T7"})
            append_run_log(task, "Protocol frozen. No completed experiment result is asserted.")


def prepare_split() -> None:
    initialize()
    protocol = load_protocol("T1")
    preparation_paths = [REV_ROOT / "scripts" / name for name in ("prepare_rev02.py", "rev02_common.py")]
    if git("status", "--porcelain", "--", *(str(path.relative_to(REV_ROOT)) for path in preparation_paths)):
        raise RuntimeError("Commit preparation/common source before the one-time split steward")
    preparation_commit = git("rev-parse", "HEAD")
    if subprocess.run([str(GIT), "-c", f"safe.directory={REV_ROOT.as_posix()}", "merge-base", "--is-ancestor", PROTOCOL_COMMIT, preparation_commit], cwd=REV_ROOT).returncode:
        raise RuntimeError("Preparation commit must follow the frozen protocol commit")
    start_receipt = REV_ROOT / "splits" / "CREATION_STARTED.json"
    if start_receipt.exists() or (REV_ROOT / "splits" / "rev02_split.json").exists():
        raise RuntimeError("REV02 split generation is one-time; an existing/partial split cannot be regenerated")
    if (ARTIFACT_ROOT / "sealed").exists():
        raise RuntimeError("Unexpected pre-existing sealed directory")
    source_manifest = Path(protocol["inputs"]["manifest"])
    source_specimens = Path(protocol["inputs"]["specimens"])
    for path, expected in ((source_manifest, protocol["inputs"]["manifest_sha256"]),
                           (source_specimens, protocol["inputs"]["specimens_sha256"])):
        if sha256_file(path) != expected:
            raise RuntimeError("Source metadata changed from preregistration")
    write_json(start_receipt, {"started_utc": utc_now(), "split_seed": 61001,
                              "preparation_source_sha256": sha256_file(Path(__file__)),
                              "common_source_sha256": sha256_file(REV_ROOT / "scripts" / "rev02_common.py"),
                              "preparation_git_commit": preparation_commit})
    record_event("split_steward_started", {"scope": "source metadata only; no new test pixels/labels",
                                           "seed": 61001})
    try:
        with steward_metadata_access():
            properties = read_csv(source_specimens)
            assignments = split_assignments(properties, 61001)
            all_rows = read_csv(source_manifest)
            rows_by_split = {key: [] for key in ("train", "validation", "test")}
            box_changes = []
            canonical_mismatches = 0
            public_inputs_verified = 0
            image_hash_memberships = defaultdict(set)
            for original in sorted(all_rows, key=lambda row: row["sample_id"]):
                split = assignments[original["specimen"]]
                row = {**original, "split": split, "source_kind": "real"}
                box, changed = canonical_box(row)
                row.update({"raw_" + key: original[key] for key in box})
                row.update(box)
                row["canonical_box_changed"] = int(changed)
                row["source_label_path"] = original["label_path"]
                row["source_label_sha256"] = original["label_sha256"]
                image_hash_memberships[row["image_sha256"]].add(split)
                if split != "test":
                    source_image = PROJECT_ROOT / original["image_path"]
                    source_label = PROJECT_ROOT / original["label_path"]
                    if sha256_file(source_image) != original["image_sha256"] or sha256_file(source_label) != original["label_sha256"]:
                        raise RuntimeError("Train/validation input hash mismatch")
                    label = load_json(source_label)
                    for axis in ("x", "y", "width", "height"):
                        label[f"anotation.bbox.{axis}"] = box[f"bbox_{axis}_px"]
                    destination = REV_ROOT / "data" / "labels" / split / original["specimen"] / original["view"] / f"{original['frame_stem']}.json"
                    write_json(destination, label)
                    row["label_path"] = str(destination)
                    row["label_sha256"] = sha256_file(destination)
                    canonical_mismatches += any(float(label[f"anotation.bbox.{axis}"]) != float(row[f"bbox_{axis}_px"]) for axis in ("x", "y", "width", "height"))
                    public_inputs_verified += 1
                else:
                    row["canonical_label_status"] = "deferred_until_T7_manifest_geometry_canonical"
                rows_by_split[split].append(row)
                if changed and split != "test":
                    box_changes.append({"sample_id": row["sample_id"], "split": split,
                                        **{key: row[key] for key in box},
                                        **{"raw_" + key: original[key] for key in box}})
            expected = {"train": 3584, "validation": 1024, "test": 1024}
            if {split: len(rows) for split, rows in rows_by_split.items()} != expected:
                raise RuntimeError("Unexpected per-split image counts")
            if len({row["sample_id"] for row in all_rows}) != 5632:
                raise RuntimeError("Duplicate source sample ID")
            public_assignments = {key: value for key, value in assignments.items() if value != "test"}
            strata = []
            for group in sorted({row["stratum"] for row in properties}):
                counts = Counter(assignments[row["specimen"]] for row in properties if row["stratum"] == group)
                if not all(counts[split] >= 1 for split in expected):
                    raise RuntimeError("A stratum is absent from a split")
                strata.append({"stratum": group, **counts})
            for split in ("train", "validation"):
                write_csv(REV_ROOT / "data" / f"{split}.csv", rows_by_split[split])
            write_csv(ARTIFACT_ROOT / "sealed" / "test_manifest.csv", rows_by_split["test"])
            write_json(ARTIFACT_ROOT / "sealed" / "test_specimens.json", {
                "split_seed": 61001, "specimens": sorted(key for key, value in assignments.items() if value == "test")})
            sealed_hashes = {name: sha256_file(ARTIFACT_ROOT / "sealed" / name)
                             for name in ("test_manifest.csv", "test_specimens.json")}
            duplicates = sum(len(memberships) > 1 for memberships in image_hash_memberships.values())
            if duplicates:
                raise RuntimeError(f"Cross-split exact image duplicates: {duplicates}; preserve split, stop for disclosure")
            public = {
                "protocol_id": protocol["protocol_id"], "seed": 61001, "sealed_utc": utc_now(),
                "specimen_counts": dict(Counter(assignments.values())), "image_counts": expected,
                "train_validation_assignments": public_assignments, "stratum_counts": strata,
                "sealed_test_hashes": sealed_hashes, "source_manifest_sha256": protocol["inputs"]["manifest_sha256"],
                "source_specimens_sha256": protocol["inputs"]["specimens_sha256"],
                "public_manifest_sha256": {split: sha256_file(REV_ROOT / "data" / f"{split}.csv") for split in ("train", "validation")},
                "disjoint_specimen_partitions": True, "cross_split_duplicate_image_hashes": duplicates,
                "steward_test_pixels_opened": 0, "steward_test_label_JSON_opened": 0,
                "public_image_and_label_hash_pairs_verified": public_inputs_verified,
                "canonical_public_label_manifest_mismatches": canonical_mismatches,
                "canonical_box_change_counts": {split: sum(int(row["canonical_box_changed"]) for row in rows) for split, rows in rows_by_split.items()},
                "prior_exposure_disclosure": "A locked internal re-split of an already studied corpus, not a new independent test corpus."
            }
            write_json(REV_ROOT / "splits" / "rev02_split.json", public)
            write_json(REV_ROOT / "splits" / "rev02_split.sha256.json", {
                "path": "splits/rev02_split.json", "sha256": sha256_file(REV_ROOT / "splits" / "rev02_split.json")})
            output = task_dir("T1")
            write_csv(output / "raw" / "stratum_counts.csv", strata)
            write_csv(output / "raw" / "public_box_changes.csv", box_changes,
                      fieldnames=list(box_changes[0]) if box_changes else ["sample_id", "split"])
            write_task_summary("T1", {"task_id": "T1", "status": "split_prepared_experiments_pending",
                                      "test_status": "locked_pending_T7", "split_integrity": public})
        record_event("split_sealed", {"split_file_sha256": sha256_file(REV_ROOT / "splits" / "rev02_split.json"),
                                       "test_metadata_sealed_hashes": sealed_hashes,
                                       "test_pixels_and_label_json_opened": 0})
        append_run_log("T1", "One-time specimen split and canonical train/validation boxes prepared; model test access remains locked.")
        print(json.dumps({"status": "split_prepared", "specimens": public["specimen_counts"],
                          "images": expected, "canonical_public_mismatches": canonical_mismatches,
                          "test_payload_opened": False}, indent=2))
    except Exception as exc:
        record_event("split_preparation_failed", {"error": str(exc), "reseed_allowed": False})
        raise


def historical_index() -> None:
    folder = REV_ROOT / "results" / "historical"
    if (folder / "index.json").exists():
        return
    baseline = PROJECT_ROOT / "revision_20260902" / "10_release" / "submission_release_20260902_v49"
    records = [{"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
               for path in sorted(baseline.rglob("*")) if path.is_file()]
    write_json(folder / "index.json", {"created_utc": utc_now(), "status": "historical_preliminary_not_REV02_results",
                                       "originals_preserved": True, "baseline": str(baseline), "files": records})
    atomic_text(folder / "README.md", "# Preserved preliminary results\n\nThe immutable v49 release and prior experiments remain at their original paths. This index records their hashes without moving or overwriting them. Historical results are not pooled with REV02 inference and are reserved for the supplement.\n")
    record_event("historical_indexed", {"files": len(records), "index_sha256": sha256_file(folder / "index.json")})


def freeze_code() -> None:
    verify_protocols()
    path = REV_ROOT / "IMPLEMENTATION_LOCK.json"
    if path.exists():
        raise FileExistsError("Implementation is already frozen; a changed source requires an explicit amendment")
    dirty = git("status", "--porcelain", "--untracked-files=normal", "--", "scripts", "tests", "protocol")
    if dirty:
        raise RuntimeError(f"Commit implementation before source freeze: {dirty}")
    source_paths = sorted((REV_ROOT / "scripts").glob("*.py")) + sorted((PROJECT_ROOT / "src" / "metal_spatter_pinn").glob("*.py"))
    source_paths += [PROJECT_ROOT / "scripts" / "audit_memorization.py",
                     PROJECT_ROOT / "revision_20260902" / "06_statistics" / "stats_core.py"]
    required = {"rev02_common.py", "prepare_rev02.py", "rev02_generator.py", "rev02_detector.py", "rev02_memorization.py", "rev02_statistics.py", "run_rev02_pipeline.py"}
    if not required.issubset({candidate.name for candidate in source_paths}):
        raise RuntimeError("Missing required implementation component")
    lock = {"created_utc": utc_now(), "protocol_commit": PROTOCOL_COMMIT,
            "implementation_commit": git("rev-parse", "HEAD"),
            "protocol_receipt_sha256": sha256_file(REV_ROOT / "protocol" / "hashes.json"),
            "source_hashes": {str(candidate.resolve()): sha256_file(candidate) for candidate in source_paths},
            "environment_sha256": sha256_file(REV_ROOT / "env" / "versions.json")}
    write_json(path, lock)
    record_event("implementation_frozen", {"implementation_commit": lock["implementation_commit"],
                                            "lock_sha256": sha256_file(path)})
    print(json.dumps({"status": "implementation_frozen", "source_files": len(source_paths)}, indent=2))


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("initialize", "split", "historical-index", "freeze-code", "verify"))
    args = parser.parse_args(argv)
    if args.action == "initialize": initialize()
    elif args.action == "split": prepare_split()
    elif args.action == "historical-index": historical_index()
    elif args.action == "freeze-code": freeze_code()
    else:
        verify_protocols()
        print(json.dumps({"protocols": "verified", "chronology_events": len(verify_event_chain())}))


if __name__ == "__main__":
    main()
