"""Shared fail-closed provenance and split-access boundaries for REV02.

The access ledger is procedural evidence, not third-party or OS-level blinding.
Only the one-time metadata steward and the global T7 coordinator may handle the
sealed partition. Modelling functions must always name their requested split.
"""
from __future__ import annotations

from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import sys
import tempfile
import time
from typing import Any, Iterable

import yaml

REV_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = REV_ROOT.parent
ARTIFACT_ROOT = Path("D:/Codex Research/LPBF_REV02_20260902")
PROTOCOL_COMMIT = "8e3aa248dd9849149561f7c938ab455392d7b5b0"
sys.path.insert(0, str(PROJECT_ROOT / "src"))
_STEWARD_METADATA_ACCESS = False


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False, default=_json_default).encode("utf-8")


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"Not a JSON value: {type(value)}")


def _inside(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _sealed(path: Path) -> bool:
    return _inside(Path(path), ARTIFACT_ROOT / "sealed")


def _check_read(path: Path) -> None:
    if _sealed(path) and not _STEWARD_METADATA_ACCESS:
        require_test_unlocked()
        record_event("test_payload_read", {"path": str(Path(path).resolve()),
                                          "scope": "fixed_T7_evaluation"})


def sha256_file(path: Path | str) -> str:
    path = Path(path)
    _check_read(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path | str) -> Any:
    path = Path(path)
    _check_read(path)
    with path.open(encoding="utf-8-sig") as handle:
        return json.load(handle)


def atomic_text(path: Path | str, text: str, overwrite: bool = False) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to replace existing artifact: {path}")
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temp_path = Path(temporary)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists() and not overwrite:
            raise FileExistsError(f"Artifact appeared during write: {path}")
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()  # Only this function's newly-created temporary file.
    return path


def write_json(path: Path | str, value: Any, overwrite: bool = False) -> Path:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                      allow_nan=False, default=_json_default) + "\n"
    return atomic_text(path, text, overwrite)


def read_csv(path: Path | str) -> list[dict[str, str]]:
    path = Path(path)
    _check_read(path)
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path | str, rows: Iterable[dict], fieldnames: list[str] | None = None,
              overwrite: bool = False) -> Path:
    rows = list(rows)
    if fieldnames is None:
        fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    if not fieldnames:
        raise ValueError("Empty CSV requires declared fieldnames")
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, extrasaction="raise", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return atomic_text(path, buffer.getvalue(), overwrite)


def verify_protocols() -> dict:
    receipt = load_json(REV_ROOT / "protocol" / "hashes.json")
    task_names = {"MASTER", "T1", "T2", "T3A", "T3B", "T3C", "T4", "T5", "T6A", "T6B", "T7", "T8"}
    required = {f"protocol/REV02_{task}.yaml" for task in task_names}
    required |= {"protocol/IMPLEMENTATION_INTERFACE.md", "protocol/INTERPRETATION_NOTES.md"}
    if (receipt.get("protocol_id") != "LPBF-WEAK-PHYSICS-REV02-20260902-v1"
        or receipt.get("algorithm") != "SHA256" or set(receipt.get("files", {})) != required):
        raise RuntimeError("Protocol receipt must contain the complete fourteen-file frozen specification")
    for relative, expected in receipt["files"].items():
        actual = sha256_file(REV_ROOT / relative)
        if actual != expected:
            raise RuntimeError(f"Frozen protocol changed: {relative}: {actual} != {expected}")
    return receipt


def load_protocol(task: str) -> dict:
    if task not in {"MASTER", "T1", "T2", "T3A", "T3B", "T3C", "T4", "T5", "T6A", "T6B", "T7", "T8"}:
        raise ValueError(f"Unknown task: {task}")
    verify_protocols()
    path = REV_ROOT / "protocol" / f"REV02_{task}.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def task_dir(task: str) -> Path:
    load_protocol(task)
    path = REV_ROOT / "results" / "REV02" / task
    for directory in (path, path / "raw", path / "figures"):
        directory.mkdir(parents=True, exist_ok=True)
    source = REV_ROOT / "protocol" / f"REV02_{task}.yaml"
    config = path / "config.yaml"
    if not config.exists():
        atomic_text(config, source.read_text(encoding="utf-8"))
    elif sha256_file(config) != sha256_file(source):
        raise RuntimeError(f"Task config differs from frozen protocol: {task}")
    return path


def resolve_path(value: Path | str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    _check_read(path)
    return path


def manifest_path(split: str) -> Path:
    if split not in {"train", "validation", "test"}:
        raise ValueError("An explicit train, validation or test split is required")
    if split == "test":
        require_test_unlocked()
    split_path = REV_ROOT / "splits" / "rev02_split.json"
    receipt_path = REV_ROOT / "splits" / "rev02_split.sha256.json"
    if not split_path.exists() or not receipt_path.exists():
        raise RuntimeError("Immutable one-time split receipt is missing")
    if sha256_file(split_path) != load_json(receipt_path)["sha256"]:
        raise RuntimeError("Frozen public split file changed")
    frozen_split = load_json(split_path)
    if split == "test":
        expected_hashes = frozen_split.get("sealed_test_hashes", {})
        if set(expected_hashes) != {"test_manifest.csv", "test_specimens.json"}:
            raise RuntimeError("Frozen sealed test identity ledger is incomplete")
        for name, expected in expected_hashes.items():
            if sha256_file(ARTIFACT_ROOT / "sealed" / name) != expected:
                raise RuntimeError(f"Frozen sealed test file changed: {name}")
        return ARTIFACT_ROOT / "sealed" / "test_manifest.csv"
    path = REV_ROOT / "data" / f"{split}.csv"
    expected = frozen_split["public_manifest_sha256"][split]
    if sha256_file(path) != expected:
        raise RuntimeError(f"Frozen {split} manifest changed")
    return path


def load_rows(split: str) -> list[dict[str, str]]:
    path = manifest_path(split)
    rows = read_csv(path)
    if not rows or any(row.get("split") != split or row.get("source_kind") != "real" for row in rows):
        raise RuntimeError(f"Split/source contamination in {split} manifest")
    if len({row["sample_id"] for row in rows}) != len(rows):
        raise RuntimeError(f"Duplicate sample IDs in {split} manifest")
    for row in rows:
        for key in ("image_path", "label_path"):
            if not row.get(key):
                raise RuntimeError(f"Missing {key} in {split} manifest")
    return rows


def generator_run_dir(arm: str, seed: int) -> Path:
    return ARTIFACT_ROOT / "runs" / "generators" / f"{arm}_s{int(seed)}"


def detector_run_dir(run_id: str) -> Path:
    if Path(run_id).name != run_id:
        raise ValueError("Run ID must be a single path component")
    return ARTIFACT_ROOT / "runs" / "detectors" / run_id


def pool_dir(arm: str) -> Path:
    if arm not in {"G1", "G2", "D5"}:
        raise ValueError(f"Unknown pool {arm}")
    return ARTIFACT_ROOT / "pools" / arm


@contextmanager
def _ledger_lock():
    logs = REV_ROOT / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    lock = logs / "chronology.lock"
    with lock.open("a+b") as handle:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def verify_event_chain() -> list[dict]:
    path = REV_ROOT / "logs" / "events.jsonl"
    if not path.exists():
        return []
    events = []
    previous = "0" * 64
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        item = json.loads(line)
        claimed = item.pop("event_sha256")
        if item["previous_sha256"] != previous or item["sequence"] != index + 1:
            raise RuntimeError("Broken chronology sequence")
        actual = hashlib.sha256(canonical_bytes(item)).hexdigest()
        if actual != claimed:
            raise RuntimeError("Modified chronology event")
        item["event_sha256"] = claimed
        events.append(item)
        previous = claimed
    return events


def record_event(event: str, payload: dict | None = None) -> dict:
    with _ledger_lock():
        events = verify_event_chain()
        item = {"sequence": len(events) + 1, "utc": utc_now(), "event": event,
                "payload": payload or {},
                "previous_sha256": events[-1]["event_sha256"] if events else "0" * 64}
        item["event_sha256"] = hashlib.sha256(canonical_bytes(item)).hexdigest()
        with (REV_ROOT / "logs" / "events.jsonl").open("a", encoding="utf-8", newline="") as handle:
            handle.write(canonical_bytes(item).decode("utf-8") + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        events.append(item)
        def first(kind):
            return next((entry["utc"] for entry in events if entry["event"] == kind), None)
        projection = {
            "protocol_id": "LPBF-WEAK-PHYSICS-REV02-20260902-v1",
            "updated_utc": item["utc"], "event_count": len(events),
            "head_sha256": item["event_sha256"], "ledger": "logs/events.jsonl",
            "protocol_freeze_utc": first("protocol_frozen"),
            "split_steward_creation_utc": first("split_steward_started"),
            "split_sealed_utc": first("split_sealed"),
            "T7_unlock_utc": first("test_unlocked"),
            "first_model_test_payload_read_utc": first("test_payload_read"),
            "status": "test_unlocked" if first("test_unlocked") else "test_locked",
            "scope": "Application-level chronology; split-steward metadata exception and prior corpus exposure disclosed."
        }
        write_json(REV_ROOT / "access_chronology.json", projection, overwrite=True)
        return item


@contextmanager
def steward_metadata_access():
    """Only prepare_rev02 uses this to assign/write/hash sealed metadata once."""
    global _STEWARD_METADATA_ACCESS
    if _STEWARD_METADATA_ACCESS:
        raise RuntimeError("Nested steward access is forbidden")
    _STEWARD_METADATA_ACCESS = True
    try:
        yield
    finally:
        _STEWARD_METADATA_ACCESS = False


def require_test_unlocked() -> dict:
    path = REV_ROOT / "T7_UNLOCK.json"
    if not path.exists():
        raise PermissionError("REV02 test is locked until every T1-T6 pretest receipt is frozen")
    # This receipt is public metadata; reading it cannot recursively request test access.
    with path.open(encoding="utf-8") as handle:
        unlock = json.load(handle)
    if unlock.get("status") != "unlocked" or unlock.get("campaign_count") != 1:
        raise PermissionError("Invalid T7 unlock receipt")
    freeze_path = REV_ROOT / "PRETEST_FREEZE.json"
    if not freeze_path.exists() or sha256_file(freeze_path) != unlock.get("pretest_freeze_sha256"):
        raise PermissionError("T7 pretest freeze is missing or changed")
    freeze = load_json(freeze_path)
    validate_pretest_freeze(freeze)
    if datetime.fromisoformat(unlock["unlocked_utc"]) <= datetime.fromisoformat(freeze["last_training_validation_end_utc"]):
        raise PermissionError("T7 chronology is out of order")
    return unlock


def validate_pretest_freeze(freeze: dict) -> None:
    """Validate public pretest evidence without creating/reading an unlock marker."""
    required_tasks = {"T1", "T2", "T3A", "T3B", "T3C", "T4", "T5", "T6A", "T6B"}
    expected_receipts = {f"results/REV02/{task}/PRETEST_COMPLETE.json" for task in required_tasks}
    if freeze.get("status") != "pretest_complete" or set(freeze.get("receipt_hashes", {})) != expected_receipts:
        raise PermissionError("T7 requires all nine pretest task receipts, not an empty or partial set")
    if freeze.get("protocol_commit") != PROTOCOL_COMMIT:
        raise PermissionError("T7 protocol commit does not match")
    verify_protocols()
    if freeze.get("protocol_receipt_sha256") != sha256_file(REV_ROOT / "protocol" / "hashes.json"):
        raise PermissionError("T7 protocol hash receipt changed")
    if freeze.get("implementation_lock_sha256") != sha256_file(REV_ROOT / "IMPLEMENTATION_LOCK.json"):
        raise PermissionError("T7 implementation lock changed")
    verify_implementation_lock()
    if not freeze.get("test_roster", {}).get("generators") or not freeze.get("test_roster", {}).get("detectors"):
        raise PermissionError("T7 frozen model/checkpoint roster is missing")
    for relative, expected in freeze["receipt_hashes"].items():
        if sha256_file(REV_ROOT / relative) != expected:
            raise PermissionError(f"Changed pretest task receipt: {relative}")
        receipt = load_json(REV_ROOT / relative)
        if receipt.get("task_id") not in required_tasks or relative != f"results/REV02/{receipt['task_id']}/PRETEST_COMPLETE.json":
            raise PermissionError("T7 task identity mismatch")
        if (receipt.get("status") != "pretest_complete" or not receipt.get("integrity_checks_passed")
            or not receipt.get("expected_units", 0) or receipt.get("completed_units") != receipt.get("expected_units")
            or receipt.get("missing_units") != [] or not receipt.get("output_hashes")):
            raise PermissionError(f"Unfinished task receipt: {relative}")
        if receipt.get("implementation_lock_sha256") != freeze["implementation_lock_sha256"]:
            raise PermissionError("Task implementation lock mismatch")
        if datetime.fromisoformat(receipt["last_training_validation_end_utc"]) > datetime.fromisoformat(freeze["last_training_validation_end_utc"]):
            raise PermissionError("Task finished after the declared global freeze cutoff")
    generators = freeze["test_roster"]["generators"]
    detectors = freeze["test_roster"]["detectors"]
    if (len(generators) != 78 or len({(row["arm"], row["seed"]) for row in generators}) != 78
        or any(len(row.get("checkpoint_sha256", "")) != 64 for row in generators)):
        raise PermissionError("T7 requires 78 uniquely bound generator checkpoints")
    if (len(detectors) not in (45, 57) or len({row["run_id"] for row in detectors}) != len(detectors)
        or any(row.get("milestones") != [896, 1792, 4480, 8960] or len(row.get("completion_sha256", "")) != 64 for row in detectors)):
        raise PermissionError("T7 requires the complete unique detector completion-hash roster")


def verify_implementation_lock() -> dict:
    lock_path = REV_ROOT / "IMPLEMENTATION_LOCK.json"
    if not lock_path.exists():
        raise RuntimeError("Implementation must be frozen before experiment execution")
    lock = load_json(lock_path)
    required_scripts = {"rev02_common.py", "prepare_rev02.py", "rev02_generator.py", "rev02_detector.py",
                        "rev02_memorization.py", "rev02_statistics.py", "run_rev02_pipeline.py"}
    source_hashes = lock.get("source_hashes", {})
    if not required_scripts.issubset({Path(path).name for path in source_hashes}):
        raise RuntimeError("Implementation lock must cover every required experiment component")
    if lock.get("protocol_commit") != PROTOCOL_COMMIT or len(lock.get("implementation_commit", "")) != 40:
        raise RuntimeError("Implementation lock lacks a valid protocol/implementation commit identity")
    if lock.get("protocol_receipt_sha256") != sha256_file(REV_ROOT / "protocol" / "hashes.json"):
        raise RuntimeError("Implementation lock is not bound to the frozen protocol receipt")
    for path, expected in lock["source_hashes"].items():
        if sha256_file(Path(path)) != expected:
            raise RuntimeError(f"Experiment source changed after lock: {path}")
    return lock


def run_provenance(task: str, config: dict, inputs: dict[str, Path | str]) -> dict:
    protocol = load_protocol(task)
    lock = verify_implementation_lock()
    env_paths = [REV_ROOT / "env" / "pip_freeze.txt", REV_ROOT / "env" / "versions.json"]
    if any(not path.exists() for path in env_paths):
        raise RuntimeError("Environment lock missing")
    if sha256_file(env_paths[1]) != lock["environment_sha256"]:
        raise RuntimeError("Environment metadata changed after implementation freeze")
    if sha256_file(env_paths[0]) != load_json(env_paths[1])["pip_freeze_sha256"]:
        raise RuntimeError("pip freeze changed after environment lock")
    input_hashes = {name: {"path": str(Path(path).resolve()), "sha256": sha256_file(path)}
                    for name, path in inputs.items()}
    return {
        "protocol_id": protocol["protocol_id"], "task_id": task,
        "protocol_commit": PROTOCOL_COMMIT,
        "task_protocol_sha256": sha256_file(REV_ROOT / "protocol" / f"REV02_{task}.yaml"),
        "protocol_receipt_sha256": sha256_file(REV_ROOT / "protocol" / "hashes.json"),
        "implementation_lock_sha256": sha256_file(REV_ROOT / "IMPLEMENTATION_LOCK.json"),
        "implementation_commit": lock["implementation_commit"],
        "source_hashes": lock["source_hashes"],
        "environment_hashes": {path.name: sha256_file(path) for path in env_paths},
        "input_hashes": input_hashes,
        "config_sha256": hashlib.sha256(canonical_bytes(config)).hexdigest()
    }


def write_task_summary(task: str, summary: dict) -> Path:
    return write_json(task_dir(task) / "summary.json", summary, overwrite=True)


def seed_everything(seed: int) -> None:
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def append_run_log(task: str, message: str) -> None:
    path = task_dir(task) / "RUN_LOG.md"
    previous = path.read_text(encoding="utf-8") if path.exists() else f"# REV02 {task} run log\n"
    atomic_text(path, previous + f"\n- {utc_now()} — {message}\n", overwrite=path.exists())
