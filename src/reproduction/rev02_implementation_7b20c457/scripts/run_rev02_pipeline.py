"""Serialized, resumable REV02 command coordinator; pretest is the default stop.

Never infers permission to unlock the test set from one completed training run.
The global pretest receipt and a separate acceptance audit are required for T7.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from rev02_common import (
    ARTIFACT_ROOT, PROJECT_ROOT, PROTOCOL_COMMIT, REV_ROOT, atomic_text,
    canonical_bytes, detector_run_dir, generator_run_dir, load_json,
    load_protocol, record_event, require_test_unlocked, sha256_file,
    task_dir, utc_now, verify_event_chain, verify_implementation_lock,
    validate_pretest_freeze, verify_protocols, write_json,
)

GENERATOR_SEEDS = tuple(range(41001, 41011))
DETECTOR_SEEDS = (51001, 51002, 51003)
BASE_ARMS = ("G0", "F00", "F01", "F10", "F11", "G2N")
SENSITIVITY_ARMS = ("F11_kappa_low", "F11_kappa_high", "F11_source_low", "F11_source_high", "F11_decay_low", "F11_decay_high")
REQUIRED_TASKS = ("T1", "T2", "T3A", "T3B", "T3C", "T4", "T5", "T6A", "T6B")


def job(module: str, *arguments) -> dict:
    argv = [str(argument) for argument in arguments]
    return {"module": module, "arguments": argv,
            "job_id": hashlib.sha256(canonical_bytes([module, argv])).hexdigest()[:20]}


def initial_jobs() -> list[dict]:
    plan = [job("rev02_generator", "make-plans")]
    for arm in BASE_ARMS:
        for seed in GENERATOR_SEEDS:
            plan.extend([job("rev02_generator", "train", "--arm", arm, "--seed", seed),
                         job("rev02_generator", "evaluate", "--arm", arm, "--seed", seed, "--split", "validation")])
    for arm in SENSITIVITY_ARMS:
        for seed in GENERATOR_SEEDS[:3]:
            plan.extend([job("rev02_generator", "train", "--arm", arm, "--seed", seed),
                         job("rev02_generator", "evaluate", "--arm", arm, "--seed", seed, "--split", "validation")])
    plan.append(job("rev02_generator", "controls", "--split", "validation"))
    for arm in ("G1", "G2", "D5"):
        plan.append(job("rev02_generator", "make-pool", "--arm", arm))
    plan.append(job("rev02_generator", "summarize"))
    for action in ("calibrate", "controls", "pools", "summarize"):
        plan.append(job("rev02_memorization", action))
    for arm in ("D2", "D4"):
        for ratio in (0.125, 0.25, 0.5, 1.0, 2.0):
            for seed in DETECTOR_SEEDS:
                plan.append(job("rev02_detector", "ratio", "--arm", arm, "--ratio", ratio, "--seed", seed))
    plan.append(job("rev02_detector", "select-ratios"))
    return plan


def selected_ratios() -> dict[str, float]:
    path = task_dir("T6A")
    freeze_path = path / "PRETEST_SELECTION_FREEZE.json"
    if not freeze_path.exists():
        raise RuntimeError("The complete validation-only ratio selection has not been frozen")
    values = {}
    for arm in ("D2", "D4"):
        receipt = load_json(path / f"{arm}_ratio_selection.json")
        ratio = receipt.get("selected_ratio")
        if ratio is None and isinstance(receipt.get("selection"), dict):
            ratio = receipt["selection"].get("selected_ratio")
        if ratio not in (0.125, 0.25, 0.5, 1.0, 2.0):
            raise RuntimeError(f"Invalid frozen selected ratio for {arm}")
        values[arm] = float(ratio)
    return values


def budget_configurations(ratios: dict[str, float]) -> list[dict]:
    rows = []
    for family in ("fasterrcnn", "retinanet"):
        axes = ("B1B3", "B2", "B4") if family == "fasterrcnn" else ("B4",)
        arms = ("D0", "D1", "D2", "D4", "D5") if family == "fasterrcnn" else ("D0", "D2", "D4", "D5")
        for arm in arms:
            arm_ratios = [0.0] if arm in ("D0", "D1") else sorted(set(ratios.values())) if arm == "D5" else [ratios[arm]]
            arm_axes = ("B2",) if family == "fasterrcnn" and arm in ("D0", "D1") else axes
            for ratio in arm_ratios:
                for axis in arm_axes:
                    for seed in DETECTOR_SEEDS:
                        rows.append({"family": family, "arm": arm, "axis": axis, "ratio": ratio, "seed": seed})
    return rows


def budget_jobs() -> list[dict]:
    rows = budget_configurations(selected_ratios())
    plan = [job("rev02_detector", "budget", "--family", row["family"], "--arm", row["arm"],
                "--axis", row["axis"], "--ratio", row["ratio"], "--seed", row["seed"]) for row in rows]
    plan += [job("rev02_detector", "summarize"),
             job("rev02_statistics", "summarize", "--split", "validation")]
    return plan


@contextmanager
def process_lock():
    directory = REV_ROOT / "logs"
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "pipeline.lock").open("a+b") as handle:
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError("Another REV02 coordinator is already active; do not duplicate training") from exc
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def update_manifest(status: dict) -> None:
    directory = REV_ROOT / "logs" / "commands"
    records = []
    if directory.exists():
        for path in sorted(directory.glob("*.json")):
            if path.name.endswith(".started.json") or path.name.endswith(".completed.json") or ".failed." in path.name:
                value = load_json(path)
                records.append((value.get("started_utc", ""), path.name, value))
    lines = ["# REV02 execution manifest", "", f"Updated: {utc_now()}", "",
             "All completed, failed and interrupted command records are retained. A command is not an independent statistical replicate. Test remains locked until the global T7 acceptance gate.", "",
             f"Current pipeline status: {status.get('status')}", "",
             "| Started (UTC) | Command | Status | Record |", "| --- | --- | --- | --- |"]
    for started, filename, record in sorted(records):
        command = record.get("module", "") + " " + " ".join(record.get("arguments", []))
        lines.append(f"| {started} | {command} | {record.get('status','started')} | logs/commands/{filename} |")
    atomic_text(REV_ROOT / "MANIFEST.md", "\n".join(lines) + "\n", overwrite=True)


def execute_job(spec: dict, index: int, total: int, stage: str) -> bool:
    verify_implementation_lock()
    directory = REV_ROOT / "logs" / "commands"
    directory.mkdir(parents=True, exist_ok=True)
    key = f"{stage}_{spec['job_id']}"
    done = directory / f"{key}.completed.json"
    if done.exists():
        receipt = load_json(done)
        if receipt["module"] != spec["module"] or receipt["arguments"] != spec["arguments"]:
            raise RuntimeError("Completed command identity mismatch")
        if receipt["implementation_lock_sha256"] != sha256_file(REV_ROOT / "IMPLEMENTATION_LOCK.json"):
            raise RuntimeError("Completed command has a different implementation lock; explicit reconciliation required")
        return False
    started_path = directory / f"{key}.started.json"
    if started_path.exists():
        raise RuntimeError(f"Interrupted/failed command {spec['job_id']} requires a documented same-seed recovery, not automatic rerun")
    free = shutil.disk_usage(ARTIFACT_ROOT).free
    if free < 8 * 1024**3:
        raise RuntimeError("Insufficient scoped artifact-drive space for retained checkpoints")
    command = [sys.executable, "-u", str(REV_ROOT / "scripts" / f"{spec['module']}.py"), *spec["arguments"]]
    log_path = directory / f"{key}.log"
    began = utc_now()
    entry = {**spec, "status": "started", "started_utc": began, "stage": stage,
             "implementation_lock_sha256": sha256_file(REV_ROOT / "IMPLEMENTATION_LOCK.json"),
             "stdout_log": str(log_path)}
    write_json(started_path, entry)
    record_event("command_started", entry)
    with log_path.open("wb") as log:
        environment = os.environ.copy()
        environment["PYTHONUNBUFFERED"] = "1"
        environment["TORCH_HOME"] = str(ARTIFACT_ROOT / "torch_cache")
        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        process = subprocess.Popen(command, cwd=REV_ROOT, stdout=log, stderr=subprocess.STDOUT,
                                   env=environment, creationflags=flags)
        while process.poll() is None:
            status = {"status": "running", "updated_utc": utc_now(), "pid": os.getpid(), "child_pid": process.pid,
                      "stage": stage, "command_index": index, "commands_in_stage": total,
                      "current": spec, "current_started_utc": began,
                      "test_status": "unlocked_fixed_T7_campaign" if stage == "T7_fixed_campaign" else "locked_pending_T7"}
            write_json(REV_ROOT / "PIPELINE_STATUS.json", status, overwrite=True)
            time.sleep(5)
    entry.update({"finished_utc": utc_now(), "exit_code": process.returncode,
                  "stdout_sha256": sha256_file(log_path)})
    if process.returncode != 0:
        entry["status"] = "failed"
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
        write_json(directory / f"{key}.failed.{timestamp}.json", entry)
        record_event("command_failed", entry)
        write_json(REV_ROOT / "PIPELINE_STATUS.json", entry, overwrite=True)
        update_manifest(entry)
        raise RuntimeError(f"Command failed; preserved log and seed: {log_path}")
    entry["status"] = "completed"
    write_json(done, entry)
    record_event("command_completed", entry)
    update_manifest(entry)
    return True


def run_pretest(max_commands: int | None = None) -> None:
    if max_commands is not None and max_commands < 1:
        raise ValueError("max-commands must be positive")
    verify_protocols()
    verify_implementation_lock()
    if (REV_ROOT / "T7_UNLOCK.json").exists():
        raise PermissionError("Training is forbidden after T7 unlock")
    executed = 0
    with process_lock():
        try:
            for stage in ("generators_audit_ratio", "detector_budgets_statistics"):
                specs = initial_jobs() if stage == "generators_audit_ratio" else budget_jobs()
                plan_path = REV_ROOT / "logs" / f"{stage}_plan.json"
                frozen_plan = {"stage": stage, "commands": specs,
                               "protocol_receipt_sha256": sha256_file(REV_ROOT / "protocol" / "hashes.json")}
                if plan_path.exists():
                    if load_json(plan_path) != frozen_plan:
                        raise RuntimeError("Managed command roster changed")
                else:
                    write_json(plan_path, frozen_plan)
                for index, spec in enumerate(specs, 1):
                    if execute_job(spec, index, len(specs), stage):
                        executed += 1
                    if max_commands is not None and executed >= max_commands:
                        write_json(REV_ROOT / "PIPELINE_STATUS.json", {"status": "paused_at_declared_command_boundary",
                                   "finished_utc": utc_now(), "completed_commands_this_invocation": executed,
                                   "test_status": "locked_pending_T7"}, overwrite=True)
                        return
            final = {"status": "pretest_runs_finished_acceptance_audit_required", "finished_utc": utc_now(),
                     "test_status": "locked_pending_T7", "next": "Verify outputs, figures and nine task receipts before global T7 freeze/unlock."}
            write_json(REV_ROOT / "PIPELINE_STATUS.json", final, overwrite=True)
            update_manifest(final)
            record_event("pretest_commands_finished", final)
        except Exception as exc:
            status = {"status": "stopped_requires_review", "updated_utc": utc_now(), "error": str(exc),
                      "test_status": "locked_pending_T7"}
            write_json(REV_ROOT / "PIPELINE_STATUS.json", status, overwrite=True)
            update_manifest(status)
            raise


def model_roster() -> dict:
    detector = importlib.import_module("rev02_detector")
    generators = [{"arm": arm, "seed": seed, "checkpoint": str(generator_run_dir(arm, seed) / "best.pt")}
                  for arm in BASE_ARMS for seed in GENERATOR_SEEDS]
    generators += [{"arm": arm, "seed": seed, "checkpoint": str(generator_run_dir(arm, seed) / "best.pt")}
                   for arm in SENSITIVITY_ARMS for seed in GENERATOR_SEEDS[:3]]
    detectors = []
    for row in budget_configurations(selected_ratios()):
        run_id = detector.budget_run_id(**row)
        completion_path = detector_run_dir(run_id) / "completion.json"
        detectors.append({**row, "run_id": run_id, "milestones": [896, 1792, 4480, 8960],
                          "completion_path": str(completion_path), "completion_sha256": sha256_file(completion_path)})
    for row in generators:
        row["checkpoint_sha256"] = sha256_file(row["checkpoint"])
    return {"generators": sorted(generators, key=lambda row: f"{row['arm']}_s{row['seed']}"),
            "detectors": sorted(detectors, key=lambda row: row["run_id"])}


def freeze_pretest() -> dict:
    verify_protocols()
    verify_implementation_lock()
    if (REV_ROOT / "T7_UNLOCK.json").exists():
        raise PermissionError("Cannot refreeze after test unlock")
    if (REV_ROOT / "PRETEST_FREEZE.json").exists():
        raise FileExistsError("Global pretest freeze already exists")
    evidences = {}
    for name in ("rev02_generator", "rev02_detector", "rev02_memorization", "rev02_statistics"):
        module = importlib.import_module(name)
        for task, evidence in module.pretest_evidence().items():
            if task in evidences:
                raise RuntimeError(f"Duplicate pretest task owner: {task}")
            evidences[task] = evidence
    if set(evidences) != set(REQUIRED_TASKS):
        raise RuntimeError(f"Incomplete pretest task evidence: {set(evidences)}")
    lock_sha = sha256_file(REV_ROOT / "IMPLEMENTATION_LOCK.json")
    roster = model_roster()
    if len(roster["generators"]) != 78 or len(roster["detectors"]) not in (45, 57):
        raise RuntimeError("Unexpected complete fixed model roster")
    prepared = {}
    hashes = {}
    last_end = None
    for task in REQUIRED_TASKS:
        evidence = evidences[task]
        if (evidence.get("status") != "pretest_complete" or not evidence.get("integrity_checks_passed")
            or evidence.get("expected_units", 0) <= 0 or evidence.get("completed_units") != evidence["expected_units"]
            or evidence.get("missing_units") != []):
            raise RuntimeError(f"Incomplete or failed pretest evidence: {task}")
        paths = evidence.pop("output_paths")
        if not paths:
            raise RuntimeError(f"No verified outputs for {task}")
        outputs = {str(Path(path).resolve()): sha256_file(path) for path in paths}
        receipt = {**evidence, "task_id": task, "receipt_created_utc": utc_now(),
                   "test_status": "locked_pending_T7", "output_hashes": outputs,
                   "implementation_lock_sha256": lock_sha,
                   "task_protocol_sha256": sha256_file(REV_ROOT / "protocol" / f"REV02_{task}.yaml")}
        path = task_dir(task) / "PRETEST_COMPLETE.json"
        if path.exists():
            existing = load_json(path)
            for key in receipt:
                if key != "receipt_created_utc" and existing.get(key) != receipt[key]:
                    raise RuntimeError(f"Existing partial freeze receipt is not identical: {task}/{key}")
            receipt = existing
        prepared[path] = receipt
        ended = datetime.fromisoformat(receipt["last_training_validation_end_utc"])
        last_end = ended if last_end is None else max(last_end, ended)
    # Validate every component, source/output hash, end time and model roster
    # before the first write. A partial I/O interruption can reuse only byte-
    # equivalent receipts; it cannot replace results or choices.
    for path, receipt in prepared.items():
        if not path.exists():
            write_json(path, receipt)
        hashes[str(path.relative_to(REV_ROOT)).replace("\\", "/")] = sha256_file(path)
    freeze = {"status": "pretest_complete", "created_utc": utc_now(), "protocol_commit": PROTOCOL_COMMIT,
              "protocol_receipt_sha256": sha256_file(REV_ROOT / "protocol" / "hashes.json"),
              "implementation_lock_sha256": lock_sha, "receipt_hashes": hashes,
              "last_training_validation_end_utc": last_end.isoformat(), "test_roster": roster,
              "test_status": "locked_pending_T7"}
    validate_pretest_freeze(freeze)
    write_json(REV_ROOT / "PRETEST_FREEZE.json", freeze)
    record_event("pretest_globally_frozen", {"sha256": sha256_file(REV_ROOT / "PRETEST_FREEZE.json"),
                                             "tasks": list(REQUIRED_TASKS), "test_remains_locked": True})
    return freeze


def unlock(audit_report: Path) -> None:
    freeze_path = REV_ROOT / "PRETEST_FREEZE.json"
    freeze = load_json(freeze_path)
    validate_pretest_freeze(freeze)
    audit = load_json(audit_report)
    if (audit.get("status") != "PASS" or audit.get("pretest_freeze_sha256") != sha256_file(freeze_path)
        or audit.get("failed_checks") != [] or not audit.get("checks_passed", 0)):
        raise PermissionError("A matching, nonempty independent pretest acceptance audit is required")
    if any(event["event"] == "test_unlocked" for event in verify_event_chain()):
        raise PermissionError("A T7 campaign has already been unlocked")
    timestamp = utc_now()
    if datetime.fromisoformat(timestamp) <= datetime.fromisoformat(freeze["last_training_validation_end_utc"]):
        raise PermissionError("Unlock must occur after every pretest training/validation end")
    receipt = {"status": "unlocked", "campaign_count": 1, "campaign_id": "LPBF_REV02_T7_v1",
               "unlocked_utc": timestamp, "pretest_freeze_sha256": sha256_file(freeze_path),
               "acceptance_audit_path": str(audit_report), "acceptance_audit_sha256": sha256_file(audit_report)}
    write_json(REV_ROOT / "T7_UNLOCK.json", receipt)
    record_event("test_unlocked", receipt)


def test_jobs(roster: dict) -> list[dict]:
    plan = []
    for row in sorted(roster["generators"], key=lambda row: f"{row['arm']}_s{row['seed']}"):
        plan.append(job("rev02_generator", "evaluate", "--arm", row["arm"], "--seed", row["seed"], "--split", "test"))
    plan += [job("rev02_generator", "controls", "--split", "test"), job("rev02_generator", "summarize", "--split", "all")]
    for row in sorted(roster["detectors"], key=lambda row: row["run_id"]):
        plan.append(job("rev02_detector", "evaluate", "--run-id", row["run_id"], "--split", "test"))
    plan += [job("rev02_detector", "summarize"), job("rev02_statistics", "summarize", "--split", "all")]
    return plan


def test_campaign() -> None:
    require_test_unlocked()
    freeze = load_json(REV_ROOT / "PRETEST_FREEZE.json")
    plan = test_jobs(freeze["test_roster"])
    with process_lock():
        for index, spec in enumerate(plan, 1):
            execute_job(spec, index, len(plan), "T7_fixed_campaign")
    record_event("T7_campaign_completed", {"commands": len(plan), "no_test_selection": True})
    write_json(REV_ROOT / "PIPELINE_STATUS.json", {"status": "T7_completed_manuscript_update_pending", "finished_utc": utc_now()}, overwrite=True)


def main(argv=None):
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("plan")
    sub.add_parser("status")
    run = sub.add_parser("run")
    run.add_argument("--max-commands", type=int)
    sub.add_parser("pretest-freeze")
    unlocking = sub.add_parser("unlock")
    unlocking.add_argument("--audit-report", type=Path, required=True)
    sub.add_parser("test-campaign")
    args = parser.parse_args(argv)
    if args.action == "plan":
        print(json.dumps({"initial_stage_commands": len(initial_jobs()), "initial_jobs": initial_jobs(),
                          "budget_trajectory_counts": {"one_selected_ratio": 45, "two_selected_ratios": 57},
                          "test": "separate_global_gate_after_pretest_acceptance"}, indent=2))
    elif args.action == "status":
        print(json.dumps(load_json(REV_ROOT / "PIPELINE_STATUS.json") if (REV_ROOT / "PIPELINE_STATUS.json").exists() else {"status": "not_started"}, indent=2))
    elif args.action == "run": run_pretest(args.max_commands)
    elif args.action == "pretest-freeze": print(json.dumps(freeze_pretest(), indent=2))
    elif args.action == "unlock": unlock(args.audit_report)
    elif args.action == "test-campaign": test_campaign()


if __name__ == "__main__":
    main()
