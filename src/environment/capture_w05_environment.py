#!/usr/bin/env python3
"""Capture the fresh W05 runtime and verify the pinned core packages."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
REQUIREMENTS = ROOT / "configs" / "w05_core_requirements.txt"
OUTPUT = ROOT / "evidence" / "environment"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_requirements(path: Path) -> dict[str, str]:
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, version = line.split("==", 1)
        result[name.lower()] = version
    return result


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    expected = parse_requirements(REQUIREMENTS)
    actual = {}
    mismatches = []
    for name, version in expected.items():
        try:
            installed = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            installed = None
        actual[name] = installed
        if installed != version:
            mismatches.append({"package": name, "expected": version, "actual": installed})

    freeze = subprocess.run(
        [sys.executable, "-m", "pip", "freeze", "--all"],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout
    freeze_path = OUTPUT / "w05_pip_freeze.txt"
    freeze_path.write_text(freeze, encoding="utf-8")

    cuda_available = torch.cuda.is_available()
    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if not mismatches and cuda_available else "FAILED",
        "fresh_environment": True,
        "executable": sys.executable,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "requirements_sha256": sha256(REQUIREMENTS),
        "pip_freeze_sha256": sha256(freeze_path),
        "expected_core_versions": expected,
        "installed_core_versions": actual,
        "version_mismatches": mismatches,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": cuda_available,
        "cuda_device_count": torch.cuda.device_count() if cuda_available else 0,
        "cuda_device": torch.cuda.get_device_name(0) if cuda_available else None,
        "full_training_authorized": False,
    }
    (OUTPUT / "runtime_manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))
    return 0 if payload["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
