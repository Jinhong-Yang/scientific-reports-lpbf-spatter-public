"""Test the public allowlist in a fresh directory outside the source checkout.

This checks the current reviewed Git index's working files, not a released ZIP.
The final published ZIP must still pass extract_verified and download checks.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from build_release_archive import ROOT, included, tracked_files, audit_payload, source_study_commit


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    candidate = Path(tempfile.mkdtemp(prefix="lpbf-isolated-reporting-"))
    selected = [p for p in tracked_files() if included(p)]
    for relative in selected:
        payload = (ROOT / relative).read_bytes()
        issues = audit_payload(relative, payload)
        if issues:
            raise RuntimeError(issues)
        target = candidate / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
    # A provenance receipt lets archive tests run without borrowing parent Git.
    provenance = candidate / "release/PUBLIC_RELEASE_MANIFEST.json"
    provenance.parent.mkdir(exist_ok=True)
    provenance.write_text(json.dumps({"status": "CANDIDATE_CHECK_ONLY", "source_study_commit": source_study_commit()}))
    commands = [
        [sys.executable, "src/reporting/build_submission_artifacts.py", "--preflight"],
        [sys.executable, "src/reporting/build_submission_artifacts.py", "--build"],
        [sys.executable, "-m", "pytest", "-q", "tests/test_submission_artifacts.py",
         "tests/test_submission_audit.py", "tests/test_release_archive.py", "tests/test_submission_packages.py"],
    ]
    for command in commands:
        result = subprocess.run(command, cwd=candidate, capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(f"Isolated command failed: {command[1:]}\n{result.stdout}\n{result.stderr}")
    compare = sorted([*ROOT.glob("manuscript/generated/*.tex"),
                      *ROOT.glob("evidence/figure_source_data/*.csv"),
                      *ROOT.glob("manuscript/figures/*.png")])
    mismatches = [p.relative_to(ROOT).as_posix() for p in compare
                  if digest(p) != digest(candidate / p.relative_to(ROOT))]
    if mismatches:
        raise RuntimeError(f"Isolated outputs differ: {mismatches}")
    binding = json.loads((ROOT / "evidence/manuscript/MANUSCRIPT_NUMERIC_MAP.json").read_text())["source_hashes"]
    binding["src/reporting/build_submission_artifacts.py"] = digest(ROOT / "src/reporting/build_submission_artifacts.py")
    value = {"status": "PASS", "scope": "current_allowlisted_candidate_not_yet_published_archive",
             "outside_source_checkout": True, "source_pixels_available": False,
             "predictions_available": False, "retraining_attempted": False,
             "preflight_build_tests_passed": True, "identical_generated_outputs": len(compare),
             "bound_input_hashes": binding}
    (ROOT / "evidence/manuscript/ISOLATED_REBUILD.json").write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "PASS", "identical_generated_outputs": len(compare), "public_tests_passed": True}))


if __name__ == "__main__":
    main()
