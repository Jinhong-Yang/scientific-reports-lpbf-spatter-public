from __future__ import annotations

import importlib.util
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]


def load_module():
    spec = importlib.util.spec_from_file_location("build_release_archive", ROOT / "scripts" / "build_release_archive.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RUNNER = load_module()


def test_release_excludes_restricted_roots_and_binaries() -> None:
    assert not RUNNER.included("historical/source.bmp")
    assert not RUNNER.included("inputs/private.md")
    assert not RUNNER.included("runs/new_study/checkpoints/model.pt")
    assert not RUNNER.included("evidence/stronger_generator/cache/features.npz")
    assert not RUNNER.included("evidence/human_review/reviewer_sheet.csv")
    assert not RUNNER.included("docs/W04_REVIEWER_HANDOFF.md")
    assert not RUNNER.included("docs/W08_APPROVAL_FORM.md")
    assert not RUNNER.included("docs/SUBMISSION_COMPLETION_CRITERIA.md")
    assert not RUNNER.included("docs/PRETEST_PIPELINE_RECOVERY_LEDGER.md")
    assert not RUNNER.included("docs/MISSING_INPUTS.md")
    assert not RUNNER.included("src/evaluation/validate_ratings.py")
    assert RUNNER.included("docs/PROTOCOL_INCIDENT_001_TEST_IDENTIFIER_DISPLAY.md")
    assert RUNNER.included("src/analysis/run_w17_statistics.py")
    assert RUNNER.included("manuscript/main.tex")
    assert RUNNER.included("scripts/build_manuscript.ps1")
    assert RUNNER.included("requirements-reporting.txt")


def test_release_audit_detects_absolute_paths_and_tokens() -> None:
    assert RUNNER.audit_payload("x.md", b"C:\\Users\\name\\file")
    assert RUNNER.audit_payload("notebook.ipynb", b"C:\\Users\\name\\file")
    assert RUNNER.audit_payload("figure.pdf", b"metadata C:\\Codex Research\\project")
    token = b"github_" + b"pat_" + b"abcdefghijklmnopqrstuvwxyz123456"
    assert RUNNER.audit_payload("weights.npz", token)
    assert RUNNER.audit_payload(
        "figure.pdf", b"Bear" + b"er " + b"abcdefghijklmnopqrstuvwxyz",
    )
    assert RUNNER.audit_payload(
        "source.tex", b"-----BEGIN " + b"PRIVATE KEY-----",
    )
    assert not RUNNER.audit_payload("x.md", b"safe relative/path")


def test_release_manifest_source_commit_is_resolvable() -> None:
    assert re.fullmatch(r"[a-f0-9]{40}", RUNNER.source_study_commit())


def test_release_source_state_requires_clean_pushed_commit() -> None:
    commit = "a" * 40
    assert RUNNER.source_state_issues("", commit, commit) == []
    assert RUNNER.source_state_issues(" M manuscript/main.tex", commit, commit)
    assert RUNNER.source_state_issues("", commit, "b" * 40)
    assert RUNNER.source_state_issues("", "not-a-commit", "not-a-commit")
