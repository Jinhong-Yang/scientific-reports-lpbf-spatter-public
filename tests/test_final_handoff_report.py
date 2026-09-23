from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("final_handoff", ROOT / "scripts/build_final_handoff_report.py")
assert SPEC and SPEC.loader
HANDOFF = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(HANDOFF)


def test_public_urls_must_be_canonical_and_version_bound() -> None:
    assert HANDOFF.validate_public_url(
        "https://github.com/example/repository", "repository_url"
    ) == "https://github.com/example/repository"
    try:
        HANDOFF.validate_public_url("D:/private/repository", "repository_url")
    except ValueError:
        pass
    else:
        raise AssertionError("local path was accepted as a public URL")


def test_handoff_report_keeps_scope_and_author_actions_explicit() -> None:
    payload = {
        "generated_at": "2026-09-14T00:00:00+00:00",
        "repository_url": "https://github.com/example/repository",
        "release_url": "https://github.com/example/repository/releases/tag/v1.0.0",
        "version": "v1.0.0",
        "git_commit": "a" * 40,
        "protocol_sha256": "b" * 64,
        "artifacts": [{"path": "output/example.zip", "bytes": 12, "sha256": "c" * 64}],
    }
    report = HANDOFF.render_report(payload)
    assert "READY_WITH_SCOPE_LIMITATION" in report
    assert "MINOR_REVISION_OR_BETTER_INTERNAL_ASSESSMENT" in report
    assert "inherited_bbox_region" in report
    assert "no external\n  validation claim" in report
    assert "authors must still\nconfirm" in report


def test_handoff_builder_requires_verified_publication_receipt() -> None:
    source = (ROOT / "scripts/build_final_handoff_report.py").read_text(encoding="utf-8")
    assert "PUBLICATION_RECEIPT.json" in source
    assert "Public release download verification has not passed" in source
    assert "publication.get(\"source_study_commit\") != study_commit" in source
