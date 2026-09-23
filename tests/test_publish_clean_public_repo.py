from __future__ import annotations

import importlib.util
import hashlib
import json
from pathlib import Path
import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("clean_publish", ROOT / "scripts/publish_clean_public_repo.py")
assert SPEC and SPEC.loader
PUBLISH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PUBLISH)


def test_safe_member_rejects_traversal_and_git_metadata() -> None:
    for name in ("../secret", "/absolute", ".git/config", "folder\\file"):
        with pytest.raises(RuntimeError):
            PUBLISH.safe_member(name)


def test_safe_member_accepts_release_paths() -> None:
    assert PUBLISH.safe_member("docs/README.md").as_posix() == "docs/README.md"
    assert PUBLISH.safe_member("release/SHA256SUMS.txt").as_posix() == "release/SHA256SUMS.txt"


def test_development_repository_must_remain_private(monkeypatch: pytest.MonkeyPatch) -> None:
    class Result:
        returncode = 0
        stdout = json.dumps({"visibility": "PUBLIC"})

    monkeypatch.setattr(PUBLISH.subprocess, "run", lambda *args, **kwargs: Result())
    with pytest.raises(RuntimeError, match="must remain private"):
        PUBLISH.ensure_development_private()


def test_local_assets_are_bound_to_package_and_visual_hashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(PUBLISH, "ROOT", tmp_path)
    pdfs = {
        "manuscript": tmp_path / "output/pdf/Scientific_Reports_LPBF_Manuscript.pdf",
        "supplement": tmp_path / "output/pdf/Scientific_Reports_LPBF_Supplementary_Information.pdf",
        "cover_letter": tmp_path / "output/pdf/Scientific_Reports_LPBF_Cover_Letter.pdf",
    }
    for name, path in pdfs.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"final {name}".encode())
    packages = {}
    for name in ("overleaf", "numerical_evidence", "journal_upload"):
        path = tmp_path / f"output/submission/{name}.zip"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(name.encode())
        packages[name] = {
            "path": str(path.relative_to(tmp_path)).replace("\\", "/"),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    manifest = tmp_path / "output/submission/SUBMISSION_PACKAGE_MANIFEST.json"
    manifest.write_text(json.dumps({
        "status": "PASS", "version": "v1.0.0", "packages": packages,
    }), encoding="utf-8")
    release = tmp_path / "release/Scientific_Reports_LPBF_Reproducibility_v1.0.0.zip"
    release.parent.mkdir(parents=True, exist_ok=True)
    release.write_bytes(b"release")
    visual = tmp_path / "evidence/manuscript/VISUAL_QA.json"
    visual.parent.mkdir(parents=True, exist_ok=True)
    visual.write_text(json.dumps({
        "status": "PASS",
        "reviewed_pdf_sha256": {
            name: hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in pdfs.items()
        },
    }), encoding="utf-8")
    assert len(PUBLISH.local_assets("v1.0.0")) == 7
    pdfs["manuscript"].write_bytes(b"changed after visual review")
    with pytest.raises(RuntimeError, match="visual QA"):
        PUBLISH.local_assets("v1.0.0")
