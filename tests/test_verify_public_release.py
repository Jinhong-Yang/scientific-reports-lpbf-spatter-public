from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("verify_release", ROOT / "scripts/verify_public_release.py")
assert SPEC and SPEC.loader
VERIFY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFY)


def release_fixture() -> dict:
    return {
        "url": f"{VERIFY.REPOSITORY_URL}/releases/tag/v1.0.0",
        "tagName": "v1.0.0",
        "isDraft": False,
        "isPrerelease": False,
        "isImmutable": False,
        "assets": [{"name": "artifact.zip"}],
    }


def test_public_release_metadata_passes_only_complete_public_state() -> None:
    issues = VERIFY.validate_metadata(
        {"url": VERIFY.REPOSITORY_URL, "visibility": "PUBLIC"},
        release_fixture(), "v1.0.0", {"artifact.zip"},
        {"url": VERIFY.DEVELOPMENT_REPOSITORY_URL, "visibility": "PRIVATE"},
    )
    assert issues == []


def test_public_release_metadata_rejects_draft_and_missing_asset() -> None:
    release = release_fixture()
    release["isDraft"] = True
    issues = VERIFY.validate_metadata(
        {"url": VERIFY.REPOSITORY_URL, "visibility": "PRIVATE"},
        release, "v1.0.0", {"artifact.zip", "missing.pdf"},
    )
    assert any("not public" in issue for issue in issues)
    assert any("still a draft" in issue for issue in issues)
    assert any("missing release assets" in issue for issue in issues)


def test_public_release_rejects_exposed_development_repository() -> None:
    issues = VERIFY.validate_metadata(
        {"url": VERIFY.REPOSITORY_URL, "visibility": "PUBLIC"},
        release_fixture(), "v1.0.0", {"artifact.zip"},
        {"url": VERIFY.DEVELOPMENT_REPOSITORY_URL, "visibility": "PUBLIC"},
    )
    assert any("development repository is not private" in issue for issue in issues)
