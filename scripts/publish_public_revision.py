"""Publish a new version on the existing allowlisted public history, never force.

Only an audited archive is copied into a fresh clone of the public repository.
Private development Git history is never added as a remote or pushed there.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

from build_release_archive import assert_source_tree_published
from publish_clean_public_repo import (ROOT, REPOSITORY, PYTHON, PUBLIC_TESTS, run,
    extract_verified, local_assets, ensure_development_private, tracked_files, sha256_file)


def publish(version: str) -> None:
    if version == "v1.0.0":
        raise RuntimeError("The initial release must be preserved")
    assert_source_tree_published()
    ensure_development_private()
    manifest = json.loads((ROOT / "release/PUBLIC_RELEASE_MANIFEST.json").read_text())
    archive = ROOT / f"release/Scientific_Reports_LPBF_Reproducibility_{version}.zip"
    if manifest.get("status") != "AUDIT_PASS" or manifest.get("version") != version:
        raise RuntimeError("Audited archive/version mismatch")
    if manifest["archive_sha256"] != sha256_file(archive):
        raise RuntimeError("Archive digest mismatch")
    head = run(["git", "rev-parse", "HEAD"], ROOT, capture=True).strip()
    if manifest["source_study_commit"] != head:
        raise RuntimeError("Archive source commit is stale")
    assets = local_assets(version)
    releases = json.loads(run(["gh", "api", f"repos/{REPOSITORY}/releases?per_page=100"], ROOT, capture=True))
    if any(r["tag_name"] == version for r in releases):
        raise RuntimeError("Version already exists; it will not be overwritten")
    if not any(r["tag_name"] == "v1.0.0" for r in releases):
        raise RuntimeError("Expected preserved initial release missing")
    workspace = Path(tempfile.mkdtemp(prefix="lpbf-public-revision-"))
    candidate = workspace / "candidate"
    candidate.mkdir()
    extracted = extract_verified(archive, candidate)
    run([str(PYTHON), "-m", "pytest", "-q", *PUBLIC_TESTS,
         "tests/test_submission_packages.py", "tests/test_secondary_reporting.py"], candidate)
    run([str(PYTHON), "scripts/verify_sha256s.py", "release/SHA256SUMS.txt"], candidate)
    # Exercise the actual archive builder away from the manuscript checkout.
    run([str(PYTHON), "src/reporting/build_submission_artifacts.py", "--build"], candidate, capture=True)
    for item in extracted["files"]:
        name = item["path"]
        if name.startswith(("manuscript/generated/", "evidence/figure_source_data/")):
            if sha256_file(candidate / name) != item["sha256"]:
                raise RuntimeError(f"Archive regeneration changed numerical/text output: {name}")
    public = workspace / "public"
    run(["git", "clone", f"https://github.com/{REPOSITORY}.git", str(public)], workspace)
    initial_tag = run(["git", "rev-parse", "v1.0.0^{commit}"], public, capture=True).strip()
    if subprocess.run(["git", "show-ref", "--verify", f"refs/tags/{version}"], cwd=public,
                      capture_output=True).returncode == 0:
        raise RuntimeError("Version tag already exists")
    expected = {item["path"] for item in extracted["files"]} | {
        "release/PUBLIC_RELEASE_MANIFEST.json", "release/SHA256SUMS.txt"}
    obsolete = tracked_files(public) - expected
    # Delete only individually verified tracked files in this disposable clone.
    for relative in obsolete:
        target = (public / relative).resolve()
        if not target.is_relative_to(public.resolve()) or ".git" in target.relative_to(public.resolve()).parts:
            raise RuntimeError("Unsafe obsolete public path")
        target.unlink()
    pristine = workspace / "pristine"
    pristine.mkdir()
    extract_verified(archive, pristine)
    for relative in expected:
        target = public / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(pristine / relative, target)
    for option in ("user.name", "user.email"):
        run(["git", "config", option, run(["git", "config", "--get", option], ROOT, capture=True).strip()], public)
    paths = sorted(expected | obsolete)
    for start in range(0, len(paths), 80):
        run(["git", "add", "-f", "--", *paths[start:start+80]], public)
    if tracked_files(public) != expected:
        raise RuntimeError("Public index differs from allowlist")
    run(["git", "commit", "-m", f"Revise manuscript and public reconstruction {version}"], public)
    run(["git", "push", "origin", "HEAD:main"], public)
    run(["git", "tag", "-a", version, "-m", f"Scientific Reports revision {version}"], public)
    run(["git", "push", "origin", version], public)
    if run(["git", "rev-parse", "v1.0.0^{commit}"], public, capture=True).strip() != initial_tag:
        raise RuntimeError("Initial tag changed unexpectedly")
    run(["gh", "release", "create", version, *map(str, assets), "--repo", REPOSITORY,
         "--verify-tag", "--title", f"Scientific Reports LPBF submission draft {version}",
         "--notes-file", str(public / "docs/GITHUB_RELEASE_NOTES.md")], public)
    run([str(PYTHON), "scripts/verify_public_release.py", "--version", version, "--write"], ROOT)
    print(json.dumps({"status": "PASS", "version": version, "v1.0.0_preserved": True,
                      "obsolete_paths_removed_only_from_new_snapshot": sorted(obsolete)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True)
    parser.add_argument("--execute", required=True, action="store_true")
    publish(parser.parse_args().version)
