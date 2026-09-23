"""Publish the audited allow-listed archive as a new clean GitHub repository."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
import tempfile
from typing import Any
import zipfile


ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = "Jinhong-Yang/scientific-reports-lpbf-spatter-public"
DEVELOPMENT_REPOSITORY = "Jinhong-Yang/scientific-reports-lpbf-spatter"
PYTHON = ROOT / ".venv-reporting" / "Scripts" / "python.exe"
PUBLIC_TESTS = (
    "tests/test_submission_artifacts.py",
    "tests/test_submission_audit.py",
    "tests/test_release_archive.py",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(command: list[str], cwd: Path, capture: bool = False) -> str:
    completed = subprocess.run(
        command, cwd=cwd, check=True, text=True,
        capture_output=capture,
    )
    return completed.stdout if capture else ""


def safe_member(name: str) -> PurePosixPath:
    if "\\" in name:
        raise RuntimeError(f"Backslash path rejected in archive: {name}")
    path = PurePosixPath(name)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise RuntimeError(f"Unsafe archive path: {name}")
    if path.parts[0] == ".git":
        raise RuntimeError("Git metadata is forbidden in the clean archive")
    return path


def extract_verified(archive_path: Path, destination: Path) -> dict[str, Any]:
    seen: set[str] = set()
    with zipfile.ZipFile(archive_path) as archive:
        for info in archive.infolist():
            path = safe_member(info.filename.rstrip("/"))
            normalized = path.as_posix()
            if normalized in seen:
                raise RuntimeError(f"Duplicate archive entry: {normalized}")
            seen.add(normalized)
            target = destination.joinpath(*path.parts)
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(info))

    manifest_path = destination / "release/PUBLIC_RELEASE_MANIFEST.json"
    checksum_path = destination / "release/SHA256SUMS.txt"
    if not manifest_path.is_file() or not checksum_path.is_file():
        raise RuntimeError("Archive is missing its public manifest or checksum ledger")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_entries = {item["path"]: item["sha256"] for item in manifest.get("files", [])}
    ledger_entries: dict[str, str] = {}
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        digest, separator, relative = line.partition("  ")
        if not separator or relative in ledger_entries:
            raise RuntimeError(f"Malformed or duplicate checksum line: {line}")
        ledger_entries[relative] = digest
    if ledger_entries != expected_entries:
        raise RuntimeError("Checksum ledger and public manifest do not name the same payload")
    mismatches = [
        relative for relative, expected in ledger_entries.items()
        if not (destination / relative).is_file()
        or sha256_file(destination / relative) != expected
    ]
    if mismatches:
        raise RuntimeError("Extracted checksum mismatch:\n" + "\n".join(mismatches))
    actual_files = {
        str(path.relative_to(destination)).replace("\\", "/")
        for path in destination.rglob("*") if path.is_file()
    }
    expected_files = set(expected_entries) | {
        "release/PUBLIC_RELEASE_MANIFEST.json", "release/SHA256SUMS.txt",
    }
    if actual_files != expected_files:
        raise RuntimeError("Extracted archive contains unmanifested files")
    return manifest


def tracked_files(repository: Path) -> set[str]:
    output = subprocess.check_output(["git", "ls-files", "-z"], cwd=repository)
    return {item.decode("utf-8") for item in output.split(b"\0") if item}


def local_assets(version: str) -> list[Path]:
    package_path = ROOT / "output/submission/SUBMISSION_PACKAGE_MANIFEST.json"
    package = json.loads(package_path.read_text(encoding="utf-8"))
    if package.get("status") != "PASS" or package.get("version") != version:
        raise RuntimeError("Submission package manifest has not passed for this version")
    pdfs = {
        "manuscript": ROOT / "output/pdf/Scientific_Reports_LPBF_Manuscript.pdf",
        "supplement": ROOT / "output/pdf/Scientific_Reports_LPBF_Supplementary_Information.pdf",
        "cover_letter": ROOT / "output/pdf/Scientific_Reports_LPBF_Cover_Letter.pdf",
    }
    package_entries = package.get("packages", {})
    package_paths = [ROOT / item["path"] for item in package_entries.values()]
    paths = [
        *pdfs.values(), *package_paths,
        ROOT / f"release/Scientific_Reports_LPBF_Reproducibility_{version}.zip",
    ]
    if len(paths) != 7 or any(not path.is_file() for path in paths):
        raise RuntimeError("Exactly seven final local release assets are required")
    stale_packages = [
        name for name, item in package_entries.items()
        if sha256_file(ROOT / item["path"]) != item.get("sha256")
    ]
    if stale_packages:
        raise RuntimeError("Submission package hash mismatch: " + ", ".join(stale_packages))
    visual_path = ROOT / "evidence/manuscript/VISUAL_QA.json"
    if not visual_path.is_file():
        raise RuntimeError("Hash-bound visual QA receipt is missing")
    visual = json.loads(visual_path.read_text(encoding="utf-8"))
    actual_pdf_hashes = {name: sha256_file(path) for name, path in pdfs.items()}
    if visual.get("status") != "PASS" or visual.get("reviewed_pdf_sha256") != actual_pdf_hashes:
        raise RuntimeError("Final PDF bytes do not match the passed visual QA receipt")
    return paths


def ensure_candidate_absent() -> None:
    completed = subprocess.run(
        ["gh", "repo", "view", REPOSITORY, "--json", "url"],
        cwd=ROOT, text=True, capture_output=True,
    )
    if completed.returncode == 0:
        raise RuntimeError(f"Publication repository already exists: {REPOSITORY}")


def ensure_development_private() -> None:
    completed = subprocess.run(
        ["gh", "repo", "view", DEVELOPMENT_REPOSITORY, "--json", "url,visibility"],
        cwd=ROOT, text=True, capture_output=True,
    )
    if completed.returncode != 0:
        raise RuntimeError("Could not verify the development repository visibility")
    metadata = json.loads(completed.stdout)
    if metadata.get("visibility") != "PRIVATE":
        raise RuntimeError(
            f"Development repository must remain private: {metadata.get('visibility')}"
        )


def publish(version: str) -> dict[str, Any]:
    if not PYTHON.is_file():
        raise RuntimeError("Reporting environment is missing")
    review = json.loads((ROOT / "evidence/manuscript/W20_INTERNAL_REVIEW.json").read_text(encoding="utf-8"))
    if review.get("status") != "PASS":
        raise RuntimeError("W20 internal review has not passed")
    archive_path = ROOT / f"release/Scientific_Reports_LPBF_Reproducibility_{version}.zip"
    local_manifest = json.loads((ROOT / "release/PUBLIC_RELEASE_MANIFEST.json").read_text(encoding="utf-8"))
    if local_manifest.get("status") != "AUDIT_PASS" or local_manifest.get("version") != version:
        raise RuntimeError("Final W19 release manifest has not passed")
    if local_manifest.get("archive_sha256") != sha256_file(archive_path):
        raise RuntimeError("Final W19 archive hash does not match its manifest")
    source_commit = run(["git", "rev-parse", "HEAD"], ROOT, capture=True).strip()
    if local_manifest.get("source_study_commit") != source_commit:
        raise RuntimeError("Release archive was not built from the current source-study commit")
    source_status = run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"], ROOT, capture=True,
    ).strip()
    upstream_commit = run(["git", "rev-parse", "@{upstream}"], ROOT, capture=True).strip()
    if source_status or upstream_commit != source_commit:
        raise RuntimeError("Publication requires a clean source tree at its pushed upstream commit")
    assets = local_assets(version)
    ensure_development_private()
    ensure_candidate_absent()

    with tempfile.TemporaryDirectory(prefix="lpbf-clean-public-") as directory:
        clean = Path(directory)
        extracted = extract_verified(archive_path, clean)
        if extracted.get("publication_topology") != "clean_public_repository_from_allowlisted_archive":
            raise RuntimeError("Clean-public topology is not recorded in the archive")
        run(["git", "init", "-b", "main"], clean)
        author_name = run(["git", "config", "--get", "user.name"], ROOT, capture=True).strip()
        author_email = run(["git", "config", "--get", "user.email"], ROOT, capture=True).strip()
        if not author_name or not author_email:
            raise RuntimeError("Development repository must define a local Git author identity for clean-public publication")
        run(["git", "config", "user.name", author_name], clean)
        run(["git", "config", "user.email", author_email], clean)
        run(["git", "add", "-f", "--all"], clean)
        expected_tracked = {item["path"] for item in extracted["files"]} | {
            "release/PUBLIC_RELEASE_MANIFEST.json", "release/SHA256SUMS.txt",
        }
        actual_tracked = tracked_files(clean)
        if actual_tracked != expected_tracked:
            raise RuntimeError("Clean Git index differs from the audited archive manifest")
        run(["git", "commit", "-m", f"Publish Scientific Reports reproducibility package {version}"], clean)
        public_commit = run(["git", "rev-parse", "HEAD"], clean, capture=True).strip()
        run([str(PYTHON), "-m", "pytest", "-q", *PUBLIC_TESTS], clean)
        run([str(PYTHON), "scripts/verify_sha256s.py", "release/SHA256SUMS.txt"], clean)
        run([
            "gh", "repo", "create", REPOSITORY, "--public", "--source", str(clean),
            "--remote", "origin", "--push",
            "--description", "Source-data-free reproducibility package for the LPBF synthesis and detection study",
        ], clean)
        run(["git", "tag", "-a", version, "-m", f"Scientific Reports submission package {version}"], clean)
        run(["git", "push", "origin", version], clean)
        notes = clean / "docs/GITHUB_RELEASE_NOTES.md"
        run([
            "gh", "release", "create", version, *[str(path) for path in assets],
            "--repo", REPOSITORY, "--verify-tag",
            "--title", f"Scientific Reports LPBF submission package {version}",
            "--notes-file", str(notes),
        ], clean)

    ensure_development_private()
    run([str(PYTHON), "scripts/verify_public_release.py", "--version", version, "--write"], ROOT)
    receipt = json.loads((ROOT / "evidence/release/PUBLICATION_RECEIPT.json").read_text(encoding="utf-8"))
    if receipt.get("status") != "PASS":
        raise RuntimeError("Public release was created but fresh-download verification failed")
    return {
        "status": "PASS", "repository": REPOSITORY, "version": version,
        "source_study_commit": source_commit, "public_tag_commit": public_commit,
        "publication_receipt": "evidence/release/PUBLICATION_RECEIPT.json",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", default="v1.0.0")
    parser.add_argument("--execute", action="store_true", required=True)
    args = parser.parse_args()
    result = publish(args.version)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
