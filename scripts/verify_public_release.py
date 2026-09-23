"""Verify the public GitHub release by re-downloading every required asset."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = "Jinhong-Yang/scientific-reports-lpbf-spatter-public"
REPOSITORY_URL = f"https://github.com/{REPOSITORY}"
DEVELOPMENT_REPOSITORY = "Jinhong-Yang/scientific-reports-lpbf-spatter"
DEVELOPMENT_REPOSITORY_URL = f"https://github.com/{DEVELOPMENT_REPOSITORY}"
OUTPUT = ROOT / "evidence" / "release" / "PUBLICATION_RECEIPT.json"
PDF_PATHS = (
    ROOT / "output/pdf/Scientific_Reports_LPBF_Manuscript.pdf",
    ROOT / "output/pdf/Scientific_Reports_LPBF_Supplementary_Information.pdf",
    ROOT / "output/pdf/Scientific_Reports_LPBF_Cover_Letter.pdf",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_gh(arguments: list[str]) -> str:
    completed = subprocess.run(
        ["gh", *arguments], cwd=ROOT, check=True,
        capture_output=True, text=True,
    )
    return completed.stdout


def current_source_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip()


def local_assets(version: str) -> dict[str, Path]:
    package_manifest = ROOT / "output/submission/SUBMISSION_PACKAGE_MANIFEST.json"
    if not package_manifest.is_file():
        raise RuntimeError("Submission package manifest is missing")
    package = json.loads(package_manifest.read_text(encoding="utf-8"))
    if package.get("status") != "PASS" or package.get("version") != version:
        raise RuntimeError("Submission package manifest has not passed for the requested version")
    paths = [*PDF_PATHS]
    paths.extend(ROOT / item["path"] for item in package.get("packages", {}).values())
    paths.append(ROOT / f"release/Scientific_Reports_LPBF_Reproducibility_{version}.zip")
    missing = [str(path.relative_to(ROOT)) for path in paths if not path.is_file()]
    if missing:
        raise RuntimeError("Missing local release assets:\n" + "\n".join(missing))
    result = {path.name: path for path in paths}
    if len(result) != 7:
        raise RuntimeError(f"Expected seven uniquely named release assets; found {len(result)}")
    return result


def validate_metadata(repository: dict[str, Any], release: dict[str, Any], version: str,
                      required_names: set[str],
                      development_repository: dict[str, Any] | None = None) -> list[str]:
    issues = []
    if repository.get("url") != REPOSITORY_URL:
        issues.append(f"repository URL mismatch: {repository.get('url')}")
    if repository.get("visibility") != "PUBLIC":
        issues.append(f"repository is not public: {repository.get('visibility')}")
    if release.get("tagName") != version:
        issues.append(f"tag mismatch: {release.get('tagName')}")
    if release.get("url") != f"{REPOSITORY_URL}/releases/tag/{version}":
        issues.append(f"release URL mismatch: {release.get('url')}")
    if release.get("isDraft") is not False:
        issues.append("release is still a draft")
    if release.get("isPrerelease") is not False:
        issues.append("release is marked as a prerelease")
    remote_names = {item.get("name") for item in release.get("assets", [])}
    missing = sorted(required_names - remote_names)
    if missing:
        issues.append("missing release assets: " + ", ".join(missing))
    if development_repository is not None:
        if development_repository.get("url") != DEVELOPMENT_REPOSITORY_URL:
            issues.append(
                f"development repository URL mismatch: {development_repository.get('url')}"
            )
        if development_repository.get("visibility") != "PRIVATE":
            issues.append(
                "development repository is not private: "
                f"{development_repository.get('visibility')}"
            )
    return issues


def verify(version: str) -> dict[str, Any]:
    assets = local_assets(version)
    repository = json.loads(run_gh([
        "repo", "view", REPOSITORY, "--json", "url,visibility",
    ]))
    development_repository = json.loads(run_gh([
        "repo", "view", DEVELOPMENT_REPOSITORY, "--json", "url,visibility",
    ]))
    release = json.loads(run_gh([
        "release", "view", version, "--repo", REPOSITORY,
        "--json", "url,tagName,isDraft,isPrerelease,isImmutable,assets",
    ]))
    issues = validate_metadata(
        repository, release, version, set(assets), development_repository,
    )
    public_tag_commit = run_gh([
        "api", f"repos/{REPOSITORY}/commits/{version}", "--jq", ".sha",
    ]).strip()
    release_manifest_path = ROOT / "release" / "PUBLIC_RELEASE_MANIFEST.json"
    if not release_manifest_path.is_file():
        raise RuntimeError("Local public-release manifest is missing")
    release_manifest = json.loads(release_manifest_path.read_text(encoding="utf-8"))
    source_commit = release_manifest.get("source_study_commit")
    local_source_commit = current_source_commit()
    source_commit_embedded = source_commit == local_source_commit
    if not source_commit_embedded:
        issues.append(
            f"source study commit mismatch: manifest={source_commit}, local={local_source_commit}"
        )
    verified = []
    if not issues:
        with tempfile.TemporaryDirectory(prefix="lpbf-release-verify-") as directory:
            destination = Path(directory)
            for name, local_path in sorted(assets.items()):
                run_gh([
                    "release", "download", version, "--repo", REPOSITORY,
                    "--pattern", name, "--dir", str(destination),
                ])
                downloaded = destination / name
                if not downloaded.is_file():
                    issues.append(f"download missing after command: {name}")
                    continue
                local_hash = sha256_file(local_path)
                downloaded_hash = sha256_file(downloaded)
                if downloaded_hash != local_hash:
                    issues.append(f"download hash mismatch: {name}")
                verified.append({
                    "name": name,
                    "bytes": downloaded.stat().st_size,
                    "sha256": downloaded_hash,
                    "download_verified": downloaded_hash == local_hash,
                })
    result = {
        "schema_version": 1,
        "status": "PASS" if not issues and len(verified) == len(assets) else "FAIL",
        "repository_url": repository.get("url"),
        "repository_visibility": repository.get("visibility"),
        "development_repository_url": development_repository.get("url"),
        "development_repository_visibility": development_repository.get("visibility"),
        "release_url": release.get("url"),
        "version": version,
        "tag_name": release.get("tagName"),
        "public_tag_commit": public_tag_commit,
        "source_study_commit": source_commit,
        "local_source_commit": local_source_commit,
        "source_commit_embedded": source_commit_embedded,
        "is_draft": release.get("isDraft"),
        "is_prerelease": release.get("isPrerelease"),
        "is_immutable": release.get("isImmutable"),
        "assets": verified,
        "issues": issues,
        "verification_method": "GitHub CLI metadata query plus fresh download and local SHA-256 comparison",
    }
    return result


def write_receipt(result: dict[str, Any]) -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(OUTPUT)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", default="v1.0.0")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    result = verify(args.version)
    if args.write:
        write_receipt(result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
