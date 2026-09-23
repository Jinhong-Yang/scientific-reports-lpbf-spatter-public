"""Build the W21 author handoff report from final audited artifacts."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PDF_ROOT = ROOT / "output" / "pdf"
SUBMISSION_ROOT = ROOT / "output" / "submission"
RELEASE_ROOT = ROOT / "release"
OUTPUT_PATH = ROOT / "docs" / "FINAL_HANDOFF_REPORT.md"
PDF_NAMES = (
    "Scientific_Reports_LPBF_Manuscript.pdf",
    "Scientific_Reports_LPBF_Supplementary_Information.pdf",
    "Scientific_Reports_LPBF_Cover_Letter.pdf",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_public_url(value: str, label: str) -> str:
    if not re.fullmatch(r"https://github\.com/[^/\s]+/[^/\s]+(?:/releases/tag/v\d+\.\d+\.\d+)?", value.rstrip("/")):
        raise ValueError(f"{label} must be a canonical public GitHub URL")
    return value.rstrip("/")


def current_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip()


def artifact_row(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"Missing final artifact: {path.relative_to(ROOT)}")
    return {
        "path": str(path.relative_to(ROOT)).replace("\\", "/"),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def render_report(payload: dict[str, Any]) -> str:
    rows = "\n".join(
        f"| `{item['path']}` | {item['bytes']:,} | `{item['sha256']}` |"
        for item in payload["artifacts"]
    )
    return f"""# Final Scientific Reports submission handoff

Status: `READY_WITH_SCOPE_LIMITATION`  
Internal readiness: `MINOR_REVISION_OR_BETTER_INTERNAL_ASSESSMENT`  
Generated: {payload['generated_at']}

## Public record

- Repository: {payload['repository_url']}
- Immutable release: {payload['release_url']}
- Release version: `{payload['version']}`
- Git commit used for the handoff: `{payload['git_commit']}`
- Frozen protocol SHA-256: `{payload['protocol_sha256']}`

## Final artifacts

| Artifact | Bytes | SHA-256 |
|---|---:|---|
{rows}

Every listed PDF passed a hash-bound, page-by-page visual review. The numerical,
manuscript, build, and public-package audits passed before this report was
generated. The journal-upload bundle contains only submission-facing files;
internal review and operational documents remain outside that bundle.

## Scope carried into the submission

- The measured target is the inherited operational `inherited_bbox_region`.
  Independent human rating, corrected-label interaction, and verified physical
  particle semantics were excluded by the user-approved scope.
- The frozen held-out campaign is an internal evaluation of a historically
  accessed corpus. No same-task external cohort was available, so no external
  validation claim is made.
- The optical residual is a dimensionless image-domain regularizer, not a
  calibrated thermal or fluid governing equation.
- Licensed AI-Hub source pixels, model checkpoints, raw predictions, private
  paths, credentials, and unused independent-review materials are not public.

## Author actions before portal submission

The scientific and technical package is prepared, but the authors must still
confirm author order and affiliations, contribution roles, funding wording,
competing interests, exclusive-submission and prior-publication statements,
reviewer suggestions or exclusions, AI-assistance disclosure, and the final
version in the journal portal. This project does not submit or attest on behalf
of the authors.
"""


def build(repository_url: str, release_url: str, version: str) -> dict[str, Any]:
    if not re.fullmatch(r"v\d+\.\d+\.\d+", version):
        raise ValueError("Version must match vMAJOR.MINOR.PATCH")
    repository_url = validate_public_url(repository_url, "repository_url")
    release_url = validate_public_url(release_url, "release_url")
    if release_url != f"{repository_url}/releases/tag/{version}":
        raise ValueError("release_url must identify the requested repository and version")

    review_path = ROOT / "evidence" / "manuscript" / "W20_INTERNAL_REVIEW.json"
    package_path = SUBMISSION_ROOT / "SUBMISSION_PACKAGE_MANIFEST.json"
    publication_path = ROOT / "evidence" / "release" / "PUBLICATION_RECEIPT.json"
    if not review_path.is_file() or load_json(review_path).get("status") != "PASS":
        raise RuntimeError("W20 internal review has not passed")
    if not package_path.is_file() or load_json(package_path).get("status") != "PASS":
        raise RuntimeError("Submission package audit has not passed")
    if not publication_path.is_file() or load_json(publication_path).get("status") != "PASS":
        raise RuntimeError("Public release download verification has not passed")
    publication = load_json(publication_path)
    study_commit = current_commit()
    if publication.get("repository_url") != repository_url or publication.get("release_url") != release_url:
        raise RuntimeError("Verified publication URLs do not match the requested handoff URLs")
    if publication.get("version") != version or publication.get("source_study_commit") != study_commit:
        raise RuntimeError("Verified public release is not bound to the current source-study commit")

    protocol_text = (ROOT / "configs" / "PROTOCOL_LOCK.sha256").read_text(encoding="utf-8")
    match = re.search(r"\b[a-fA-F0-9]{64}\b", protocol_text)
    if not match:
        raise RuntimeError("Frozen protocol hash is missing")

    package_manifest = load_json(package_path)
    package_paths = [ROOT / item["path"] for item in package_manifest["packages"].values()]
    paths = [*(PDF_ROOT / name for name in PDF_NAMES), *package_paths,
             RELEASE_ROOT / f"Scientific_Reports_LPBF_Reproducibility_{version}.zip",
             package_path]
    payload = {
        "schema_version": 1,
        "status": "PASS",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "repository_url": repository_url,
        "release_url": release_url,
        "version": version,
        "git_commit": study_commit,
        "protocol_sha256": match.group(0).lower(),
        "artifacts": [artifact_row(path) for path in paths],
        "independent_evaluator": "EXCLUDED_BY_USER_SCOPE",
        "journal_submission_performed": False,
        "author_confirmations_pending": True,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT_PATH.with_suffix(".md.tmp")
    temporary.write_text(render_report(payload).rstrip() + "\n", encoding="utf-8")
    temporary.replace(OUTPUT_PATH)
    receipt = ROOT / "evidence" / "manuscript" / "W21_HANDOFF.json"
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt_temp = receipt.with_suffix(".json.tmp")
    receipt_temp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    receipt_temp.replace(receipt)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-url", required=True)
    parser.add_argument("--release-url", required=True)
    parser.add_argument("--version", default="v1.0.0")
    args = parser.parse_args()
    print(json.dumps(build(args.repository_url, args.release_url, args.version), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
