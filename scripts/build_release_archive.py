"""Build and audit a source-data-free public reproducibility archive."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile
import zipfile


ROOT = Path(__file__).resolve().parents[1]
RELEASE_ROOT = ROOT / "release"
INCLUDE_PREFIXES = ("configs/", "docs/", "evidence/", "manuscript/", "notebooks/", "scripts/", "src/", "tests/")
INCLUDE_FILES = {
    ".gitignore", "pytest.ini", "README.md", "ENVIRONMENT_LOCK.json",
    "DATA_AVAILABILITY.md", "CODE_AVAILABILITY.md", "REPRODUCIBILITY_GUIDE.md",
    "reproduce.ps1", "reproduce.sh", "requirements-reporting.txt",
}
EXCLUDE_PREFIXES = (
    "docs/PROJECT_COMPLETION_AUDIT.md", "evidence/completion/",
    "docs/FINAL_HANDOFF", "evidence/manuscript/W21_HANDOFF.json",
    "evidence/release/PUBLICATION_RECEIPT.json",
    "docs/EXPERIMENT_ANALYSIS_AND_REVISION_PLAN_KO_20260923.md",
    "docs/INSTRUCTION_PROVENANCE.md",
    "docs/ANNOTATION_PROTOCOL.md", "docs/W04_", "configs/protocol_draft.yaml",
    "docs/GO_NO_GO.md", "docs/MISSING_INPUTS.md",
    "docs/NEAREST_WORK_MATRIX.csv", "docs/PRETEST_EXECUTION_SCHEDULE.md",
    "docs/PRETEST_PIPELINE_RECOVERY_LEDGER.md", "docs/RECOVERY_STATUS.csv",
    "docs/SUBMISSION_COMPLETION_CRITERIA.md", "docs/TASK_LEDGER.csv",
    "docs/W08_APPROVAL_FORM.md",
    "evidence/custody/", "evidence/environment/", "evidence/stronger_generator/cache/",
    "evidence/evaluator/", "evidence/human_review/",
    "evidence/stronger_generator/generation_per_image.parquet",
    "evidence/resolution/common64_predictions/",
    "evidence/test_campaign/TEST_ACCESS_EVENT.json",
    "src/data/build_audit_bundle.py", "src/evaluation/validate_ratings.py",
    "tests/test_build_audit_bundle.py", "tests/test_validate_ratings.py", "tests/test_w04_bundle_evidence.py",
)
FORBIDDEN_PATTERNS = (
    re.compile(rb"[A-Za-z]:\\Users\\", re.I),
    re.compile(rb"[A-Za-z]:\\Codex Research", re.I),
    re.compile(rb"ghp_[A-Za-z0-9_]{20,}"),
    re.compile(rb"github_pat_[A-Za-z0-9_]{20,}"),
    re.compile(rb"Bearer\s+[A-Za-z0-9._-]{20,}", re.I),
    re.compile(rb"sk-[A-Za-z0-9_-]{20,}"),
    re.compile(rb"AKIA[A-Z0-9]{16}"),
    re.compile(rb"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def tracked_files() -> list[str]:
    output = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT)
    return [value.decode("utf-8") for value in output.split(b"\0") if value]


def source_study_commit() -> str:
    # An extracted public archive has no .git; do not inherit an enclosing repo.
    if (ROOT / ".git").exists():
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    manifest = ROOT / "release/PUBLIC_RELEASE_MANIFEST.json"
    if manifest.is_file():
        return json.loads(manifest.read_text(encoding="utf-8"))["source_study_commit"]
    raise RuntimeError("Neither source Git metadata nor release provenance exists")


def source_state_issues(status: str, head: str, upstream: str) -> list[str]:
    issues = []
    if status.strip():
        issues.append("tracked or untracked source changes are not committed")
    if not re.fullmatch(r"[a-f0-9]{40}", head.strip()):
        issues.append("local HEAD is not a resolved Git commit")
    if head.strip() != upstream.strip():
        issues.append("local HEAD is not identical to its pushed upstream commit")
    return issues


def assert_source_tree_published() -> None:
    status = subprocess.check_output(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=ROOT, text=True,
    )
    head = source_study_commit()
    try:
        upstream = subprocess.check_output(
            ["git", "rev-parse", "@{upstream}"], cwd=ROOT, text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except subprocess.CalledProcessError as error:
        raise RuntimeError("Source branch has no verifiable pushed upstream") from error
    issues = source_state_issues(status, head, upstream)
    if issues:
        raise RuntimeError("Release source state is not publication-ready:\n" + "\n".join(issues))


def included(relative: str) -> bool:
    normalized = relative.replace("\\", "/")
    if normalized in INCLUDE_FILES:
        return True
    if not normalized.startswith(INCLUDE_PREFIXES):
        return False
    if normalized.startswith(EXCLUDE_PREFIXES):
        return False
    if any(part in normalized for part in ("/human_review/", "/raw_link/", "/checkpoints/")):
        return False
    if Path(normalized).suffix.lower() in {".pt", ".pth", ".npy", ".bmp", ".png", ".jpg", ".jpeg"}:
        return normalized.startswith("manuscript/figures/")
    return True


def audit_payload(relative: str, payload: bytes) -> list[str]:
    issues = []
    # Scan every payload, including notebooks, HTML, PDFs, archives, and other
    # nominally binary formats.  Paths and credentials can be embedded in
    # metadata or compressed-container source text, so suffix-based scanning
    # would leave an avoidable publication gap.
    for pattern in FORBIDDEN_PATTERNS:
        if pattern.search(payload):
            issues.append(f"forbidden byte pattern in {relative}: {pattern.pattern!r}")
    if len(payload) > 100 * 1024 * 1024:
        issues.append(f"unexpected file over 100 MiB: {relative}")
    return issues


def build(version: str) -> dict:
    if not re.fullmatch(r"v\d+\.\d+\.\d+", version):
        raise ValueError("Version must match vMAJOR.MINOR.PATCH")
    assert_source_tree_published()
    selected = [relative for relative in tracked_files() if included(relative)]
    if not selected:
        raise RuntimeError("Release selection is empty")
    issues, entries = [], []
    payloads = {}
    for relative in selected:
        path = ROOT / relative
        payload = path.read_bytes()
        issues.extend(audit_payload(relative, payload))
        payloads[relative] = payload
        entries.append({"path": relative, "sha256": sha256_bytes(payload), "bytes": len(payload)})
    if issues:
        raise RuntimeError("Release audit failed:\n" + "\n".join(issues))
    manifest = {
        "schema_version": 1,
        "status": "AUDIT_PASS",
        "version": version,
        "source_study_commit": source_study_commit(),
        "publication_topology": "clean_public_repository_from_allowlisted_archive",
        "source_data_included": False,
        "model_checkpoints_included": False,
        "files": entries,
    }
    manifest_payload = (json.dumps(manifest, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    checksum_payload = "".join(f"{entry['sha256']}  {entry['path']}\n" for entry in entries).encode("utf-8")
    archive_path = RELEASE_ROOT / f"Scientific_Reports_LPBF_Reproducibility_{version}.zip"
    RELEASE_ROOT.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=RELEASE_ROOT, suffix=".zip", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            for relative, payload in sorted(payloads.items()):
                archive.writestr(relative, payload)
            archive.writestr("release/PUBLIC_RELEASE_MANIFEST.json", manifest_payload)
            archive.writestr("release/SHA256SUMS.txt", checksum_payload)
        temporary.replace(archive_path)
    finally:
        temporary.unlink(missing_ok=True)
    result = {
        **manifest,
        "archive": str(archive_path.relative_to(ROOT)).replace("\\", "/"),
        "archive_sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest(),
        "archive_bytes": archive_path.stat().st_size,
    }
    (RELEASE_ROOT / "PUBLIC_RELEASE_MANIFEST.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (RELEASE_ROOT / "SHA256SUMS.txt").write_bytes(checksum_payload)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", default="v1.0.0")
    args = parser.parse_args()
    print(json.dumps(build(args.version), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
