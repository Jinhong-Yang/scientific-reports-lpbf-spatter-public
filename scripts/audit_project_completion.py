"""Audit the complete work order, submission package, and public release."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence" / "completion"
VERSION = "v1.0.0"
REPOSITORY_URL = "https://github.com/Jinhong-Yang/scientific-reports-lpbf-spatter-public"
RECEIPTS = {
    "W09 generator factorial": ("evidence/generator_factorial/W09_COMPLETENESS.json", {"COMPLETE"}),
    "W10 prior path": ("evidence/prior_sweep/W10_SWEEP_COMPLETENESS.json", {"COMPLETE"}),
    "W11 diffusion": ("evidence/stronger_generator/W11_DIFFUSION_COMPLETENESS.json", {"COMPLETE"}),
    "W11 pools": ("evidence/stronger_generator/W11_POOL_COMPLETENESS.json", {"COMPLETE"}),
    "W11 quality": ("evidence/stronger_generator/W11_QUALITY_COMPLETENESS.json", {"PASS_WITH_SCOPE_LIMITATION"}),
    "W12 detector core": ("evidence/detector_core/CORE_COMPLETENESS.json", {"COMPLETE", "PASS"}),
    "W14 generators": ("evidence/low_data/W14_GENERATOR_COMPLETENESS.json", {"COMPLETE"}),
    "W14 detectors": ("evidence/low_data/W14_DETECTOR_COMPLETENESS.json", {"COMPLETE", "PASS"}),
    "W14 full integrity audit": ("evidence/low_data/W14_FULL_INTEGRITY_AUDIT.json", {"PASS"}),
    "W15 resolution": ("evidence/resolution/W15_COMPLETENESS.json", {"PASS"}),
    "W16 generators": ("evidence/grouped_internal/W16_GENERATOR_COMPLETENESS.json", {"COMPLETE"}),
    "W16 detectors": ("evidence/grouped_internal/W16_DETECTOR_COMPLETENESS.json", {"COMPLETE", "PASS"}),
    "W16 full integrity audit": ("evidence/grouped_internal/W16_FULL_INTEGRITY_AUDIT.json", {"PASS"}),
    "W17 held-out campaign": ("evidence/test_campaign/W17_TEST_COMPLETENESS.json", {"COMPLETE"}),
    "W17 statistics": ("evidence/final_statistics/W17_STATISTICS_COMPLETENESS.json", {"PASS"}),
    "W20 internal review": ("evidence/manuscript/W20_INTERNAL_REVIEW.json", {"PASS"}),
    "W21 handoff": ("evidence/manuscript/W21_HANDOFF.json", {"PASS"}),
}
PDFS = {
    "manuscript": "output/pdf/Scientific_Reports_LPBF_Manuscript.pdf",
    "supplement": "output/pdf/Scientific_Reports_LPBF_Supplementary_Information.pdf",
    "cover_letter": "output/pdf/Scientific_Reports_LPBF_Cover_Letter.pdf",
}
TERMINAL_TASK_STATUSES = frozenset({
    "PASS",
    "PASS_WITH_LIMITATION",
    "PASS_WITH_SCOPE_LIMITATION",
    "EXCLUDED_BY_USER_SCOPE",
})


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def check(name: str, passed: bool, detail: str, evidence: str) -> dict[str, Any]:
    return {"name": name, "status": "PASS" if passed else "WAITING_OR_FAIL", "detail": detail, "evidence": evidence}


def receipt_check(root: Path, name: str, relative: str, accepted: set[str]) -> dict[str, Any]:
    path = root / relative
    if not path.is_file():
        return check(name, False, "missing receipt", relative)
    try:
        status = load_json(path).get("status")
    except (OSError, ValueError) as error:
        return check(name, False, f"unreadable receipt: {error}", relative)
    return check(name, status in accepted, f"status={status}; accepted={sorted(accepted)}", relative)


def asset_hashes_match(required: set[str], local_paths: dict[str, Path], assets: list[dict[str, Any]]) -> bool:
    receipt_hashes = {item.get("name"): item.get("sha256") for item in assets}
    return required <= set(local_paths) and required <= set(receipt_hashes) and all(
        local_paths[name].is_file()
        and receipt_hashes[name] == sha256_file(local_paths[name])
        for name in required
    )


def evaluate(root: Path = ROOT) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    copy_path = root / "evidence/custody/COPY_RECEIPT.json"
    if copy_path.is_file():
        copy = load_json(copy_path)
        trees = copy.get("trees", [])
        files = sum(int(tree.get("files", 0)) for tree in trees)
        total_bytes = sum(int(tree.get("bytes", 0)) for tree in trees)
        clean = all(
            int(tree.get("copy_failures", -1)) == 0
            and int(tree.get("copy_mismatches", -1)) == 0
            and int(tree.get("post_copy_dry_run", {}).get("would_copy_files", -1)) == 0
            and int(tree.get("post_copy_dry_run", {}).get("extra_files", -1)) == 0
            for tree in trees
        )
        checks.append(check(
            "Managed project copy", copy.get("result") == "PASS" and len(trees) == 2 and files == 197_912 and clean,
            f"trees={len(trees)}, files={files}, bytes={total_bytes}, zero_failures_and_mismatches={clean}",
            "evidence/custody/COPY_RECEIPT.json",
        ))
    else:
        checks.append(check("Managed project copy", False, "missing receipt", "evidence/custody/COPY_RECEIPT.json"))

    identity_path = root / "PROJECT_IDENTITY.json"
    identity = load_json(identity_path) if identity_path.is_file() else {}
    managed = identity.get("managed_workspace")
    checks.append(check(
        "New managed workspace identity",
        identity.get("work_order_id") == "SR-LPBF-WO-20260914"
        and isinstance(managed, str) and managed.replace("\\", "/").endswith("/Scientific_Reports_LPBF_20260914"),
        f"managed_workspace={managed}", "PROJECT_IDENTITY.json",
    ))

    scope_path = root / "docs/SCOPE_DECISION_001.md"
    scope = scope_path.read_text(encoding="utf-8") if scope_path.is_file() else ""
    checks.append(check(
        "Independent evaluator excluded",
        "EXCLUDED_BY_USER_SCOPE" in scope and "corrected-versus-inherited" in scope,
        "W04 independent review and W13 corrected-label interaction remain excluded",
        "docs/SCOPE_DECISION_001.md",
    ))

    ledger_path = root / "docs/TASK_LEDGER.csv"
    if ledger_path.is_file():
        with ledger_path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        required = {f"W{number:02d}" for number in range(20)} | {"W20-W21"}
        by_name = {row.get("work_package"): row for row in rows}
        allowed = TERMINAL_TASK_STATUSES
        missing = sorted(required - set(by_name))
        unfinished = sorted(
            name for name in required & set(by_name)
            if by_name[name].get("status") not in allowed
        )
        checks.append(check(
            "W00-W21 task ledger terminal",
            not missing and not unfinished,
            f"missing={missing}, unfinished={unfinished}",
            "docs/TASK_LEDGER.csv",
        ))
    else:
        checks.append(check("W00-W21 task ledger terminal", False, "missing ledger", "docs/TASK_LEDGER.csv"))

    for name, (relative, accepted) in RECEIPTS.items():
        checks.append(receipt_check(root, name, relative, accepted))

    review_doc_path = root / "docs/PRE_SUBMISSION_REVIEW.md"
    readiness_doc_path = root / "docs/SUBMISSION_READINESS.md"
    review_doc = review_doc_path.read_text(encoding="utf-8") if review_doc_path.is_file() else ""
    readiness_doc = readiness_doc_path.read_text(encoding="utf-8") if readiness_doc_path.is_file() else ""
    checks.append(check(
        "Minor-revision-or-better internal assessment",
        "MINOR_REVISION_OR_BETTER_INTERNAL_ASSESSMENT" in review_doc
        and "READY_PENDING_AUTHOR_CONFIRMATIONS" in readiness_doc
        and "not an independent evaluator" in review_doc,
        "internal assessment passed with only author confirmations remaining",
        "docs/PRE_SUBMISSION_REVIEW.md; docs/SUBMISSION_READINESS.md",
    ))

    visual_path = root / "evidence/manuscript/VISUAL_QA.json"
    if visual_path.is_file():
        visual = load_json(visual_path)
        actual_hashes = {
            name: sha256_file(root / relative) if (root / relative).is_file() else None
            for name, relative in PDFS.items()
        }
        page_values = visual.get("reviewed_pages", {})
        hashes_match = visual.get("reviewed_pdf_sha256") == actual_hashes
        pages_reviewed = set(page_values) == set(PDFS) and all(isinstance(value, int) and value > 0 for value in page_values.values())
        checks.append(check(
            "Hash-bound full-page PDF review",
            visual.get("status") == "PASS" and hashes_match and pages_reviewed and visual.get("defects") == 0,
            f"hashes_match={hashes_match}, pages_reviewed={pages_reviewed}, defects={visual.get('defects')}",
            "evidence/manuscript/VISUAL_QA.json",
        ))
    else:
        checks.append(check("Hash-bound full-page PDF review", False, "missing receipt", "evidence/manuscript/VISUAL_QA.json"))

    package_path = root / "output/submission/SUBMISSION_PACKAGE_MANIFEST.json"
    if package_path.is_file():
        package = load_json(package_path)
        package_entries = package.get("packages", {})
        paths = [root / item.get("path", "") for item in package_entries.values()]
        hashes_match = all(path.is_file() for path in paths) and all(
            sha256_file(root / item["path"]) == item.get("sha256") for item in package_entries.values()
        )
        checks.append(check(
            "Submission packages", package.get("status") == "PASS" and len(package_entries) == 3 and hashes_match,
            f"packages={len(package_entries)}, hashes_match={hashes_match}",
            "output/submission/SUBMISSION_PACKAGE_MANIFEST.json",
        ))
    else:
        checks.append(check("Submission packages", False, "missing manifest", "output/submission/SUBMISSION_PACKAGE_MANIFEST.json"))

    release_manifest_path = root / f"release/PUBLIC_RELEASE_MANIFEST.json"
    if release_manifest_path.is_file():
        release = load_json(release_manifest_path)
        archive = root / str(release.get("archive", ""))
        archive_match = archive.is_file() and sha256_file(archive) == release.get("archive_sha256")
        checks.append(check(
            "Source-data-free reproducibility archive",
            release.get("status") == "AUDIT_PASS" and release.get("version") == VERSION
            and release.get("publication_topology") == "clean_public_repository_from_allowlisted_archive"
            and bool(re.fullmatch(r"[a-f0-9]{40}", str(release.get("source_study_commit", ""))))
            and release.get("source_data_included") is False
            and release.get("model_checkpoints_included") is False and archive_match,
            f"version={release.get('version')}, archive_hash_matches={archive_match}",
            "release/PUBLIC_RELEASE_MANIFEST.json",
        ))
    else:
        checks.append(check("Source-data-free reproducibility archive", False, "missing manifest", "release/PUBLIC_RELEASE_MANIFEST.json"))

    publication_path = root / "evidence/release/PUBLICATION_RECEIPT.json"
    publication: dict[str, Any] = {}
    if publication_path.is_file():
        publication = load_json(publication_path)
        expected_release = f"{REPOSITORY_URL}/releases/tag/{VERSION}"
        assets = publication.get("assets", [])
        asset_names = {item.get("name") for item in assets}
        required_assets = {
            *[Path(path).name for path in PDFS.values()],
            "Scientific_Reports_LPBF_Initial_Submission_Overleaf.zip",
            f"Scientific_Reports_LPBF_Numerical_Evidence_{VERSION}.zip",
            f"Scientific_Reports_LPBF_Journal_Upload_{VERSION}.zip",
            f"Scientific_Reports_LPBF_Reproducibility_{VERSION}.zip",
        }
        local_asset_paths = {
            **{Path(path).name: root / path for path in PDFS.values()},
            **{
                Path(item.get("path", "")).name: root / item.get("path", "")
                for item in (load_json(package_path).get("packages", {}).values() if package_path.is_file() else [])
            },
            f"Scientific_Reports_LPBF_Reproducibility_{VERSION}.zip":
                root / f"release/Scientific_Reports_LPBF_Reproducibility_{VERSION}.zip",
        }
        local_hashes_match = asset_hashes_match(required_assets, local_asset_paths, assets)
        checks.append(check(
            "Public Git release",
            publication.get("status") == "PASS"
            and publication.get("repository_url") == REPOSITORY_URL
            and publication.get("repository_visibility") == "PUBLIC"
            and publication.get("development_repository_url")
            == "https://github.com/Jinhong-Yang/scientific-reports-lpbf-spatter"
            and publication.get("development_repository_visibility") == "PRIVATE"
            and publication.get("release_url") == expected_release
            and publication.get("version") == VERSION
            and publication.get("source_commit_embedded") is True
            and bool(re.fullmatch(r"[a-f0-9]{40}", str(publication.get("source_study_commit", ""))))
            and bool(re.fullmatch(r"[a-f0-9]{40}", str(publication.get("public_tag_commit", ""))))
            and required_assets <= asset_names
            and all(re.fullmatch(r"[a-f0-9]{64}", str(item.get("sha256", ""))) for item in assets)
            and local_hashes_match,
            f"release_url={publication.get('release_url')}, verified_assets={len(asset_names)}, local_hashes_match={local_hashes_match}",
            "evidence/release/PUBLICATION_RECEIPT.json",
        ))
    else:
        checks.append(check("Public Git release", False, "missing verified publication receipt", "evidence/release/PUBLICATION_RECEIPT.json"))

    handoff_path = root / "evidence/manuscript/W21_HANDOFF.json"
    if handoff_path.is_file():
        handoff = load_json(handoff_path)
        checks.append(check(
            "Author-safe handoff",
            handoff.get("status") == "PASS"
            and handoff.get("journal_submission_performed") is False
            and handoff.get("author_confirmations_pending") is True
            and handoff.get("independent_evaluator") == "EXCLUDED_BY_USER_SCOPE"
            and handoff.get("git_commit") == publication.get("source_study_commit"),
            f"journal submission not performed; author-only attestations remain explicit; source_commit_match={handoff.get('git_commit') == publication.get('source_study_commit')}",
            "evidence/manuscript/W21_HANDOFF.json",
        ))
    else:
        checks.append(check(
            "Author-safe handoff", False, "missing handoff receipt",
            "evidence/manuscript/W21_HANDOFF.json",
        ))

    passed = all(item["status"] == "PASS" for item in checks)
    return {
        "schema_version": 1,
        "status": "PASS" if passed else "WAITING_OR_FAIL",
        "completion_classification": "READY_WITH_SCOPE_LIMITATION" if passed else "NOT_COMPLETE",
        "checks_passed": sum(item["status"] == "PASS" for item in checks),
        "checks_total": len(checks),
        "checks": checks,
        "goal_complete": passed,
        "independent_evaluator": "EXCLUDED_BY_USER_SCOPE",
        "journal_submission_performed": False,
    }


def render(result: dict[str, Any]) -> str:
    rows = "\n".join(
        f"| {item['name']} | `{item['status']}` | {item['detail']} | `{item['evidence']}` |"
        for item in result["checks"]
    )
    return f"""# Project completion audit

Status: `{result['status']}`  
Classification: `{result['completion_classification']}`  
Checks: {result['checks_passed']}/{result['checks_total']}

| Requirement | Status | Detail | Evidence |
|---|---|---|---|
{rows}

The goal may be marked complete only when every row reports `PASS`. The audit
preserves the user-approved exclusion of independent evaluator work and does
not treat author-only attestations or journal submission as automated actions.
"""


def write_result(result: dict[str, Any]) -> None:
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    json_path = EVIDENCE / "PROJECT_COMPLETION_AUDIT.json"
    md_path = ROOT / "docs" / "PROJECT_COMPLETION_AUDIT.md"
    json_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(render(result).rstrip() + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--run", action="store_true")
    args = parser.parse_args()
    result = evaluate()
    if args.run:
        write_result(result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
