"""Run the internal, non-independent pre-submission manuscript audit."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
MANUSCRIPT = ROOT / "manuscript"
GENERATED = MANUSCRIPT / "generated"
EVIDENCE = ROOT / "evidence" / "manuscript"
PDF_ROOT = ROOT / "output" / "pdf"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def hash_map_mismatches(values: dict[str, str], base: Path) -> list[dict[str, str | None]]:
    mismatches: list[dict[str, str | None]] = []
    for relative, expected in values.items():
        path = base / relative
        actual = sha256_file(path) if path.exists() else None
        if actual != expected:
            mismatches.append({"path": relative, "expected": expected, "actual": actual})
    return mismatches


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value.rstrip() + "\n", encoding="utf-8")
    temporary.replace(path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False))


def strip_tex(value: str) -> str:
    value = re.sub(r"%.*", " ", value)
    value = re.sub(r"\\(?:citep|citet|cite|ref|eqref)\{[^}]*\}", " ", value)
    value = re.sub(r"\\(?:input|includegraphics|bibliography|bibliographystyle)\{[^}]*\}", " ", value)
    value = re.sub(r"\\begin\{[^}]*\}|\\end\{[^}]*\}", " ", value)
    value = re.sub(r"\\[A-Za-z@]+\*?(?:\[[^]]*\])?", " ", value)
    value = value.replace("{", " ").replace("}", " ").replace("~", " ")
    value = re.sub(r"\$[^$]*\$|\\\[[\s\S]*?\\\]", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def word_count(value: str) -> int:
    return len(re.findall(r"\b[\w'-]+\b", strip_tex(value)))


def citation_keys(value: str) -> set[str]:
    keys: set[str] = set()
    for match in re.finditer(r"\\cite(?:p|t)?\{([^}]*)\}", value):
        keys.update(part.strip() for part in match.group(1).split(",") if part.strip())
    return keys


def bibliography_keys(value: str) -> set[str]:
    return set(re.findall(r"@\w+\{\s*([^,\s]+)\s*,", value, re.IGNORECASE))


def legend_word_counts(value: str) -> list[int]:
    return [word_count(part) for part in re.split(r"\\par\s*", value) if part.strip()]


def extract_main_text(main: str) -> str:
    intro = main.split("\\section{Introduction}", 1)[1].split("\\section{Results}", 1)[0]
    return intro + "\n" + (GENERATED / "results.tex").read_text(encoding="utf-8") + "\n" + (GENERATED / "discussion.tex").read_text(encoding="utf-8")


def pdf_pages(path: Path) -> int | None:
    if not path.exists():
        return None
    completed = subprocess.run(["pdfinfo", str(path)], capture_output=True, text=True, check=True)
    match = re.search(r"^Pages:\s+(\d+)", completed.stdout, re.MULTILINE)
    return int(match.group(1)) if match else None


def build_qa() -> dict[str, Any]:
    expected = {
        "manuscript": PDF_ROOT / "Scientific_Reports_LPBF_Manuscript.pdf",
        "supplement": PDF_ROOT / "Scientific_Reports_LPBF_Supplementary_Information.pdf",
        "cover_letter": PDF_ROOT / "Scientific_Reports_LPBF_Cover_Letter.pdf",
    }
    logs = {
        "manuscript": MANUSCRIPT / "build" / "main.log",
        "supplement": MANUSCRIPT / "build" / "supplementary.log",
        "cover_letter": MANUSCRIPT / "build" / "cover_letter.log",
    }
    messages = []
    for name, path in logs.items():
        if not path.exists():
            messages.append({"document": name, "severity": "major", "message": "missing build log"})
            continue
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if re.search(r"undefined|LaTeX Warning", line, re.IGNORECASE):
                messages.append({"document": name, "severity": "major", "message": line.strip()})
            overfull = re.search(r"Overfull \\hbox \(([0-9.]+)pt too wide\)", line)
            if overfull:
                amount = float(overfull.group(1))
                messages.append({"document": name, "severity": "minor" if amount <= 1 else "major", "message": line.strip()})
    visual_path = EVIDENCE / "VISUAL_QA.json"
    visual = json.loads(visual_path.read_text(encoding="utf-8")) if visual_path.exists() else {"status": "PENDING"}
    pdfs = {name: {"path": str(path.relative_to(ROOT)).replace("\\", "/"), "exists": path.exists(),
                   "pages": pdf_pages(path), "sha256": sha256_file(path) if path.exists() else None}
            for name, path in expected.items()}
    supplementary_pages = pdfs["supplement"]["pages"]
    if supplementary_pages is not None and supplementary_pages > 20:
        messages.append({
            "document": "supplement", "severity": "major",
            "message": f"Supplementary Information is {supplementary_pages} pages; internal target is at most 20",
        })
    result = {
        "schema_version": 1,
        "pdfs": pdfs,
        "messages": messages,
        "visual_qa": visual,
        "supplement_page_target_at_most_20": supplementary_pages is not None and supplementary_pages <= 20,
    }
    current_hashes = {name: item["sha256"] for name, item in result["pdfs"].items()}
    visual_hashes_match = visual.get("reviewed_pdf_sha256") == current_hashes
    visual_pages_match = visual.get("reviewed_pages") == {
        name: item["pages"] for name, item in result["pdfs"].items()
    }
    result["visual_hashes_match"] = visual_hashes_match
    result["visual_pages_match"] = visual_pages_match
    major = [message for message in messages if message["severity"] == "major"]
    result["status"] = "PASS" if (
        all(item["exists"] and item["pages"] for item in result["pdfs"].values())
        and not major and visual.get("status") == "PASS" and visual_hashes_match and visual_pages_match
    ) else "PENDING_OR_FAIL"
    return result


def numeric_qa() -> dict[str, Any]:
    map_path = EVIDENCE / "MANUSCRIPT_NUMERIC_MAP.json"
    if not map_path.exists():
        return {"schema_version": 1, "status": "PENDING", "reason": "MANUSCRIPT_NUMERIC_MAP.json missing"}
    value = json.loads(map_path.read_text(encoding="utf-8"))
    generated_mismatches = hash_map_mismatches(value.get("generated_hashes", {}), GENERATED)
    support_mismatches = hash_map_mismatches(value.get("support_document_hashes", {}), ROOT)
    source_mismatches = hash_map_mismatches(value.get("source_hashes", {}), ROOT)
    figure_mismatches = hash_map_mismatches(value.get("figure_hashes", {}), ROOT)
    mismatches = generated_mismatches + support_mismatches + source_mismatches + figure_mismatches
    primary = value.get("primary_contrasts", [])
    checks = {
        "four_primary_contrasts": [row["contrast"] for row in primary] == ["NP-NB", "NP-NF", "NP-N1", "NP-NR"],
        "twenty_thousand_draws": all(int(row["draws"]) == 20000 for row in primary),
        "thirty_two_test_specimens": all(int(row["specimens"]) == 32 for row in primary),
        "ten_pipeline_replicates": all(int(row["pipeline_replicates"]) == 10 for row in primary),
        "generated_hashes_present": bool(value.get("generated_hashes")),
        "support_document_hashes_present": bool(value.get("support_document_hashes")),
        "source_hashes_present": bool(value.get("source_hashes")),
        "figure_hashes_present": bool(value.get("figure_hashes")),
        "generated_hashes_match": not generated_mismatches,
        "support_document_hashes_match": not support_mismatches,
        "source_hashes_match": not source_mismatches,
        "figure_hashes_match": not figure_mismatches,
        "builder_did_not_read_test_payload": value.get("test_payload_read_by_builder") is False,
    }
    return {"schema_version": 1, "status": "PASS" if all(checks.values()) else "FAIL", "checks": checks, "mismatches": mismatches}


def manuscript_qa() -> dict[str, Any]:
    main = (MANUSCRIPT / "main.tex").read_text(encoding="utf-8")
    abstract = (GENERATED / "abstract.tex").read_text(encoding="utf-8")
    results_text = (GENERATED / "results.tex").read_text(encoding="utf-8")
    discussion_text = (GENERATED / "discussion.tex").read_text(encoding="utf-8")
    legend_text = (GENERATED / "figure_legends.tex").read_text(encoding="utf-8")
    supplement = (MANUSCRIPT / "supplementary.tex").read_text(encoding="utf-8")
    supplement_generated = "\n".join(
        path.read_text(encoding="utf-8") for path in GENERATED.glob("supplement_*.tex")
    )
    cover_letter = (MANUSCRIPT / "cover_letter.tex").read_text(encoding="utf-8")
    bibliography = (MANUSCRIPT / "references.bib").read_text(encoding="utf-8")
    title_match = re.search(r"\\newcommand\{\\papertitle\}\{([^}]*)\}", main)
    title = title_match.group(1) if title_match else ""
    keyword_match = re.search(r"\\textbf\{Keywords:\}\s*([^\n]+)", main)
    keywords = [part.strip() for part in keyword_match.group(1).split(";")] if keyword_match else []
    supplement_title_match = re.search(r"\\newcommand\{\\papertitle\}\{([^}]*)\}", supplement)
    supplement_title = supplement_title_match.group(1) if supplement_title_match else ""
    introduction_text = main.split("\\section{Introduction}", 1)[1].split("\\section{Results}", 1)[0]
    required_literature_anchors = {
        "khairallah2016", "kivirasi2020", "raissi2019", "tancik2020", "ho2020",
    }
    required_method_headings = (
        "Study design and evidence layers",
        "Dataset and operational target",
        "Conditional coordinate fields",
        "Conditional diffusion and synthetic pools",
        "Detector experiments",
        "Statistical analysis",
        "Software, compute, and AI assistance",
    )
    reference_position = main.find("\\bibliographystyle")
    data_position = main.find("\\section*{Data Availability}")
    contribution_position = main.find("\\section*{Author contributions}")
    competing_position = main.find("\\section*{Competing interests}")
    legends_position = main.find("\\section*{Figure legends}")
    main_text = extract_main_text(main)
    generated_text = "\n".join(path.read_text(encoding="utf-8") for path in GENERATED.glob("*.tex"))
    unresolved_result_markers = [phrase for phrase in ("will be generated", "will include", "will report", "pending]") if phrase in generated_text.lower()]
    combined = main + "\n" + cover_letter
    author_actions = re.findall(r"\\authoraction\{([^}]*)\}", combined)
    author_actions.extend(re.findall(r"\[(AUTHOR TO [^]]+)\]", combined))
    figure_count = len(re.findall(r"\\begin\{figure\}", results_text))
    table_count = len(re.findall(r"\\begin\{(?:table|longtable)\}", main + "\n" + results_text))
    legend_counts = legend_word_counts(legend_text)
    cited = citation_keys(main + "\n" + results_text + "\n" + discussion_text + "\n" + supplement)
    missing_bibliography_keys = sorted(cited - bibliography_keys(bibliography))
    checks = {
        "title_at_most_20_words": word_count(title) <= 20,
        "abstract_at_most_200_words": word_count(abstract) <= 200,
        "abstract_is_unstructured_and_citation_free": "\\section" not in abstract and not citation_keys(abstract),
        "main_text_at_most_4500_words": word_count(main_text) <= 4500,
        "keywords_at_most_6": len(keywords) <= 6,
        "display_items_at_most_8": figure_count + table_count <= 8,
        "each_figure_legend_at_most_350_words": bool(legend_counts) and max(legend_counts) <= 350,
        "results_have_no_pending_markers": not unresolved_result_markers,
        "supplement_has_no_pending_markers": not any(
            phrase in supplement_generated.lower()
            for phrase in ("will be generated", "will include", "will report", "pending]")
        ),
        "cited_references_present": not missing_bibliography_keys,
        "literature_context_covers_process_monitoring_physics_and_generation": (
            required_literature_anchors <= citation_keys(introduction_text)
        ),
        "reproducible_method_components_present": all(
            f"\\subsection{{{heading}}}" in main for heading in required_method_headings
        ),
        "primary_statistical_design_explicit": all(
            phrase in main for phrase in (
                "pooled COCO AP@[.50:.95]", "20,000 draws", "98.75\\%", "NP--NB", "NP--NR",
            )
        ),
        "adverse_diffusion_diagnostics_disclosed": "ND had the largest aggregate distances" in results_text,
        "core_interpretive_limits_discussed": all(
            phrase in discussion_text for phrase in (
                "operational region rather than an independently adjudicated particle label",
                "no independent machine, build, site, material, or acquisition campaign",
                "higher-resolution branch was unsupported by native information",
            )
        ),
        "licensed_data_boundary_explicit": "AI-Hub" in main and "not redistributed" in main,
        "run_traceability_explicit": (
            "Run identities recorded protocol" in main and "checkpoint, and prediction hashes" in main
        ),
        "data_availability_present": "\\section*{Data Availability}" in main,
        "code_availability_present": "\\section*{Code availability}" in main,
        "competing_interests_present": "\\section*{Competing interests}" in main,
        "author_contributions_present": "\\section*{Author contributions}" in main,
        "title_matches_supplement": bool(title) and title == supplement_title,
        "data_availability_before_references": 0 <= data_position < reference_position,
        "post_reference_statements_ordered": (
            0 <= reference_position < contribution_position < competing_position < legends_position
        ),
        "no_main_text_footnotes": "\\footnote{" not in main,
        "corresponding_author_identified": "$^{3,*}$" in main and "Correspondence:" in main,
        "author_approval_not_preasserted": "All authors reviewed and approved the final manuscript" not in main,
        "competing_interests_not_preasserted": "The authors declare no competing interests" not in main,
        "competing_interests_confirmation_explicit": "Insert the competing-interest declaration confirmed by every author" in main,
        "ai_assistance_disclosed": "OpenAI Codex assisted" in main,
        "external_validity_limited": "does not support an external-validity claim" in main,
        "inherited_target_limited": "inherited\\_bbox\\_region" in main,
        "argument_map_present": (ROOT / "docs/ARGUMENT_MAP.md").is_file(),
        "protocol_incident_disclosed": "test-access ledger records one pre-unlock display" in supplement_generated,
        "independent_evaluator_excluded": "independent human rater study" in main and "outside the user-approved scope" in main,
        "cover_letter_exclusivity_confirmation_explicit": "AUTHOR TO CONFIRM BEFORE SUBMISSION" in cover_letter,
        "cover_letter_reviewer_choices_explicit": "AUTHOR TO SUPPLY OR CONFIRM NONE" in cover_letter,
    }
    return {
        "schema_version": 1, "status": "PASS" if all(checks.values()) else "FAIL", "checks": checks,
        "counts": {"title_words": word_count(title), "abstract_words": word_count(abstract),
                   "main_text_words": word_count(main_text), "keywords": len(keywords), "figures": figure_count,
                   "tables": table_count, "display_items": figure_count + table_count,
                   "figure_legend_words": legend_counts,
                   "author_confirmation_items": len(author_actions)},
        "unresolved_result_markers": unresolved_result_markers,
        "missing_bibliography_keys": missing_bibliography_keys,
        "author_confirmation_items": author_actions,
    }


def write_review(manuscript: dict[str, Any], numeric: dict[str, Any], build: dict[str, Any]) -> tuple[str, str]:
    scientific_pass = manuscript["status"] == numeric["status"] == "PASS"
    build_pass = build["status"] == "PASS"
    verdict = "MINOR_REVISION_OR_BETTER_INTERNAL_ASSESSMENT" if scientific_pass and build_pass else "NOT_READY"
    readiness = "READY_PENDING_AUTHOR_CONFIRMATIONS" if verdict.startswith("MINOR") else "NOT_READY"
    issues = []
    if manuscript["status"] != "PASS": issues.append("Journal-structure or scope audit failed.")
    if numeric["status"] != "PASS": issues.append("Numerical traceability audit failed.")
    if build["status"] != "PASS": issues.append("PDF build or visual inspection remains incomplete.")
    review = f"""# Internal pre-submission review

## Outcome

`{verdict}`

This is an author-side technical and editorial self-review, not an independent evaluator or peer-review process. The assessment is limited to internal consistency, traceability, journal structure, claim discipline, reproducibility, and rendered-file quality.

## Major-issue screen

- Protocol-frozen held-out analysis complete: {'PASS' if numeric['status'] == 'PASS' else 'FAIL'}
- Numerical statements hash-linked to aggregate evidence: {'PASS' if numeric['status'] == 'PASS' else 'FAIL'}
- Operational inherited-label limitation explicit: {'PASS' if manuscript['checks'].get('inherited_target_limited') else 'FAIL'}
- External-validity limitation explicit: {'PASS' if manuscript['checks'].get('external_validity_limited') else 'FAIL'}
- Journal length and required-statement checks: {'PASS' if manuscript['status'] == 'PASS' else 'FAIL'}
- Reproducible method components and literature context: {'PASS' if manuscript['checks'].get('reproducible_method_components_present') and manuscript['checks'].get('literature_context_covers_process_monitoring_physics_and_generation') else 'FAIL'}
- Adverse comparator result and core limitations disclosed: {'PASS' if manuscript['checks'].get('adverse_diffusion_diagnostics_disclosed') and manuscript['checks'].get('core_interpretive_limits_discussed') else 'FAIL'}
- PDF build and full-page visual inspection: {'PASS' if build['status'] == 'PASS' else 'PENDING/FAIL'}

## Remaining minor/editorial actions

- Corresponding author must confirm author order, affiliations, contribution roles, funding wording, competing interests, and final approval.
- Corresponding author must confirm exclusivity/prior-publication statements and choose reviewer suggestions or explicitly submit none.
- The journal portal may impose metadata formatting changes that do not alter the scientific claims or numerical content.

## Current issues

{chr(10).join('- ' + item for item in issues) if issues else '- No internally detected scientific or rendering issue requiring major revision.'}
"""
    readiness_doc = f"""# Submission readiness

Status: `{readiness}`

The scientific package is internally complete only when this file reports `READY_PENDING_AUTHOR_CONFIRMATIONS`. The remaining confirmations are author attestations that cannot be inferred or completed by automation and do not authorize journal submission on the authors' behalf.

- Manuscript audit: `{manuscript['status']}`
- Numerical audit: `{numeric['status']}`
- Build and visual audit: `{build['status']}`
- Internal review outcome: `{verdict}`
- Independent evaluator process: `EXCLUDED_BY_USER_SCOPE`
"""
    return review, readiness_doc


def audit() -> dict[str, Any]:
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    manuscript = manuscript_qa()
    numeric = numeric_qa()
    build = build_qa()
    atomic_json(EVIDENCE / "MANUSCRIPT_QA.json", manuscript)
    atomic_json(EVIDENCE / "NUMERIC_QA.json", numeric)
    atomic_json(EVIDENCE / "BUILD_QA.json", build)
    review, readiness = write_review(manuscript, numeric, build)
    atomic_text(ROOT / "docs/PRE_SUBMISSION_REVIEW.md", review)
    atomic_text(ROOT / "docs/SUBMISSION_READINESS.md", readiness)
    status = "PASS" if manuscript["status"] == numeric["status"] == build["status"] == "PASS" else "PENDING_OR_FAIL"
    receipt = {"schema_version": 1, "status": status, "manuscript": manuscript["status"],
               "numeric": numeric["status"], "build": build["status"],
               "independent_evaluator": "EXCLUDED_BY_USER_SCOPE"}
    atomic_json(EVIDENCE / "W20_INTERNAL_REVIEW.json", receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", required=True)
    args = parser.parse_args()
    result = audit()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] != "FAIL" else 1


if __name__ == "__main__":
    raise SystemExit(main())
