from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("submission_audit", ROOT / "src/reporting/audit_submission.py")
assert SPEC and SPEC.loader
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


def test_word_count_ignores_common_tex_commands() -> None:
    assert AUDIT.word_count(r"A \emph{small} test \citep{x}.") == 3


def test_escaped_percentage_does_not_discard_the_rest_of_a_paragraph() -> None:
    assert "reported afterward" in AUDIT.strip_tex(r"95\% intervals reported afterward")
    assert "comment" not in AUDIT.strip_tex("Text % comment")


def test_citation_and_bibliography_key_checks() -> None:
    cited = AUDIT.citation_keys(r"Text \citep{alpha,beta} and \citet{gamma}.")
    available = AUDIT.bibliography_keys("@article{alpha, title={A}}\n@misc{beta, title={B}}")
    assert cited == {"alpha", "beta", "gamma"}
    assert cited - available == {"gamma"}


def test_legend_words_are_counted_separately() -> None:
    assert AUDIT.legend_word_counts("First short legend.\\par Second legend here.") == [3, 3]


def test_hash_map_mismatches_detects_changed_and_missing_files(tmp_path: Path) -> None:
    path = tmp_path / "source.csv"
    path.write_bytes(b"frozen bytes")
    expected = AUDIT.sha256_file(path)
    assert AUDIT.hash_map_mismatches({"source.csv": expected}, tmp_path) == []
    path.write_bytes(b"changed bytes")
    assert AUDIT.hash_map_mismatches({"source.csv": expected}, tmp_path)[0]["path"] == "source.csv"
    path.unlink()
    assert AUDIT.hash_map_mismatches({"source.csv": expected}, tmp_path)[0]["actual"] is None


def test_submission_audit_distinguishes_author_confirmations() -> None:
    review, readiness = AUDIT.write_review(
        {"status": "PASS", "checks": {"inherited_target_limited": True, "external_validity_limited": True}},
        {"status": "PASS"}, {"status": "PASS"}, {"status": "PASS"},
    )
    assert "MINOR_REVISION_OR_BETTER_INTERNAL_ASSESSMENT" in review
    assert "READY_PENDING_AUTHOR_CONFIRMATIONS" in readiness
    assert "not an independent evaluator" in review


def test_automated_checks_alone_cannot_award_minor_revision() -> None:
    review, readiness = AUDIT.write_review(
        {"status": "PASS", "checks": {}}, {"status": "PASS"}, {"status": "PASS"})
    assert "MINOR_REVISION_OR_BETTER_INTERNAL_ASSESSMENT" not in review
    assert "READY_PENDING_AUTHOR_CONFIRMATIONS" not in readiness.split("The scientific")[0]


def test_initial_submission_structure_matches_current_checklist() -> None:
    checks = AUDIT.manuscript_qa()["checks"]
    assert checks["title_matches_supplement"]
    assert checks["data_availability_before_references"]
    assert checks["post_reference_statements_ordered"]
    assert checks["no_main_text_footnotes"]
    assert checks["corresponding_author_identified"]
    assert checks["abstract_is_unstructured_and_citation_free"]
    assert checks["literature_context_covers_process_monitoring_physics_and_generation"]
    assert checks["reproducible_method_components_present"]
    assert checks["primary_statistical_design_explicit"]
    assert checks["licensed_data_boundary_explicit"]
    assert checks["run_traceability_explicit"]
