from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("submission_packages", ROOT / "scripts/build_submission_packages.py")
assert SPEC and SPEC.loader
PACKAGES = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PACKAGES)


def test_overleaf_selection_has_three_entry_documents_and_bibliography() -> None:
    files = PACKAGES.overleaf_files()
    assert "manuscript/main.tex" in files
    assert "manuscript/supplementary.tex" in files
    assert "manuscript/cover_letter.tex" in files
    assert "manuscript/references.bib" in files


def test_overleaf_handoff_name_matches_work_order() -> None:
    assert PACKAGES.INITIAL_OVERLEAF_NAME == "Scientific_Reports_LPBF_Initial_Submission_Overleaf.zip"


def test_numerical_bundle_excludes_predictions_and_source_pixels() -> None:
    files = PACKAGES.numerical_files()
    assert not any("predictions" in path for path in files)
    assert not any(Path(path).suffix.lower() in {".bmp", ".jpg", ".jpeg", ".npy", ".pt", ".pth"} for path in files)


def test_journal_bundle_defines_four_separate_vector_figures() -> None:
    expected = {
        "Figure_1_Generator_Mechanism.pdf",
        "Figure_2_Synthetic_Quality.pdf",
        "Figure_3_Primary_Detection.pdf",
        "Figure_4_Sensitivity.pdf",
    }
    assert {name for name, _source in PACKAGES.JOURNAL_FIGURE_NAMES} == expected


def test_journal_bundle_excludes_internal_review_and_working_documents() -> None:
    files = PACKAGES.journal_files()
    assert len(files) == 3
    assert all(path.startswith("output/pdf/") and path.endswith(".pdf") for path in files)
    assert not any("PRE_SUBMISSION_REVIEW" in path or "SUBMISSION_READINESS" in path for path in files)
    assert not any(path.endswith(".md") for path in files)


def test_addition_payloads_receive_private_path_audit(tmp_path: Path) -> None:
    try:
        PACKAGES.build_zip(
            tmp_path / "bundle.zip", [],
            additions={"note.txt": b"C:\\Users\\person\\secret"},
        )
    except RuntimeError:
        pass
    else:
        raise AssertionError("unsafe addition bypassed package audit")


def test_package_audit_rejects_private_paths() -> None:
    try:
        PACKAGES.safe_payload("x.txt", b"C:\\Users\\person\\secret")
    except RuntimeError:
        pass
    else:
        raise AssertionError("private path was not rejected")


def test_package_audit_rejects_credentials_in_any_payload() -> None:
    for payload in (
        b"Bear" + b"er " + b"abcdefghijklmnopqrstuvwxyz",
        b"s" + b"k-" + b"abcdefghijklmnopqrstuvwxyz123456",
        b"AK" + b"IA" + b"ABCDEFGHIJKLMNOP",
        b"-----BEGIN " + b"PRIVATE KEY-----",
    ):
        try:
            PACKAGES.safe_payload("figure.pdf", payload)
        except RuntimeError:
            pass
        else:
            raise AssertionError("credential pattern was not rejected")
