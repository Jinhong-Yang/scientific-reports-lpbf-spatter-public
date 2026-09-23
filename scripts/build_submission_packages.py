"""Build the final Overleaf, numerical-evidence, and journal-upload bundles."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import tempfile
import zipfile


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output" / "submission"
PDF_ROOT = ROOT / "output" / "pdf"
INITIAL_OVERLEAF_NAME = "Scientific_Reports_LPBF_Initial_Submission_Overleaf.zip"
JOURNAL_FIGURE_NAMES = (
    ("Figure_1_Generator_Mechanism.pdf", "figure_1_generator_mechanism.pdf"),
    ("Figure_2_Synthetic_Quality.pdf", "figure_2_synthetic_quality.pdf"),
    ("Figure_3_Primary_Detection.pdf", "figure_3_primary_detection.pdf"),
    ("Figure_4_Sensitivity.pdf", "figure_4_sensitivity.pdf"),
)
FORBIDDEN = (
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


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def safe_payload(relative: str, payload: bytes) -> None:
    if any(pattern.search(payload) for pattern in FORBIDDEN):
        raise RuntimeError(f"Forbidden private path or token in {relative}")
    suffix = Path(relative).suffix.lower()
    if suffix in {".bmp", ".jpg", ".jpeg", ".npy", ".pt", ".pth"}:
        raise RuntimeError(f"Restricted source/model payload selected: {relative}")


def overleaf_files() -> list[str]:
    fixed = ["manuscript/main.tex", "manuscript/supplementary.tex", "manuscript/supplement_methods.tex", "manuscript/cover_letter.tex", "manuscript/references.bib"]
    generated = [str(path.relative_to(ROOT)).replace("\\", "/") for path in sorted((ROOT / "manuscript/generated").glob("*.tex"))]
    figures = [str(path.relative_to(ROOT)).replace("\\", "/") for path in sorted((ROOT / "manuscript/figures").glob("*.pdf"))]
    return fixed + generated + figures


def numerical_files() -> list[str]:
    fixed = [
        "evidence/reporting/W17_SECONDARY_RECOVERY.json",
        "evidence/manuscript/CONTENT_REVIEW.json",
        "evidence/manuscript/ISOLATED_REBUILD.json",
        "configs/PROTOCOL_LOCK.yaml", "configs/PROTOCOL_LOCK.sha256",
        "configs/w17_statistics.json", "evidence/final_statistics/primary_comparisons.parquet",
        "evidence/final_statistics/low_data_comparisons.parquet",
        "evidence/final_statistics/grouped_oof_comparisons.parquet",
        "evidence/final_statistics/ALL_COMPARISONS.parquet",
        "evidence/final_statistics/primary_bootstrap_samples.npz",
        "evidence/final_statistics/CLAIM_EVIDENCE_MATRIX.csv",
        "evidence/final_statistics/FAILURE_LEDGER.csv",
        "evidence/final_statistics/W17_STATISTICS_COMPLETENESS.json",
        "evidence/test_campaign/test_metrics.parquet",
        "evidence/test_campaign/W17_TEST_COMPLETENESS.json",
        "evidence/generator_factorial/factorial_effects_summary.csv",
        "evidence/generator_factorial/factorial_seed_metrics.csv",
        "evidence/prior_sweep/regularization_path_aggregate.csv",
        "evidence/prior_sweep/smoothing_match.csv",
        "evidence/stronger_generator/quality_arm_summary.csv",
        "evidence/stronger_generator/within_target_diversity.csv",
        "evidence/stronger_generator/W11_QUALITY_COMPLETENESS.json",
        "evidence/resolution/resolution_sensitivity.parquet",
        "evidence/resolution/W15_COMPLETENESS.json",
        "evidence/final_statistics/grouped_oof_metrics.parquet",
        "evidence/grouped_internal/W16_DETECTOR_COMPLETENESS.json",
        "evidence/low_data/W14_DETECTOR_COMPLETENESS.json",
        "evidence/detector_core/CORE_COMPLETENESS.json",
        "evidence/manuscript/MANUSCRIPT_NUMERIC_MAP.json",
        "evidence/manuscript/MANUSCRIPT_QA.json", "evidence/manuscript/NUMERIC_QA.json",
        "evidence/manuscript/BUILD_QA.json", "evidence/manuscript/VISUAL_QA.json",
        "evidence/manuscript/W20_INTERNAL_REVIEW.json",
    ]
    figure_data = [str(path.relative_to(ROOT)).replace("\\", "/") for path in sorted((ROOT / "evidence/figure_source_data").glob("*.csv"))]
    return fixed + figure_data


def journal_files() -> list[str]:
    return [
        "output/pdf/Scientific_Reports_LPBF_Manuscript.pdf",
        "output/pdf/Scientific_Reports_LPBF_Supplementary_Information.pdf",
        "output/pdf/Scientific_Reports_LPBF_Cover_Letter.pdf",
    ]


def journal_figure_additions() -> dict[str, bytes]:
    additions: dict[str, bytes] = {}
    missing = []
    for archive_name, source_name in JOURNAL_FIGURE_NAMES:
        source = ROOT / "manuscript" / "figures" / source_name
        if not source.is_file():
            missing.append(str(source.relative_to(ROOT)).replace("\\", "/"))
        else:
            additions[archive_name] = source.read_bytes()
    if missing:
        raise RuntimeError("Missing journal figure inputs:\n" + "\n".join(missing))
    return additions


def build_zip(path: Path, relatives: list[str], prefix_to_strip: str | None = None,
              additions: dict[str, bytes] | None = None) -> dict:
    missing = [relative for relative in relatives if not (ROOT / relative).is_file()]
    if missing:
        raise RuntimeError("Missing package inputs:\n" + "\n".join(missing))
    payloads: dict[str, bytes] = {}
    for relative in relatives:
        payload = (ROOT / relative).read_bytes()
        safe_payload(relative, payload)
        archive_name = relative
        if prefix_to_strip and archive_name.startswith(prefix_to_strip):
            archive_name = archive_name[len(prefix_to_strip):]
        payloads[archive_name] = payload
    for name, payload in (additions or {}).items():
        safe_payload(name, payload)
        payloads[name] = payload
    entries = [{"path": name, "sha256": sha256_bytes(payload), "bytes": len(payload)} for name, payload in sorted(payloads.items())]
    manifest = json.dumps({"schema_version": 1, "status": "PASS", "files": entries}, indent=2).encode() + b"\n"
    payloads["PACKAGE_MANIFEST.json"] = manifest
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".zip", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            for name, payload in sorted(payloads.items()):
                archive.writestr(name, payload)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return {"path": str(path.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256_file(path), "bytes": path.stat().st_size, "files": len(entries)}


def build(version: str) -> dict:
    if not re.fullmatch(r"v\d+\.\d+\.\d+", version):
        raise ValueError("Version must match vMAJOR.MINOR.PATCH")
    review_path = ROOT / "evidence/manuscript/W20_INTERNAL_REVIEW.json"
    if not review_path.exists() or json.loads(review_path.read_text(encoding="utf-8")).get("status") != "PASS":
        raise RuntimeError("W20 internal review has not passed")
    readme = (
        "Compile main.tex for the manuscript, supplementary.tex for Supplementary Information, "
        "and cover_letter.tex for the cover letter. The figures directory must remain beside these files.\n"
    ).encode()
    results = {
        "overleaf": build_zip(OUTPUT / INITIAL_OVERLEAF_NAME, overleaf_files(),
                              prefix_to_strip="manuscript/", additions={"README.txt": readme}),
        "numerical_evidence": build_zip(OUTPUT / f"Scientific_Reports_LPBF_Numerical_Evidence_{version}.zip", numerical_files()),
        "journal_upload": build_zip(
            OUTPUT / f"Scientific_Reports_LPBF_Journal_Upload_{version}.zip",
            journal_files(), prefix_to_strip="output/pdf/",
            additions=journal_figure_additions(),
        ),
    }
    receipt = {"schema_version": 1, "status": "PASS", "version": version, "packages": results,
               "source_pixels_included": False, "model_checkpoints_included": False,
               "prediction_payloads_included": False, "independent_evaluator": "EXCLUDED_BY_USER_SCOPE"}
    (OUTPUT / "SUBMISSION_PACKAGE_MANIFEST.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", default="v1.0.0")
    args = parser.parse_args()
    print(json.dumps(build(args.version), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
