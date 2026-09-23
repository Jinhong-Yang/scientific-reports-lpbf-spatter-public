from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("submission_builder", ROOT / "src/reporting/build_submission_artifacts.py")
assert SPEC and SPEC.loader
BUILDER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILDER)


def test_primary_direction_language_is_claim_bounded() -> None:
    crossing = pd.Series({"simultaneous_low": -0.01, "simultaneous_high": 0.02})
    positive = pd.Series({"simultaneous_low": 0.01, "simultaneous_high": 0.02})
    negative = pd.Series({"simultaneous_low": -0.02, "simultaneous_high": -0.01})
    assert "both directions" in BUILDER.direction(crossing)
    assert "favoured NP" in BUILDER.direction(positive)
    assert "favoured the comparator" in BUILDER.direction(negative)


def test_latex_escape_protects_public_tables() -> None:
    escaped = BUILDER.tex("NP_NB & 95%")
    assert escaped == r"NP\_NB \& 95\%"


def test_submission_preflight_waits_until_frozen_results_exist() -> None:
    status = BUILDER.preflight()["status"]
    assert status in {"PASS", "WAITING_FOR_RESULTS"}
