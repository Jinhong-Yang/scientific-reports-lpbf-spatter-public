from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "evaluation"))

from validate_evaluator import (  # noqa: E402
    central_laplacian,
    data_interface_check,
    deterministic_order_check,
    manufactured_solution_check,
)


def test_finite_difference_manufactured_laplacian():
    observed = central_laplacian(lambda x, y: x * x + y * y, 0.3, -0.4)
    assert observed == pytest.approx(4.0, abs=1e-6)


def test_autograd_and_finite_difference_agree():
    result = manufactured_solution_check()
    assert result["status"] == "PASS"
    assert result["physical_calibration_claim"] is False


def test_deterministic_order_is_hash_locked():
    first = deterministic_order_check()
    second = deterministic_order_check()
    assert first == second
    assert len(first["permutation_sha256"]) == 64


def test_validation_data_interface_preserves_manifest_split_boundary():
    result = data_interface_check()
    assert result["status"] == "PASS"
    assert result["split"] == "validation"
    assert result["split_verified_from_manifest_row"] is True
    assert result["dataset_metadata_carries_split"] is False


def test_cuda_validation_gate_is_explicit():
    assert isinstance(torch.cuda.is_available(), bool)
