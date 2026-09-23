import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
GENERATOR_DIR = ROOT / "src" / "generators"
sys.path.insert(0, str(GENERATOR_DIR))
SPEC = importlib.util.spec_from_file_location("run_w10_prior_sweep", GENERATOR_DIR / "run_w10_prior_sweep.py")
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_w10_roster_is_protocol_frozen():
    rows = MODULE.roster()
    assert len(rows) == 18
    assert len(set(rows)) == 18
    assert {weight for weight, _ in rows} == {0.0, 0.00035, 0.0035, 0.01, 0.035, 0.1}
    assert {seed for _, seed in rows} == {66001, 66002, 66003}


def test_lambda_slug_is_stable():
    assert MODULE.lambda_slug(0.0) == "L0p00000"
    assert MODULE.lambda_slug(0.0035) == "L0p00350"
    assert MODULE.lambda_slug(0.1) == "L0p10000"


def test_w10_pretest_roles_only():
    value = MODULE.config()
    assert value["data_roles"] == ["train", "validation"]
    assert value["test_access"] == "PROHIBITED"
