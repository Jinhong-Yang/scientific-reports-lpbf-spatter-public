import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "generators"))
SPEC = importlib.util.spec_from_file_location("run_w11_diffusion", ROOT / "src" / "generators" / "run_w11_diffusion.py")
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_full_diffusion_contract_matches_lock():
    value = MODULE.config()
    assert value["optimizer_updates"] == 100000
    assert value["sampling_rule"] == "DDIM_eta_0_exactly_50_steps"
    assert value["data_roles"] == ["train", "validation"]
    assert len(value["seeds"]) == 10


def test_stateless_streams_are_stable_and_distinct():
    assert MODULE.stream_seed(61001, 1, "indices") == MODULE.stream_seed(61001, 1, "indices")
    assert MODULE.stream_seed(61001, 1, "indices") != MODULE.stream_seed(61001, 2, "indices")
    assert MODULE.stream_seed(61001, 1, "indices") != MODULE.stream_seed(61001, 1, "diffusion")
