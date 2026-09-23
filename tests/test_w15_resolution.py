from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "analysis"))


def load_module():
    spec = importlib.util.spec_from_file_location("analyze_w15_resolution", ROOT / "src" / "analysis" / "analyze_w15_resolution.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RUNNER = load_module()


class Fixture:
    family = "fasterrcnn"

    def __len__(self):
        return 1

    def __getitem__(self, _index):
        image = torch.rand(3, 300, 300)
        return image, {"boxes": torch.tensor([[1.0, 2.0, 3.0, 4.0]])}, {"sample_id": "x"}


def test_common64_preserves_detector_shape_and_target() -> None:
    source = Fixture()
    transformed = RUNNER.Common64Dataset(source)
    image, target, meta = transformed[0]
    assert image.shape == (3, 300, 300)
    assert target["boxes"].tolist() == [[1.0, 2.0, 3.0, 4.0]]
    assert meta["sample_id"] == "x"


def test_w15_roster_is_four_arms_by_ten_fixed_seeds() -> None:
    specs = RUNNER.selected_specs()
    assert len(specs) == 40
    assert {row["arm"] for row in specs} == {"N1", "NR", "NB", "NP"}
    assert len({row["detector_seed"] for row in specs}) == 10


def test_w15_preflight_waits_without_test_access() -> None:
    result = RUNNER.preflight()
    assert result["status"] in {"WAITING_FOR_W12", "PASS"}
    assert result["test_payload_accessed"] is False
    assert result["checks"]["zero_new_training"] is True
