import importlib.util
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "materialize_w09_validation_arrays", ROOT / "src" / "analysis" / "materialize_w09_validation_arrays.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_reconstruct_preserves_roster_and_mse():
    class Encoder(torch.nn.Module):
        def forward(self, image, condition):
            return torch.zeros((len(image), 1)), torch.zeros((len(image), 1))

    class Toy(torch.nn.Module):
        image_size = 2
        encoder = Encoder()

        def encode(self, image, condition):
            return self.encoder(image, condition)

        def decode(self, coordinates, condition, latent):
            return torch.zeros((*coordinates.shape[:-1], 1))

    images = torch.ones((3, 1, 2, 2))
    conditions = torch.zeros((3, 13))
    predictions, mse = MODULE.reconstruct(Toy(), images, conditions, torch.device("cpu"), batch_size=2)
    assert predictions.shape == (3, 1, 2, 2)
    assert predictions.dtype.name == "float16"
    assert mse == 1.0
