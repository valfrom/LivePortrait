import sys
import os
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.modules.util import upsample_nearest3d_mps


def _run_case(shape, scale):
    torch.manual_seed(0)
    inp = torch.randn(*shape, dtype=torch.float32)
    expected = F.interpolate(inp, scale_factor=scale, mode="nearest")
    actual = upsample_nearest3d_mps(inp, scale)
    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-4)


def test_upsample_basic():
    _run_case((1, 2, 3, 4, 5), (1, 2, 2))


def test_upsample_varying_factors():
    _run_case((2, 1, 2, 3, 3), (2, 3, 4))
