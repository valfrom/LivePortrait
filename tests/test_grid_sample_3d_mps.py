import sys
import os
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.modules.util import grid_sample_3d_mps


def _run_case(shape_inp, shape_out):
    n, c, d, h, w = shape_inp
    D, H, W = shape_out
    torch.manual_seed(0)
    inp = torch.randn(n, c, d, h, w, dtype=torch.float32)
    grid = torch.rand(n, D, H, W, 3, dtype=torch.float32) * 2 - 1

    expected = F.grid_sample(inp, grid, align_corners=False)
    actual = grid_sample_3d_mps(inp, grid)
    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-4)


def test_grid_sample_identity_size():
    _run_case((1, 2, 3, 4, 5), (3, 4, 5))


def test_grid_sample_different_output_size():
    _run_case((2, 3, 5, 6, 7), (4, 3, 8))
