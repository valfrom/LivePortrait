import sys
import os
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.modules.util import avg_pool3d_mps


def _run_case(shape_inp, kernel_size, stride=None, padding=0):
    torch.manual_seed(0)
    inp = torch.randn(*shape_inp, dtype=torch.float32)
    expected = F.avg_pool3d(inp, kernel_size, stride=stride, padding=padding)
    actual = avg_pool3d_mps(inp, kernel_size, stride=stride, padding=padding)
    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-4)


def test_pool_basic():
    _run_case((1, 3, 4, 8, 8), (1, 2, 2))


def test_pool_stride():
    _run_case((2, 2, 5, 7, 7), (2, 2, 2), stride=(2, 2, 2))
