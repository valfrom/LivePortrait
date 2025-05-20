import time
from types import SimpleNamespace
import contextlib
import torch
import torch.nn as nn
import pytest

from src.live_portrait_wrapper import LivePortraitWrapper
from src.utils.timer import Timer


class DummyWarpingModule(nn.Module):
    def forward(self, feature_3d, kp_source=None, kp_driving=None):
        bs = feature_3d.shape[0]
        device = feature_3d.device
        out = torch.randn(bs, 256, 64, 64, device=device)
        return {"out": out}


class DummySPADEGenerator(nn.Module):
    def forward(self, feature):
        bs, _, h, w = feature.shape
        device = feature.device
        return torch.randn(bs, 3, h, w, device=device)


class DummyLivePortraitWrapper(LivePortraitWrapper):
    def __init__(self, device="mps"):
        # intentionally avoid loading heavy models
        self.device = device
        self.inference_cfg = SimpleNamespace(flag_use_half_precision=False)
        self.compile = False
        self.warping_module = DummyWarpingModule().to(device)
        self.spade_generator = DummySPADEGenerator().to(device)
        self.timer = Timer()

    def inference_ctx(self):
        return contextlib.nullcontext()


def test_warp_decode_speed():
    if not torch.backends.mps.is_available():
        pytest.skip("MPS device not available")

    device = "mps"
    wrapper = DummyLivePortraitWrapper(device=device)
    feature_3d = torch.randn(1, 32, 16, 64, 64, device=device)
    kp_source = torch.randn(1, 21, 3, device=device)
    kp_driving = torch.randn(1, 21, 3, device=device)

    # warm up
    wrapper.warp_decode(feature_3d, kp_source, kp_driving)

    runs = 5
    start = time.perf_counter()
    for _ in range(runs):
        wrapper.warp_decode(feature_3d, kp_source, kp_driving)
    elapsed = time.perf_counter() - start
    avg_time = elapsed / runs

    print(f"Average warp_decode time: {avg_time:.4f}s")
    assert avg_time > 0
