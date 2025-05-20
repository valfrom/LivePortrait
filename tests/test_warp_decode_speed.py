import time
import torch
import pytest

from src.live_portrait_wrapper import LivePortraitWrapper
from src.config.inference_config import InferenceConfig


def test_warp_decode_speed():
    if not torch.backends.mps.is_available():
        pytest.skip("MPS device not available")

    cfg = InferenceConfig()
    wrapper = LivePortraitWrapper(cfg)
    device = wrapper.device
    feature_3d = torch.randn(1, 32, 16, 64, 64, device=device)
    kp_source = torch.randn(1, 21, 3, device=device)
    kp_driving = torch.randn(1, 21, 3, device=device)

    # warm up
    for _ in range(10):
        wrapper.warp_decode(feature_3d, kp_source, kp_driving)

    runs = 10
    start = time.perf_counter()
    for _ in range(runs):
        wrapper.warp_decode(feature_3d, kp_source, kp_driving)
    elapsed = time.perf_counter() - start
    avg_time = elapsed / runs

    print(f"Average warp_decode time: {avg_time:.4f}s")
    assert avg_time > 0
