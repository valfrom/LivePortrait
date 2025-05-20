# coding: utf-8

"""
Benchmark the inference speed of each module in LivePortrait.

TODO: heavy GPT style, need to refactor
"""

import os
import sys
import argparse
import torch
from src.utils.rprint import rlog as log

torch._dynamo.config.suppress_errors = True  # Suppress errors and fall back to eager execution

import yaml
import time
import numpy as np
from src.utils.profiler import profile_operations

# Enable reasonable defaults for macOS users
if sys.platform == "darwin":
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    torch.set_float32_matmul_precision("high")

from src.utils.helper import load_model, concat_feat
from src.config.inference_config import InferenceConfig


def _sync():
    """Synchronize depending on the available backend."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elif hasattr(torch, "mps") and torch.backends.mps.is_available():
        torch.mps.synchronize()


def get_device(device_id=0):
    """Return the best available :class:`torch.device` string."""
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return f"cuda:{device_id}"
    return "cpu"


def initialize_inputs(batch_size=1, device_id=0):
    """Generate random input tensors and move them to the chosen device."""
    device = get_device(device_id)
    feature_3d = torch.randn(
        batch_size, 32, 16, 64, 64, device=device, dtype=torch.float16
    )
    kp_source = torch.randn(batch_size, 21, 3, device=device, dtype=torch.float16)
    kp_driving = torch.randn(batch_size, 21, 3, device=device, dtype=torch.float16)
    source_image = (
        torch.randn(batch_size, 3, 256, 256, device=device, dtype=torch.float16)
        .to(memory_format=torch.channels_last)
    )
    generator_input = (
        torch.randn(batch_size, 256, 64, 64, device=device, dtype=torch.float16)
        .to(memory_format=torch.channels_last)
    )
    eye_close_ratio = torch.randn(batch_size, 3, device=device, dtype=torch.float16)
    lip_close_ratio = torch.randn(batch_size, 2, device=device, dtype=torch.float16)
    feat_stitching = concat_feat(kp_source, kp_driving).half()
    feat_eye = concat_feat(kp_source, eye_close_ratio).half()
    feat_lip = concat_feat(kp_source, lip_close_ratio).half()

    inputs = {
        'feature_3d': feature_3d,
        'kp_source': kp_source,
        'kp_driving': kp_driving,
        'source_image': source_image,
        'generator_input': generator_input,
        'feat_stitching': feat_stitching,
        'feat_eye': feat_eye,
        'feat_lip': feat_lip
    }

    return inputs


def load_and_compile_models(cfg, model_config, compile_models=True):
    """Load models and optionally compile them for inference.

    Parameters
    ----------
    cfg : InferenceConfig
        Configuration with checkpoint paths and device id.
    model_config : dict
        Model hyperparameters loaded from YAML.
    compile_models : bool, optional
        If ``True``, wrap every module with ``torch.compile``.
    """
    device = get_device(cfg.device_id)
    appearance_feature_extractor = load_model(
        cfg.checkpoint_F, model_config, device, 'appearance_feature_extractor'
    )
    motion_extractor = load_model(
        cfg.checkpoint_M, model_config, device, 'motion_extractor'
    )
    warping_module = load_model(
        cfg.checkpoint_W, model_config, device, 'warping_module'
    )
    spade_generator = load_model(
        cfg.checkpoint_G, model_config, device, 'spade_generator'
    )
    stitching_retargeting_module = load_model(
        cfg.checkpoint_S, model_config, device, 'stitching_retargeting_module'
    )

    models_with_params = [
        ('Appearance Feature Extractor', appearance_feature_extractor),
        ('Motion Extractor', motion_extractor),
        ('Warping Network', warping_module),
        ('SPADE Decoder', spade_generator)
    ]

    compiled_models = {}
    for name, model in models_with_params:
        model = model.half()
        if compile_models:
            model = torch.compile(model, mode='max-autotune')  # Optimize for inference
        model.eval()  # Switch to evaluation mode
        compiled_models[name] = model

    retargeting_models = ['stitching', 'eye', 'lip']
    for retarget in retargeting_models:
        module = stitching_retargeting_module[retarget].half()
        if compile_models:
            module = torch.compile(module, mode='max-autotune')  # Optimize for inference
        module.eval()  # Switch to evaluation mode
        stitching_retargeting_module[retarget] = module

    return compiled_models, stitching_retargeting_module


def warm_up_models(compiled_models, stitching_retargeting_module, inputs):
    """
    Warm up models to prepare them for benchmarking
    """
    log("Warm up start!")
    with torch.no_grad():
        for _ in range(10):
            compiled_models['Appearance Feature Extractor'](inputs['source_image'])
            compiled_models['Motion Extractor'](inputs['source_image'])
            compiled_models['Warping Network'](inputs['feature_3d'], inputs['kp_driving'], inputs['kp_source'])
            compiled_models['SPADE Decoder'](inputs['generator_input'])  # Adjust input as required
            stitching_retargeting_module['stitching'](inputs['feat_stitching'])
            stitching_retargeting_module['eye'](inputs['feat_eye'])
            stitching_retargeting_module['lip'](inputs['feat_lip'])
    log("Warm up end!")


def measure_inference_times(compiled_models, stitching_retargeting_module, inputs):
    """
    Measure inference times for each model
    """
    times = {name: [] for name in compiled_models.keys()}
    times['Stitching and Retargeting Modules'] = []

    overall_times = []

    with torch.no_grad():
        for _ in range(100):
            _sync()
            overall_start = time.time()

            start = time.time()
            compiled_models['Appearance Feature Extractor'](inputs['source_image'])
            _sync()
            times['Appearance Feature Extractor'].append(time.time() - start)

            start = time.time()
            compiled_models['Motion Extractor'](inputs['source_image'])
            _sync()
            times['Motion Extractor'].append(time.time() - start)

            start = time.time()
            compiled_models['Warping Network'](inputs['feature_3d'], inputs['kp_driving'], inputs['kp_source'])
            _sync()
            times['Warping Network'].append(time.time() - start)

            start = time.time()
            compiled_models['SPADE Decoder'](inputs['generator_input'])  # Adjust input as required
            _sync()
            times['SPADE Decoder'].append(time.time() - start)

            start = time.time()
            stitching_retargeting_module['stitching'](inputs['feat_stitching'])
            stitching_retargeting_module['eye'](inputs['feat_eye'])
            stitching_retargeting_module['lip'](inputs['feat_lip'])
            _sync()
            times['Stitching and Retargeting Modules'].append(time.time() - start)

            overall_times.append(time.time() - overall_start)

    return times, overall_times


def print_benchmark_results(compiled_models, stitching_retargeting_module, retargeting_models, times, overall_times):
    """
    Print benchmark results with average and standard deviation of inference times
    """
    average_times = {name: np.mean(times[name]) * 1000 for name in times.keys()}
    std_times = {name: np.std(times[name]) * 1000 for name in times.keys()}

    for name, model in compiled_models.items():
        num_params = sum(p.numel() for p in model.parameters())
        num_params_in_millions = num_params / 1e6
        print(f"Number of parameters for {name}: {num_params_in_millions:.2f} M")

    for index, retarget in enumerate(retargeting_models):
        num_params = sum(p.numel() for p in stitching_retargeting_module[retarget].parameters())
        num_params_in_millions = num_params / 1e6
        print(f"Number of parameters for part_{index} in Stitching and Retargeting Modules: {num_params_in_millions:.2f} M")

    for name, avg_time in average_times.items():
        std_time = std_times[name]
        print(f"Average inference time for {name} over 100 runs: {avg_time:.2f} ms (std: {std_time:.2f} ms)")


def profile_models(compiled_models, stitching_retargeting_module, inputs, top_k=10):
    """Run a single step under :mod:`torch.profiler` and return the slowest operations."""

    def _run_once():
        with torch.no_grad():
            compiled_models['Appearance Feature Extractor'](inputs['source_image'])
            compiled_models['Motion Extractor'](inputs['source_image'])
            compiled_models['Warping Network'](inputs['feature_3d'], inputs['kp_driving'], inputs['kp_source'])
            compiled_models['SPADE Decoder'](inputs['generator_input'])
            stitching_retargeting_module['stitching'](inputs['feat_stitching'])
            stitching_retargeting_module['eye'](inputs['feat_eye'])
            stitching_retargeting_module['lip'](inputs['feat_lip'])

    table = profile_operations(_run_once, top_k=top_k)
    log(table)
    return table


def main():
    """
    Main function to benchmark speed and model parameters
    """
    parser = argparse.ArgumentParser(description="Benchmark LivePortrait")
    parser.add_argument("--profile", action="store_true", help="run profiler once")
    parser.add_argument(
        "--slow-ops",
        nargs="?",
        const=10,
        type=int,
        help="display table of the slowest torch operations (optionally specify top K)"
    )
    parser.add_argument(
        "--no-compile",
        action="store_true",
        help="disable torch.compile for all models",
    )
    args = parser.parse_args()

    # Load configuration
    cfg = InferenceConfig()
    model_config_path = cfg.models_config
    with open(model_config_path, 'r') as file:
        model_config = yaml.safe_load(file)

    # Sample input tensors
    inputs = initialize_inputs(device_id=cfg.device_id)

    # Load and compile models
    compiled_models, stitching_retargeting_module = load_and_compile_models(
        cfg, model_config, compile_models=not args.no_compile
    )

    # Warm up models
    warm_up_models(compiled_models, stitching_retargeting_module, inputs)

    # Measure inference times
    times, overall_times = measure_inference_times(compiled_models, stitching_retargeting_module, inputs)

    if args.profile or args.slow_ops is not None:
        top_k = args.slow_ops if args.slow_ops is not None else 10
        profile_models(compiled_models, stitching_retargeting_module, inputs, top_k=top_k)

    # Print benchmark results
    print_benchmark_results(compiled_models, stitching_retargeting_module, ['stitching', 'eye', 'lip'], times, overall_times)


if __name__ == "__main__":
    main()
