import torch


def profile_operations(fn, top_k=10):
    """Run ``fn`` under :mod:`torch.profiler` and return a table of the slowest operations.

    Parameters
    ----------
    fn : Callable
        The function to profile.
    top_k : int, optional
        Number of slowest operations to report. Defaults to 10.
    """
    activities = [torch.profiler.ProfilerActivity.CPU]
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() and hasattr(torch.profiler.ProfilerActivity, "MPS"):
        activities.append(torch.profiler.ProfilerActivity.MPS)
        sort_by = "self_cpu_time_total"  # MPS currently shares CPU times
    elif torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)
        sort_by = "cuda_time_total"
    else:
        sort_by = "self_cpu_time_total"

    with torch.profiler.profile(activities=activities, record_shapes=True) as prof:
        fn()

    return prof.key_averages().table(sort_by=sort_by, row_limit=top_k)
