import torch
import cProfile


def profile_operations(fn, row_limit=None):
    """Run ``fn`` under :mod:`torch.profiler` and return an operations table.

    Parameters
    ----------
    fn : Callable
        The function to profile.
    row_limit : int or None, optional
        Limit the number of rows returned from :func:`~torch.profiler.profile`.
        ``None`` means no limit and returns all operations.
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

    return prof.key_averages().table(sort_by=sort_by, row_limit=row_limit)


def run_cprofile(fn, output_path):
    """Profile ``fn`` with :mod:`cProfile` and dump stats to ``output_path``."""
    profiler = cProfile.Profile()
    profiler.runcall(fn)
    profiler.dump_stats(str(output_path))
