# coding: utf-8

"""
tools to measure elapsed time
"""

import time
import functools

from .rprint import rlog as log

class Timer(object):
    """A simple timer."""

    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        return self.diff

    def clear(self):
        self.start_time = 0.
        self.diff = 0.


def log_time(func):
    """Decorator to log the execution time of a function."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        timer = Timer()
        timer.tic()
        result = func(*args, **kwargs)
        elapsed = timer.toc()
        log(f"{func.__qualname__} took {elapsed:.3f}s")
        return result

    return wrapper
