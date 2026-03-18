import functools
import time
import torch

def timer(func):
    @functools.wraps(func)
    def wrapper(*args, debug=False, **kwargs):
        if not debug:
            return func(*args, **kwargs)
        start = time.perf_counter()
        try:
            result = func(*args, debug=debug, **kwargs)
        except TypeError:
            result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"[{func.__name__}] {elapsed:.4f}s")
        return result
    return wrapper

def gpu_memory_tracker(func):
    @functools.wraps(func)
    def wrapper(*args, debug=False, **kwargs):
        if not debug:
            return func(*args, **kwargs)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            peak_memory_init = torch.cuda.max_memory_allocated() / (1<<20)  # Convert to MB
            torch.cuda.synchronize()
        
        try:
            result = func(*args, debug=debug, **kwargs)
        except TypeError:
            result = func(*args, **kwargs)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            peak_memory = torch.cuda.max_memory_allocated() / (1<<20)  # Convert to GB
            print(f"[{func.__name__}] Peak GPU memory internal: {peak_memory-peak_memory_init:.2f} MB")
            print(f"[{func.__name__}] Peak GPU memory total: {peak_memory:.2f} MB")
        
        return result
    return wrapper

import os
import subprocess
