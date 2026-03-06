import functools
import torch
import time

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
            peak_memory_init = torch.cuda.max_memory_allocated() / (1<<20)  # Convert to GB
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

def load_cats():
    file_path = './cats_tensor.pt'
    if not os.path.exists(file_path):
        subprocess.run(["python", "download_afhq.py"], check=True)
    return torch.load(file_path)
    

def images_to_patches(x: torch.Tensor, patch_size: int = 32) -> torch.Tensor:
    N, C, H, W = x.shape
    p = patch_size
    Ph, Pw = H // p, W // p  # number of patches: 32x32 = 1024

    return (
        x.reshape(N, C, Ph, p, Pw, p)  # (N, C, Ph, p, Pw, p)
         .permute(3, 5, 0, 1, 2, 4)    # (p, p, N,  C, Ph, Pw)
         .reshape(p, p, N, -1)         # (p, p, N,  C * Ph * Pw)
    )

from torch.utils.data import DataLoader
def make_loader(x: torch.Tensor, batch_dim: int=-2, **kwargs) -> DataLoader:
    batch_dim = batch_dim % x.ndim

    x_perm = x.moveaxis(batch_dim, 0)

    def collate(samples):
        # stack along 0, then move back to batch_dim
        stacked = torch.stack(samples, dim=0)        # (8, 10, 3, 2)
        return stacked.moveaxis(0, batch_dim)        # (10, 3, 8, 2)

    return DataLoader(x_perm, collate_fn=collate, **kwargs)
