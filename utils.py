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
         .permute(2, 4, 0, 1, 3, 5)    # (Ph, Pw, N,  C, p, p)
         .reshape(Ph, Pw, N, -1)       # (Ph, Pw, N,  C * p * p)
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


import numpy as np
def masked_sum(tensor, dim=None, axis=None):
    dim = dim if dim is not None else (axis if axis is not None else 0)
    mask = torch.isnan(tensor) | torch.isinf(tensor)
    return tensor.masked_fill(mask, 0).sum(dim=dim)

def masked_mean(tensor, dim=None, axis=None):
    dim = dim if dim is not None else (axis if axis is not None else 0)
    mask = torch.isnan(tensor) | torch.isinf(tensor)
    return tensor.masked_fill(mask, 0).sum(dim=dim) / (~mask).sum(dim=dim).float()


def compute_mle(dists, k=10, fixnan=False):
    assert torch.all(dists>=0), 'Beware some dists are negative'
    k = min(dists.shape[-1], k)
    if fixnan:
        inv_dim = torch.sum(torch.log(dists[:,:,:, k - 1: k] / dists[:,:,:, 0:k - 1]), axis=-1)/(k-2)
        inv_dim_est = masked_mean(inv_dim, axis=-1)
    else:
        inv_dim = torch.log(dists[:,:,:, k - 1: k] / dists[:,:,:, 0:k - 1]).sum(axis=-1)/(k-2)
        inv_dim_est = inv_dim.mean(axis=-1)
    dim_est = 1/inv_dim_est
    return dim_est, inv_dim_est

def compute_mle_averaged_over_k(dists, kmin=None, kmax=None, fixnan=False):
    if kmax is None:
        kmax = dists.shape[-1]
    if kmin is None:
        kmin = 3
    assert kmin >= 3, f'Beware kmin should larger than 3 (to divide by k-2) and here {kmin=}'
    def inv_est(k):
        _, inv_dims = compute_mle(dists, k, fixnan)
        return inv_dims

    avg_est = 1/(sum(inv_est(k) for k in range(kmin, kmax))/(kmax-kmin))
    return avg_est

