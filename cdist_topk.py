import torch
import time
from math import prod

from functools import wraps

import functools

def print_call(fn):
    @wraps(fn)
    def wrapper(*args, debug=False, **kwargs):
        if debug:
            print(f"[CALL] {fn.__name__}")
        result = fn(*args, **kwargs)
        if debug:
            print(f"[DONE] {fn.__name__}")
        return result
    return wrapper

# ============================================================
# Reference implementation
# ============================================================

@print_call
def reference_cdist_topk(x, y, k):
    d = torch.cdist(x, y)
    return torch.topk(d, k, dim=-1, largest=False)


# ============================================================
# Version 1: Simple chunked implementation
# ============================================================

@print_call
def chunked_cdist_topk_simple(x, y, k, chunk_size=1024):
    *batch_dims, B, F = x.shape

    best_vals = None
    best_idx = None

    for start in range(0, B, chunk_size):
        end = min(start + chunk_size, B)

        d_chunk = torch.cdist(x, y[..., start:end, :])  # (..., B, chunk)

        vals, idx = torch.topk(
            d_chunk,
            k=min(k, d_chunk.shape[-1]),
            dim=-1,
            largest=False,
        )

        idx = idx + start

        if best_vals is None:
            best_vals = vals
            best_idx = idx
        else:
            all_vals = torch.cat([best_vals, vals], dim=-1)
            all_idx = torch.cat([best_idx, idx], dim=-1)

            best_vals, order = torch.topk(all_vals, k, dim=-1, largest=False)
            best_idx = torch.gather(all_idx, -1, order)

    return best_vals, best_idx


# ============================================================
# Version 2: Preallocated / GPU-friendlier
# ============================================================

@print_call
def chunked_cdist_topk_prealloc(x, y, k, chunk_size=1024):
    *batch_dims, B, F = x.shape
    device = x.device
    dtype = x.dtype

    best_vals = torch.full((*batch_dims, B, k), float("inf"), device=device, dtype=dtype)
    best_idx = torch.full((*batch_dims, B, k), -1, device=device, dtype=torch.long)

    for start in range(0, B, chunk_size):
        end = min(start + chunk_size, B)

        d_chunk = torch.cdist(x, y[..., start:end, :])

        vals, idx = torch.topk(
            d_chunk,
            k=min(k, d_chunk.shape[-1]),
            dim=-1,
            largest=False,
        )

        idx = idx + start

        combined_vals = torch.cat([best_vals, vals], dim=-1)
        combined_idx = torch.cat([best_idx, idx], dim=-1)

        best_vals, order = torch.topk(combined_vals, k, dim=-1, largest=False)
        best_idx = torch.gather(combined_idx, -1, order)

    return best_vals, best_idx


# ============================================================
# Deterministic correctness tests
# ============================================================

def test_correctness():
    torch.manual_seed(0)

    shapes = [
        (8, 16, 32),
        (4, 5, 12, 32),
        (3, 4, 5, 20, 16),
    ]

    k = 5

    for shape in shapes:
        x = torch.randn(*shape)
        y = torch.randn(*shape)

        ref_vals, ref_idx = reference_cdist_topk(x, y, k)

        vals1, idx1 = chunked_cdist_topk_simple(x, y, k, chunk_size=7)
        vals2, idx2 = chunked_cdist_topk_prealloc(x, y, k, chunk_size=7)

        assert torch.allclose(ref_vals, vals1, atol=1e-6)
        assert torch.equal(ref_idx, idx1)

        assert torch.allclose(ref_vals, vals2, atol=1e-6)
        assert torch.equal(ref_idx, idx2)

    print("✓ deterministic correctness tests passed")



# ============================================================
# Benchmark
# ============================================================

def benchmark(device="cpu"):
    """
    Stronger benchmark showing:
    - larger B (where chunking matters)
    - averaged timings
    - memory comparison
    """

    shape = (3, 2, 16000, 64)  # D1xD2xBxF with large B to stress memory
    k = 10
    chunk_size = 512
    iters = 10

    print(f"\nBenchmark on {device}")
    print(f"shape={shape}, k={k}, chunk_size={chunk_size}\n")

    x = torch.randn(*shape, device=device)
    y = torch.randn(*shape, device=device)

    *Ds, B, F = shape
    batch = prod(Ds)

    dtype_size = x.element_size()

    full_mem = batch * B * B * dtype_size / 1e9
    chunk_mem = batch * B * (2*chunk_size+k) * dtype_size / 1e9

    print(f"Estimated full cdist memory  : {full_mem:.2f} GB")
    print(f"Estimated chunked memory     : {chunk_mem:.2f} GB\n")

    def time_fn(fn, name):
        print(f"[CALL] {name}")
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        t0 = time.time()
        for _ in range(iters):
            fn()

        if device == "cuda":
            torch.cuda.synchronize()
            peak_mem = torch.cuda.max_memory_allocated() / 1024**2
            print(f"{name:25s} | peak GPU memory: {peak_mem:.1f} MiB")

        print(f"[DONE] {name:20s}: {(time.time() - t0) / iters:.6f}s")


    benchmarks = [
        ("chunked simple", lambda: chunked_cdist_topk_simple(x, y, k)),
        ("chunked prealloc", lambda: chunked_cdist_topk_prealloc(x, y, k)),
        ("full cdist + topk", lambda: reference_cdist_topk(x, y, k)),
    ]

    for name, fn in benchmarks:
        fn()
        time_fn(fn, name)

# ============================================================
# Run everything
# ============================================================

if __name__ == "__main__":
    test_correctness()

    #benchmark("cpu")

    if torch.cuda.is_available():
        benchmark("cuda")
