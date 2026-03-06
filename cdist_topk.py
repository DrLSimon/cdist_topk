import torch
import time
from math import prod

# ============================================================
# Reference implementation
# ============================================================
def reference_cdist_topk(x, y, k, approx=True):
    if approx == False:
        def cdist(x,y):
            return torch.cdist(x, y, compute_mode='donot_use_mm_for_euclid_dist')
    else:
        def cdist(x,y):
            return torch.cdist(x, y)
    d = cdist(x,y)
    return torch.topk(d, k, dim=-1, largest=False)


# ============================================================
# Chunked implementation
# ============================================================
def chunked_cdist_topk(x, y, k, chunk_size, approx=True):
    if approx == False:
        def cdist(x,y):
            return torch.cdist(x, y, compute_mode='donot_use_mm_for_euclid_dist')
    else:
        def cdist(x,y):
            return torch.cdist(x, y)
    *batch_dims, B, F = y.shape

    best_vals = None
    best_idx = None

    for start in range(0, B, chunk_size):
        end = min(start + chunk_size, B)

        d_chunk = cdist(x, y[..., start:end, :])  # (..., B, chunk)

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

            kb=min(k, all_vals.shape[-1])
            best_vals, order = torch.topk(all_vals, kb, dim=-1, largest=False)
            best_idx = torch.gather(all_idx, -1, order)

    return best_vals, best_idx


def iterative_chunked_cdist_topk(x, y, k, chunk_size, approx=True):
    best_vals = None
    for a,b in zip(x,y):
        vals, idx = chunked_cdist_topk(a.unsqueeze(0), b.unsqueeze(0), k, chunk_size, approx)
        if best_vals is None:
            best_vals = vals
            best_idx = idx
            continue
        best_vals = torch.cat([best_vals, vals], dim=0)
        best_idx = torch.cat([best_idx, idx], dim=0)
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
        shapey = list(shape)
        shapey[-2]+=2 #! bug when I use a differnt size for y the test fail
        y = torch.randn(*shapey)

        ref_vals, ref_idx = reference_cdist_topk(x, y, k, approx=False)

        vals, idx = chunked_cdist_topk(x, y, k, chunk_size=7)
        assert torch.allclose(ref_vals, vals, atol=1e-6)
        assert torch.equal(ref_idx, idx)

        vals, idx = iterative_chunked_cdist_topk(x, y, k, chunk_size=7)
        assert torch.allclose(ref_vals, vals, atol=1e-6)
        assert torch.equal(ref_idx, idx)

    print("✓ deterministic correctness tests passed")


def estimate_memory_footprint(x, y, chunk_size, k):
    dtype_size = x.element_size()
    *Ds, Bx, F = x.shape
    *Ds, By, F = y.shape
    batch_numel = prod(Ds)


    full_mem = batch_numel * Bx * By * dtype_size / (1<<30)
    chunk_mem = batch_numel * Bx * (2*chunk_size+k) * dtype_size / (1<<30)
    return full_mem, chunk_mem


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
    shape = (3, 32, 32, 400, 256)  # D1xD2xBxF with large B to stress memory
    k = 10
    chunk_size = 50#2048

    print(f"\nBenchmark on {device}")
    print(f"shape={shape}, k={k}, chunk_size={chunk_size}\n")

    x = torch.randn(*shape, device=device)
    y = torch.randn(*shape, device=device)

    full_mem, chunk_mem = estimate_memory_footprint(x, y, chunk_size, k)

    print(f"Estimated full cdist memory  : {full_mem:.2f} GB")
    print(f"Estimated chunked memory     : {chunk_mem:.2f} GB")

    def time_fn(fn):
        if device == "cuda":
            torch.cuda.empty_cache() # so that nvidia-smi "reacts"
            torch.cuda.reset_peak_memory_stats()
            peak_mem0 = torch.cuda.max_memory_allocated() / (1<<30)
            torch.cuda.synchronize()

        iters = 5
        t0 = time.time()
        for _ in range(iters):
            fn()

        if device == "cuda":
            torch.cuda.synchronize()
            peak_mem = torch.cuda.max_memory_allocated() / (1<<30)
            print(f"[MEM]: peak GPU memory: {peak_mem-peak_mem0:.1f} GB")

        print(f"[TIME]: {(time.time() - t0) / iters:.6f}s")


    benchmarks = [
        ("full cdist + topk", lambda: reference_cdist_topk(x, y, k)),
        ("chunked version  ", lambda: chunked_cdist_topk(x, y, k, chunk_size=chunk_size)),
        ("iterative version  ", lambda: iterative_chunked_cdist_topk(x, y, k, chunk_size=chunk_size)),
    ]

    for name, fn in benchmarks:
        # warmup 
        fn()
        print(f"\n[CALL] {name}")
        time_fn(fn)

# ============================================================
# Run everything
# ============================================================
if __name__ == "__main__":
    test_correctness()

    if torch.cuda.is_available():
        benchmark("cuda")
    else:
        benchmark("cpu")
