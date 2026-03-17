import functools
import torch
import time
import tqdm

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

def patches_to_images(patches: torch.Tensor, patch_size: int = 32) -> torch.Tensor:
    """(Ph, Pw, N, C*p*p) -> (N, C, H, W) — exact inverse of images_to_patches."""
    Ph, Pw, N, Cpp = patches.shape
    p = patch_size
    C = Cpp // (p * p)
    return (
        patches.reshape(Ph, Pw, N, C, p, p)   # (Ph, Pw, N, C, p, p)
               .permute(2, 3, 0, 4, 1, 5)     # (N, C, Ph, p, Pw, p)
               .reshape(N, C, Ph * p, Pw * p)  # (N, C, H, W)
    )

def test_patch_vs_image(patch_size=8, nb_channels=3, img_size=32, n_images=5):
    # ── round-trip: image -> patches -> image ────────────────────────────────
    x = torch.randn(n_images, nb_channels, img_size, img_size)
    patches     = images_to_patches(x, patch_size)
    x_recovered = patches_to_images(patches, patch_size)

    Ph, Pw = img_size // patch_size, img_size // patch_size
    assert patches.shape == (Ph, Pw, n_images, nb_channels * patch_size * patch_size), \
        f"images_to_patches shape mismatch: {patches.shape}"
    assert x_recovered.shape == x.shape, \
        f"patches_to_images shape mismatch: {x_recovered.shape}"
    assert torch.allclose(x, x_recovered), \
        f"Round-trip image->patches->image failed, max error: {(x - x_recovered).abs().max():.2e}"

    # ── round-trip: patches -> image -> patches ───────────────────────────────
    p = torch.randn(Ph, Pw, n_images, nb_channels * patch_size * patch_size)
    p_recovered = images_to_patches(patches_to_images(p, patch_size), patch_size)

    assert p_recovered.shape == p.shape, \
        f"images_to_patches shape mismatch on reverse round-trip: {p_recovered.shape}"
    assert torch.allclose(p, p_recovered), \
        f"Round-trip patches->image->patches failed, max error: {(p - p_recovered).abs().max():.2e}"

    print(f"test_patch_vs_image passed  |  "
          f"img ({n_images}, {nb_channels}, {img_size}, {img_size})  "
          f"patch_size={patch_size}  patches ({Ph}, {Pw}, {n_images}, {nb_channels * patch_size**2})")


if __name__ == "__main__":
    test_patch_vs_image(patch_size=8, nb_channels=3, img_size=32, n_images=5)


from torch.utils.data import DataLoader
def make_loader(x: torch.Tensor, batch_dim: int = -2, **kwargs) -> DataLoader:
    use_tqdm = kwargs.pop('tqdm', False)
    transform = kwargs.pop('transform', None)
    device = kwargs.pop('device', None)
    batch_dim = batch_dim % x.ndim
    x_perm = x.moveaxis(batch_dim, 0)
    def collate(samples):
        stacked = torch.stack(samples, dim=0)
        out = stacked.moveaxis(0, batch_dim)
        if device: out = out.to(device)
        if transform: out = transform(out)
        return out
    loader = DataLoader(x_perm, collate_fn=collate, **kwargs)
    return tqdm.tqdm(loader, leave=False) if use_tqdm else loader

