import torch

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
