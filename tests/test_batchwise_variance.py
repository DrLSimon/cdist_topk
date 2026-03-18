from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from intrinsic_dim.data.patches import images_to_patches
from intrinsic_dim.data.loaders import make_loader
from intrinsic_dim.data.afhq import load_cats


def compute_patch_variance_batch(all_cats, bsx=256, bsy=128):
    all_cats_patches = images_to_patches(all_cats)
    loaderx = make_loader(all_cats_patches, batch_dim=-2, batch_size=bsx)
    loadery = make_loader(all_cats_patches, batch_dim=-2, batch_size=bsy)

    total = None
    count = 0
    for x in loaderx:
        x = x.float() / 255
        for y in loadery:
            y = y.float() / 255
            sq_dists = torch.cdist(x, y) ** 2
            total = sq_dists.sum(dim=(-1, -2)) if total is None else total + sq_dists.sum(dim=(-1, -2))
            count += x.shape[-2] * y.shape[-2]

    return total / (2 * count)


def compute_patch_variance_simple(all_cats):
    patches = images_to_patches(all_cats.float() / 255)
    return patches.var(dim=-2, unbiased=False).sum(dim=-1)


def test_correctness(all_cats):
    some_cats = all_cats[:100]
    ref_vars = compute_patch_variance_simple(some_cats)
    other_vars = compute_patch_variance_batch(some_cats, bsx=9, bsy=11)
    assert torch.allclose(ref_vars, other_vars, atol=1e-6)


def main():
    all_cats = load_cats()
    test_correctness(all_cats)
    print("✓ test_correctness passed")


if __name__ == '__main__':
    main()
