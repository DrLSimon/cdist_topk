import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from intrinsic_dim.neighbors.knn import raw_topk_dists, compute_dataset_topk_dists
from intrinsic_dim.data.afhq import load_cats
from intrinsic_dim.data.loaders import make_loader
from intrinsic_dim.data.patches import images_to_patches


def test_correctness(all_cats, patch_size=32):
    some_cats = all_cats[:100]
    some_cats_patches = images_to_patches(some_cats, patch_size).to('cuda' if torch.cuda.is_available() else 'cpu').float() / 255
    ref_dists = raw_topk_dists(some_cats_patches, some_cats_patches, k=25)

    patches = images_to_patches(some_cats, patch_size)
    loaderx = make_loader(patches, batch_dim=-2, batch_size=7,
                          device=('cuda' if torch.cuda.is_available() else 'cpu'), transform=lambda x: x.float() / 255)
    loadery = make_loader(patches, batch_dim=-2, batch_size=9,
                          device=('cuda' if torch.cuda.is_available() else 'cpu'), transform=lambda x: x.float() / 255)
    topk_dists = compute_dataset_topk_dists(loaderx, loadery, k=25)

    assert torch.allclose(ref_dists, topk_dists, atol=1e-6)
    print("✓ correctness tests passed")


def main():
    all_cats = load_cats()
    test_correctness(all_cats)

    patches = images_to_patches(all_cats, patch_size=32)
    loaderx = make_loader(patches, batch_dim=-2, batch_size=256, tqdm=True,
                          device=('cuda' if torch.cuda.is_available() else 'cpu'), transform=lambda x: x.float() / 255)
    loadery = make_loader(patches, batch_dim=-2, batch_size=128,
                          device=('cuda' if torch.cuda.is_available() else 'cpu'), transform=lambda x: x.float() / 255)
    all_topk_dists = compute_dataset_topk_dists(loaderx, loadery, k=25)
    torch.save(all_topk_dists, 'top25_dists.pt')


if __name__ == '__main__':
    main()