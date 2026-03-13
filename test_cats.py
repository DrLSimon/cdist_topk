import torch
from knn import raw_topk_dists, compute_dataset_topk_dists
from utils import load_cats, make_loader, images_to_patches


def test_correctness(all_cats, patch_size=32):
    some_cats = all_cats[:100]
    some_cats_patches = images_to_patches(some_cats, patch_size).to('cuda').float() / 255
    ref_dists = raw_topk_dists(some_cats_patches, some_cats_patches, k=25)

    patches = images_to_patches(some_cats, patch_size)
    loaderx = make_loader(patches, batch_dim=-2, batch_size=7,
                          device='cuda', transform=lambda x: x.float() / 255)
    loadery = make_loader(patches, batch_dim=-2, batch_size=9,
                          device='cuda', transform=lambda x: x.float() / 255)
    topk_dists = compute_dataset_topk_dists(loaderx, loadery, k=25)

    assert torch.allclose(ref_dists, topk_dists, atol=1e-6)
    print("✓ correctness tests passed")


def main():
    all_cats = load_cats()
    test_correctness(all_cats)

    patches = images_to_patches(all_cats, patch_size=32)
    loaderx = make_loader(patches, batch_dim=-2, batch_size=256, tqdm=True,
                          device='cuda', transform=lambda x: x.float() / 255)
    loadery = make_loader(patches, batch_dim=-2, batch_size=128,
                          device='cuda', transform=lambda x: x.float() / 255)
    all_topk_dists = compute_dataset_topk_dists(loaderx, loadery, k=25)
    torch.save(all_topk_dists, 'top25_dists.pt')


if __name__ == '__main__':
    main()