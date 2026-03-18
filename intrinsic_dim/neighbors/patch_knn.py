import torch
from intrinsic_dim.neighbors.knn import compute_dataset_topk_dists
from intrinsic_dim.data.loaders import make_loader
from intrinsic_dim.data.patches import images_to_patches

def _default_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

def patch_topk_dists(patches_x, patches_y, k=10, patch_size=32, remove_self=False, bsx=256, bsy=128, device=None):
    _k = k + 1 if remove_self else k
    device = device or _default_device()
    loaderx = make_loader(
        patches_x, batch_dim=-2, batch_size=bsx, tqdm=True, device=device,
        transform=lambda x: x.float() / 255 if x.dtype == torch.uint8 else x.float()
    )
    loadery = make_loader(
        patches_y, batch_dim=-2, batch_size=bsy, device=device,
        transform=lambda x: x.float() / 255 if x.dtype == torch.uint8 else x.float()
    )
    all_topk_dists = compute_dataset_topk_dists(loaderx, loadery, k=_k)
    return all_topk_dists[:, :, :, 1:] if remove_self else all_topk_dists

def image_topk_dists(images_x, images_y, k=10, patch_size=32, remove_self=False, bsx=256, bsy=128, device=None):
    patches_x = images_to_patches(images_x, patch_size)
    patches_y = images_to_patches(images_y, patch_size)
    return patch_topk_dists(patches_x, patches_y, k=k, patch_size=patch_size, remove_self=remove_self, bsx=bsx, bsy=bsy, device=device)
