import torch
from knn import compute_dataset_topk_dists
from utils import make_loader, images_to_patches


def patch_topk_dists(patches_x, patches_y, k=10, patch_size=32, remove_self=False, bsx=256, bsy=128):
    _k = k + 1 if remove_self else k
    loaderx = make_loader(patches_x, batch_dim=-2, batch_size=bsx, tqdm=True, device="cuda", transform=lambda x: x.float() / 255)
    loadery = make_loader(patches_y, batch_dim=-2, batch_size=bsy, device="cuda", transform=lambda x: x.float() / 255)
    all_topk_dists = compute_dataset_topk_dists(loaderx, loadery, k=_k)
    return all_topk_dists[:, :, :, 1:] if remove_self else all_topk_dists
 

def image_topk_dists(images_x, images_y, k=10, patch_size=32, remove_self=False, bsx=256, bsy=128):
    patches_x = images_to_patches(images_x, patch_size)
    patches_y = images_to_patches(images_y, patch_size)
    return patch_topk_dists(patches_x, patches_y, k=k, patch_size=patch_size, remove_self=remove_self, bsx=bsx, bsy=bsy)


def masked_sum(tensor, dim=None, axis=None):
    dim = dim if dim is not None else (axis if axis is not None else 0)
    mask = torch.isnan(tensor) | torch.isinf(tensor)
    return tensor.masked_fill(mask, 0).sum(dim=dim)

def masked_mean(tensor, dim=None, axis=None):
    dim = dim if dim is not None else (axis if axis is not None else 0)
    mask = torch.isnan(tensor) | torch.isinf(tensor)
    return tensor.masked_fill(mask, 0).sum(dim=dim) / (~mask).sum(dim=dim).float()

def compute_mle(dists, k, fixnan):
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

def compute_mle_averaged_over_k(dists, kmin=None, kmax=None, fixnan=True):
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

