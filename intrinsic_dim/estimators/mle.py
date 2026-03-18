
import torch
from intrinsic_dim.neighbors.patch_knn import patch_topk_dists

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

    avg_est = 1/(sum(inv_est(k) for k in range(kmin, kmax+1))/(kmax-kmin))
    return avg_est

def compute_mle_dims(samples: torch.Tensor, k: int = 10, n_anchors: int = 1000):
    """
    Estimate intrinsic dimensionality via MLE (Levina-Bickel) for every patch position.

    Args:
        samples:   (Ph, Pw, N, full_dim) — same convention as images_to_patches
        k:         number of nearest neighbours for MLE (default 10)
        n_anchors: number of randomly selected query points (batch dim is -2)

    Returns:
        mle_dims:     (Ph, Pw) tensor, MLE estimate at fixed k
        mle_avg_dims: (Ph, Pw) tensor, MLE estimate averaged over k in [3, k]
    """
    nb_h, nb_w, n_samples, _ = samples.shape
    n_anchors = min(n_anchors, n_samples)

    # random anchor indices, same draw for all patch positions
    idx      = torch.randperm(n_samples)[:n_anchors]
    anchors  = samples[:, :, idx, :]   # (Ph, Pw, n_anchors, full_dim)

    # (Ph, Pw, n_anchors, k) — anchor queries into full samples, excluding self-hit
    dists = patch_topk_dists(anchors, samples, k=k + 1, remove_self=True)

    mle_dims,     _ = compute_mle(dists, k=k, fixnan=True)
    mle_avg_dims    = compute_mle_averaged_over_k(dists, kmax=k)
    return mle_dims, mle_avg_dims
