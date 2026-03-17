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


def compute_mle_dims_variance(sample_pool, k, n_anchors, n_subsample, n_trials):
    mle_dims_trials = []
    for _ in range(n_trials):
        # Subsample defines the REFERENCE set
        ref_idx = torch.randperm(sample_pool.shape[-2])[:n_subsample]
        ref_samples     = sample_pool[:, :, ref_idx, :]

        # Anchors are a subset of the reference — self-exclusion is safe
        mle_dims, _ = compute_mle_dims(ref_samples, k=k, n_anchors=n_anchors)
        mle_dims_trials.append(mle_dims)
    mle_dims_trials = torch.stack(mle_dims_trials, dim=0)  # (n_trials, Ph, Pw)
    mle_dims_var = mle_dims_trials.var(dim=0)  # (Ph, Pw)
    return mle_dims_var

def compute_mle_dims_sample_variance(sample_pool: torch.Tensor, k: int, n_anchors: int, n_subsample: int, n_trials: int):
    nb_h, nb_w, n_samples, _ = sample_pool.shape
    assert n_anchors + n_subsample <= n_samples, (
        f"n_anchors + n_subsample ({n_anchors + n_subsample}) must be <= n_samples ({n_samples})"
    )

    # Fix anchors once, reserve the rest as the pool for reference subsampling
    perm          = torch.randperm(n_samples)
    anchor_idx    = perm[:n_anchors]
    remaining_idx = perm[n_anchors:]  # size: n_samples - n_anchors

    anchors = sample_pool[:, :, anchor_idx, :]  # (Ph, Pw, n_anchors, D)

    inv_dim_trials = []
    for _ in range(n_trials):
        ref_idx = remaining_idx[torch.randperm(len(remaining_idx))[:n_subsample]]
        ref     = sample_pool[:, :, ref_idx, :]

        dists   = patch_topk_dists(anchors, ref, k=k, remove_self=False)

        inv_dim = torch.log(dists[:, :, :, k - 1:k] / dists[:, :, :, 0:k - 1]).sum(dim=-1) / (k - 2)
        inv_dim_trials.append(inv_dim)

    inv_dim_trials = torch.stack(inv_dim_trials, dim=0)  # (n_trials, Ph, Pw, n_anchors)

    per_anchor_var = inv_dim_trials.var(dim=0)            # (Ph, Pw, n_anchors)
    mean_var       = per_anchor_var.mean(dim=-1)          # (Ph, Pw)

    mean_inv_dim   = inv_dim_trials.mean(dim=(0, -1))     # (Ph, Pw)
    dim_sample_var = mean_var / (mean_inv_dim ** 4)

    return dim_sample_var

# ── PCA utils ────────────────────────────────────────────────────────────────
def pca_effective_dim(samples: torch.Tensor, threshold: float = 0.999):
    """
    samples: (n_samples, full_dim)
    Returns effective_dim
    """
    S = samples - samples.mean(dim=0)
    _, sv, _ = torch.linalg.svd(S, full_matrices=False)
    var_ratio = (sv ** 2) / (sv ** 2).sum()
    return int(torch.searchsorted(var_ratio.cumsum(dim=0), threshold).item()) + 1


def compute_pca_dims(samples: torch.Tensor, threshold: float = 0.999):
    """
    Compute PCA effective dimensionality for every patch position.
    Args:
        samples:   (Ph, Pw, N, C*p*p) — same convention as images_to_patches
        threshold: cumulative variance threshold (default 0.95)
    Returns:
        pca_dims: (nb_h, nb_w) torch.Tensor of effective dims
    """
    nb_h, nb_w, _, _ = samples.shape
    pca_dims = torch.zeros((nb_h, nb_w), dtype=torch.int32, device=samples.device)
    for i in range(nb_h):
        for j in range(nb_w):
            pca_dims[i, j] = pca_effective_dim(
                samples[i, j, :, :], threshold
            )
    return pca_dims