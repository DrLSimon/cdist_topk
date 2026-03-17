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


def compute_mle_dims_sample_variance(sample_pool: torch.Tensor, k: int, n_anchors: int,
                                      n_subsample: int, n_trials: int):
    '''
        Estimate the variance of (per anchor) MLE dimension estimates.  It
        involves two source of varaince:
        * Variance across random subsamples of the reference set, Variance
        * across random anchors (queries) for a fixed reference set.

        Note: reference sets are drawn with replacement from the remaining pool,
        which introduces positive correlation between trials and causes the
        empirical variance to underestimate the true per-sample variance.  

        Note:
        this estimate is supposed to be compared to Eq.10 in
        https://www.stat.berkeley.edu/~bickel/mldim.pdf which reports a 
        Var[\hat m(x)]= \frac{m^2}{k} 
        where \hat m(x) is the MLE dimension estimate for a single anchor x, m is
        the true dimension and k is the number of neighbors used in the
        estimation. 
    '''
def compute_mle_dims_sample_variance(sample_pool: torch.Tensor, k: int, n_anchors: int,
                                      n_subsample: int, n_trials: int):
    '''
        Estimate the variance of (per anchor) MLE dimension estimates.  It
        involves two source of varaince:
        * Variance across random subsamples of the reference set, Variance
        * across random anchors (queries) for a fixed reference set.

        Note: reference sets are drawn with replacement from the remaining pool,
        which introduces positive correlation between trials and causes the
        empirical variance to underestimate the true per-sample variance.  

        Note:
        this estimate is supposed to be compared to Eq.10 in
        https://www.stat.berkeley.edu/~bickel/mldim.pdf which reports a 
        Var[\hat m(x)]= \frac{m^2}{k} 
        where \hat m(x) is the MLE dimension estimate for a single anchor x, m is
        the true dimension and k is the number of neighbors used in the
        estimation. 
    '''
    *_, n_samples, _ = sample_pool.shape
    assert n_anchors + n_subsample <= n_samples, (
        f"n_anchors + n_subsample ({n_anchors + n_subsample}) must be <= n_samples ({n_samples})"
    )

    # Fix anchors once
    perm          = torch.randperm(n_samples)
    anchor_idx    = perm[:n_anchors]
    remaining_idx = perm[n_anchors:]

    anchors = sample_pool[:, :, anchor_idx, :]  # (Ph, Pw, n_anchors, D)

    # --- source 1: variance due to reference subsampling (fixed anchors) ---
    inv_dim_trials = []
    for _ in range(n_trials):
        ref_idx = remaining_idx[torch.randperm(len(remaining_idx))[:n_subsample]]
        ref     = sample_pool[:, :, ref_idx, :]
        dists   = patch_topk_dists(anchors, ref, k=k, remove_self=False)
        inv_dim = torch.log(dists[:, :, :, k-1:k] / dists[:, :, :, 0:k-1]).sum(dim=-1) / (k-2)
        inv_dim_trials.append(inv_dim)

    inv_dim_trials = torch.stack(inv_dim_trials, dim=0)  # (n_trials, Ph, Pw, n_anchors)
    ref_var = inv_dim_trials.var(dim=0).mean(dim=-1)     # (Ph, Pw)

    # --- source 2: variance across anchors (fixed reference) ---
    ref_idx = remaining_idx[torch.randperm(len(remaining_idx))[:n_subsample]]
    ref     = sample_pool[:, :, ref_idx, :]
    dists   = patch_topk_dists(anchors, ref, k=k, remove_self=False)
    inv_dim = torch.log(dists[:, :, :, k-1:k] / dists[:, :, :, 0:k-1]).sum(dim=-1) / (k-2)
    anchor_var = inv_dim.var(dim=-1)                     # (Ph, Pw)

    # total per-sample variance in inv_dim space, converted to dim space
    total_inv_dim_var = ref_var + anchor_var
    mean_inv_dim      = inv_dim_trials.mean(dim=(0, -1)) # (Ph, Pw)
    dim_sample_var    = total_inv_dim_var / (mean_inv_dim ** 4)

    return dim_sample_var

def check_poisson_regime(samples: torch.Tensor, k: int = 10, n_anchors: int = 1000,
                          n_subsample: int = None,
                          k_over_n_threshold: float = 0.01,
                          ks_pvalue_threshold: float = 0.05,
                          r_ratio_threshold: float = 2.0) -> tuple[dict, torch.Tensor]:
    """
    Evaluate whether the Poisson regime assumption is valid for the MLE estimator.

    Two checks are performed:
    1. k/n ratio: should be << 1 (formally k/n -> 0)
    2. Kolmogorov-Smirnov test of {m * log(r_k/r_j)}_{j=1}^{k-1} against Exp(1)
       order statistics, where m is estimated from the data via MLE.

    Args:
        samples:              (Ph, Pw, N, full_dim)
        k:                    number of nearest neighbours
        n_anchors:            number of query points
        n_subsample:          reference set size (defaults to all samples)
        k_over_n_threshold:   maximum acceptable k/n ratio (default: 0.01)
        ks_pvalue_threshold:  minimum acceptable mean KS p-value (default: 0.05)
        r_ratio_threshold:    maximum acceptable mean r_k/r_1 (default: 2.0)

    Returns:
        is_valid: (Ph, Pw) bool tensor, True where all conditions are satisfied
        stats:    dict with keys 'k_over_n', 'ks_stat', 'ks_pvalue', 'r_ratio',
                  each a (Ph, Pw) tensor
    """
    from scipy.stats import kstest, expon
    import numpy as np

    nb_h, nb_w, n_samples, _ = samples.shape
    n_anchors   = min(n_anchors, n_samples)
    n_subsample = n_subsample or n_samples

    idx     = torch.randperm(n_samples)[:n_anchors]
    anchors = samples[:, :, idx, :]

    dists = patch_topk_dists(anchors, samples, k=k + 1, remove_self=True)
    # dists: (Ph, Pw, n_anchors, k)

    mle_dims, _ = compute_mle(dists, k=k, fixnan=True)  # (Ph, Pw)

    k_over_n  = torch.full((nb_h, nb_w), k / n_subsample)
    ks_stat   = torch.zeros(nb_h, nb_w)
    ks_pvalue = torch.zeros(nb_h, nb_w)
    r_ratio   = torch.zeros(nb_h, nb_w)

    for i in range(nb_h):
        for j in range(nb_w):
            m_est = mle_dims[i, j].item()
            d     = dists[i, j]                          # (n_anchors, k)

            r_ratio[i, j] = (d[:, -1] / d[:, 0]).mean()

            stats, pvals = [], []
            for a in range(d.shape[0]):
                scaled = m_est * torch.log(d[a, -1] / d[a, :-1]).cpu().numpy()
                stat, pval = kstest(np.sort(scaled), expon.cdf)
                stats.append(stat)
                pvals.append(pval)

            ks_stat[i, j]   = float(np.mean(stats))
            ks_pvalue[i, j] = float(np.mean(pvals))

    is_valid = (
        (k_over_n  <  k_over_n_threshold) &
        (ks_pvalue >  ks_pvalue_threshold) &
        (r_ratio   <  r_ratio_threshold)
    )

    stats = {
        'k_over_n':  k_over_n,
        'ks_stat':   ks_stat,
        'ks_pvalue': ks_pvalue,
        'r_ratio':   r_ratio,
    }

    return is_valid, stats 

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