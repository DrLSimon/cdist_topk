
import torch
from intrinsic_dim.estimators.mle_estimator import compute_mle_dims
from intrinsic_dim.neighbors.patch_knn import patch_topk_dists

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
