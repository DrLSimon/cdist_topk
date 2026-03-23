"""
MLE variance diagnostics.

compute_mle_dims_sample_variance — estimates variance of the per-anchor inv_dim
                                   intermediate quantity. Not part of the variance
                                   estimator protocol; exposed as a standalone utility.
"""

import torch
from intrinsic_dim.neighbors.patch_knn import patch_topk_dists


def compute_mle_dims_sample_variance(
    sample_pool: torch.Tensor,
    k: int,
    n_anchors: int,
    n_subsample: int,
    n_trials: int,
    total_variance: bool = False,
) -> torch.Tensor:
    """
    Diagnostic: estimate variance of the per-anchor inv_dim intermediate quantity.

    Involves two sources of variance:
    - Variance across random subsamples of the reference set
    - Variance across random anchors for a fixed reference set (if total_variance=True)

    Note: reference sets are drawn with replacement from the remaining pool,
    introducing positive correlation between trials and causing the empirical
    variance to underestimate the true per-sample variance.

    This is meant to be compared to Eq.10 in
    https://www.stat.berkeley.edu/~bickel/mldim.pdf:
        Var[m_hat(x)] = m^2 / (k-3)
    """
    *_, n_samples, _ = sample_pool.shape
    assert n_anchors + n_subsample <= n_samples, (
        f"n_anchors + n_subsample ({n_anchors + n_subsample}) must be <= n_samples ({n_samples})"
    )
    perm          = torch.randperm(n_samples)
    anchor_idx    = perm[:n_anchors]
    remaining_idx = perm[n_anchors:]
    anchors       = sample_pool[:, :, anchor_idx, :]  # (Ph, Pw, n_anchors, D)

    inv_dim_trials = []
    for _ in range(n_trials):
        ref_idx = remaining_idx[torch.randperm(len(remaining_idx))[:n_subsample]]
        ref     = sample_pool[:, :, ref_idx, :]
        dists   = patch_topk_dists(anchors, ref, k=k, remove_self=False)
        inv_dim = torch.log(dists[:, :, :, k-1:k] / dists[:, :, :, 0:k-1]).sum(dim=-1) / (k-2)
        inv_dim_trials.append(inv_dim)

    dim_trials = 1 / torch.stack(inv_dim_trials, dim=0)  # (n_trials, Ph, Pw, n_anchors)
    ref_var    = dim_trials.var(dim=0).mean(dim=-1)       # (Ph, Pw)

    if not total_variance:
        return ref_var

    anchor_var = dim_trials.var(dim=-1).mean(dim=0)       # (Ph, Pw)
    return ref_var + anchor_var