
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

def compute_mle(dists, k, fixnan, unbiased=False):
    '''
    Note: if unbiased is True, the MLE estimate is multiplied by a correction factor to make it unbiased.
       the factor accounts for
        * (k-1)/(k-2) factor to make one anchor point estimate unbiased (see Eq.10 in https://www.stat.berkeley.edu/~bickel/mldim.pdf)
        * for harmonic means across anchors, which introduces a negrative bias that can be bounded above and below.
        The harmonic means bias corrective factor can be bracketed between $1$ and $\\frac{1}{1-\\var[\\hat m(x]/m^2}$ which can be combined
        with the estimate of the variance of the MLE estimator $\\var(\\hat m(x))= \\frac{m^2}{k-3}$ to give the correction factor of at most $1/(1-\\frac{1}{k-3}) = (k-3)/(k-4)$.
    '''
    assert torch.all(dists>=0), 'Beware some dists are negative'
    k = min(dists.shape[-1], k)
    if fixnan:
        inv_dim = torch.sum(torch.log(dists[:,:,:, k - 1: k] / dists[:,:,:, 0:k - 1]), axis=-1)/(k-1)
        inv_dim_est = masked_mean(inv_dim, axis=-1)
    else:
        inv_dim = torch.log(dists[:,:,:, k - 1: k] / dists[:,:,:, 0:k - 1]).sum(axis=-1)/(k-1)
        inv_dim_est = inv_dim.mean(axis=-1)
    dim_est = 1/inv_dim_est
    if unbiased:
        dim_est *= (k-1)/(k-2)/(1-1/(k-3)) #!note correction factor for harmonic means
    return dim_est, inv_dim_est

def compute_mle_averaged_over_k(dists, kmin=None, kmax=None, fixnan=True):
    if kmax is None:
        kmax = dists.shape[-1]
    if kmin is None:
        kmin = 5
    assert kmin >= 5, f'Beware kmin should larger than 5 (to divide by (1-1/(k-3)) and here {kmin=}'
    def inv_est(k):
        _, inv_dims = compute_mle(dists, k, fixnan)
        return inv_dims

    avg_est = 1/(sum(inv_est(k) for k in range(kmin, kmax+1))/(kmax+1-kmin))
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
