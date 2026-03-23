"""MLE-based intrinsic dimension (Levina-Bickel) — pure computation functions."""

import torch


def masked_mean(tensor, dim=None, axis=None):
    dim = dim if dim is not None else (axis if axis is not None else 0)
    mask = torch.isnan(tensor) | torch.isinf(tensor)
    return tensor.masked_fill(mask, 0).sum(dim=dim) / (~mask).sum(dim=dim).float()


def compute_mle(dists, k, fixnan, unbiased):
    """
    Note: if unbiased is True, the MLE estimate is multiplied by a correction factor:
      * (k-1)/(k-2) to make one anchor point estimate unbiased (Eq.10 in
        https://www.stat.berkeley.edu/~bickel/mldim.pdf)
      * (k-3)/(k-4) upper-bounds the harmonic mean bias correction, derived from
        Var[m_hat(x)] = m^2/(k-3).
    """
    assert torch.all(dists >= 0), 'Beware some dists are negative'
    k = min(dists.shape[-1], k)
    if fixnan:
        inv_dim = torch.sum(torch.log(dists[:,:,:, k-1:k] / dists[:,:,:, 0:k-1]), axis=-1) / (k-1)
        inv_dim_est = masked_mean(inv_dim, axis=-1)
    else:
        inv_dim = torch.log(dists[:,:,:, k-1:k] / dists[:,:,:, 0:k-1]).sum(axis=-1) / (k-1)
        inv_dim_est = inv_dim.mean(axis=-1)
    dim_est = 1 / inv_dim_est
    if unbiased:
        dim_est *= (k-1)/(k-2) / (1-1/(k-3))
    return dim_est, inv_dim_est


def compute_mle_averaged_over_k(dists, kmin=None, kmax=None, fixnan=True, unbiased=True):
    if kmax is None:
        kmax = dists.shape[-1]
    if kmin is None:
        kmin = 5
    assert kmin >= 5, f'Beware kmin should be larger than 5 (to divide by (1-1/(k-3))) and here {kmin=}'

    inv_sum = sum(compute_mle(dists, k, fixnan, unbiased)[1] for k in range(kmin, kmax+1))
    return 1 / (inv_sum / (kmax+1-kmin))