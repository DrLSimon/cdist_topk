
import torch
from intrinsic_dim.estimators.mle import compute_mle
from intrinsic_dim.neighbors.patch_knn import patch_topk_dists

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
