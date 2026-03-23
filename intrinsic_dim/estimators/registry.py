"""
registry.py — all estimator and variance estimator classes and their registered names.

To add a new dim estimator:
  1. Create `dim_estimators/my_estimator.py` with pure computation functions.
  2. Define the wrapper class here, decorate with @register("my_key").

To add a new variance estimator:
  1. Add pure computation functions to the appropriate file.
  2. Define the wrapper class here, decorate with @register_variance("my_key").
     The wrapper must accept a `estimator` kwarg (the dim estimator instance).
"""

import torch

from .core import register, register_variance, get_variance_estimator
from .pca_estimator import pca_effective_dim
from .mle_estimator import compute_mle, compute_mle_averaged_over_k
from intrinsic_dim.neighbors.patch_knn import patch_topk_dists


# ---------------------------------------------------------------------------
# Variance estimators
# ---------------------------------------------------------------------------

@register_variance("none")
class NullVarianceEstimator:
    """Default no-op — always returns zeros."""
    def __init__(self, estimator, **kwargs):
        pass

    def __call__(self, samples: torch.Tensor) -> torch.Tensor:
        nb_h, nb_w = samples.shape[:2]
        return torch.zeros((nb_h, nb_w), device=samples.device)


@register_variance("bootstrap")
class BootstrapVarianceEstimator:
    """
    Black-box bootstrap variance: re-runs the dim estimator on random
    subsamples and measures spread across trials.

    Parameters
    ----------
    estimator   : DimEstimator instance — injected automatically by the dim estimator
    n_subsample : size of each random reference subsample
    n_trials    : number of bootstrap trials (default 50)
    """
    def __init__(self, estimator, n_subsample: int, n_trials: int = 50):
        self.estimator   = estimator
        self.n_subsample = n_subsample
        self.n_trials    = n_trials

    def __call__(self, samples: torch.Tensor) -> torch.Tensor:
        """samples: (Ph, Pw, N, D)  →  variance: (Ph, Pw)"""
        n_samples = samples.shape[2]
        assert self.n_subsample <= n_samples, (
            f"n_subsample ({self.n_subsample}) must be <= n_samples ({n_samples})"
        )
        trials = []
        for _ in range(self.n_trials):
            ref_idx = torch.randperm(n_samples)[:self.n_subsample]
            trials.append(self.estimator(samples[:, :, ref_idx, :]))
        return torch.stack(trials, dim=0).var(dim=0)


# ---------------------------------------------------------------------------
# PCA
# ---------------------------------------------------------------------------

@register("pca")
class PCADimEstimator:
    """
    Effective dimensionality via cumulative explained variance (PCA).

    Parameters
    ----------
    threshold       : cumulative variance fraction to retain (default 0.999)
    variance        : variance estimator name (default "none")
    variance_kwargs : extra kwargs forwarded to the variance estimator
    """

    def __init__(self, threshold: float = 0.999,
                 variance: str = "none", variance_kwargs: dict = {}):
        self.threshold    = threshold
        self.variance_est = get_variance_estimator(variance, estimator=self, **variance_kwargs)

    def __call__(self, samples: torch.Tensor) -> torch.Tensor:
        """samples: (Ph, Pw, N, D)  →  dims: (Ph, Pw) int32"""
        nb_h, nb_w, _, _ = samples.shape
        dims = torch.zeros((nb_h, nb_w), dtype=torch.int32, device=samples.device)
        for i in range(nb_h):
            for j in range(nb_w):
                dims[i, j] = pca_effective_dim(samples[i, j], self.threshold)
        return dims

    def variance_of(self, samples: torch.Tensor) -> torch.Tensor:
        """samples: (Ph, Pw, N, D)  →  variance: (Ph, Pw)"""
        return self.variance_est(samples)


# ---------------------------------------------------------------------------
# MLE (Levina-Bickel) — fixed k
# ---------------------------------------------------------------------------

@register("mle")
class MLEDimEstimator:
    """
    Levina-Bickel MLE at a fixed number of neighbours k.

    Parameters
    ----------
    k               : number of nearest neighbours (default 10)
    n_anchors       : random query points per patch position (default 1000)
    fixnan          : mask NaN/Inf values in the log-ratio (default True)
    unbiased        : apply bias correction factor (default True)
    variance        : variance estimator name (default "none")
    variance_kwargs : extra kwargs forwarded to the variance estimator
    """

    def __init__(self, k: int = 10, n_anchors: int = 1000,
                 fixnan: bool = True, unbiased: bool = True,
                 variance: str = "none", variance_kwargs: dict = {}):
        self.k            = k
        self.n_anchors    = n_anchors
        self.fixnan       = fixnan
        self.unbiased     = unbiased
        self.variance_est = get_variance_estimator(variance, estimator=self, **variance_kwargs)

    def _get_dists(self, samples: torch.Tensor) -> torch.Tensor:
        n_samples = samples.shape[2]
        n_anchors = min(self.n_anchors, n_samples)
        idx       = torch.randperm(n_samples)[:n_anchors]
        anchors   = samples[:, :, idx, :]
        return patch_topk_dists(anchors, samples, k=self.k + 1, remove_self=True)

    def __call__(self, samples: torch.Tensor) -> torch.Tensor:
        """samples: (Ph, Pw, N, D)  →  dims: (Ph, Pw)"""
        dims, _ = compute_mle(self._get_dists(samples), k=self.k,
                               fixnan=self.fixnan, unbiased=self.unbiased)
        return dims

    def variance_of(self, samples: torch.Tensor) -> torch.Tensor:
        """samples: (Ph, Pw, N, D)  →  variance: (Ph, Pw)"""
        return self.variance_est(samples)


# ---------------------------------------------------------------------------
# MLE (Levina-Bickel) — averaged over k
# ---------------------------------------------------------------------------

@register("mle_avg")
class MLEAvgDimEstimator(MLEDimEstimator):
    """
    Levina-Bickel MLE averaged over k in [kmin, k].

    Parameters
    ----------
    k               : upper bound for k averaging (default 10)
    kmin            : lower bound for k averaging (default 5)
    n_anchors       : random query points per patch position (default 1000)
    fixnan          : mask NaN/Inf values in the log-ratio (default True)
    unbiased        : apply bias correction factor (default True)
    variance        : variance estimator name (default "none")
    variance_kwargs : extra kwargs forwarded to the variance estimator
    """

    def __init__(self, k: int = 10, kmin: int = 5, n_anchors: int = 1000,
                 fixnan: bool = True, unbiased: bool = True,
                 variance: str = "none", variance_kwargs: dict = {}):
        super().__init__(k=k, n_anchors=n_anchors, fixnan=fixnan,
                         unbiased=unbiased, variance=variance,
                         variance_kwargs=variance_kwargs)
        self.kmin = kmin

    def __call__(self, samples: torch.Tensor) -> torch.Tensor:
        """samples: (Ph, Pw, N, D)  →  dims: (Ph, Pw)"""
        return compute_mle_averaged_over_k(
            self._get_dists(samples), kmin=self.kmin, kmax=self.k,
            fixnan=self.fixnan, unbiased=self.unbiased,
        )