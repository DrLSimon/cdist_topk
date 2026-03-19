"""
registry.py — all estimator classes and their registered names.

To add a new estimator:
  1. Create `dim_estimators/my_estimator.py` with pure computation functions.
  2. Define the wrapper class here and call register().
"""

import torch

from .core import register
from .pca_estimator import pca_effective_dim
from .mle_estimator import compute_mle, compute_mle_averaged_over_k
from intrinsic_dim.neighbors.patch_knn import patch_topk_dists


# ---------------------------------------------------------------------------
# PCA
# ---------------------------------------------------------------------------

@register("pca")
class PCADimEstimator:
    """
    Effective dimensionality via cumulative explained variance (PCA).

    Parameters
    ----------
    threshold : float
        Cumulative variance fraction to retain (default 0.999).
    """

    def __init__(self, threshold: float = 0.999):
        self.threshold = threshold

    def __call__(self, samples: torch.Tensor) -> torch.Tensor:
        """samples: (Ph, Pw, N, D)  →  dims: (Ph, Pw) int32"""
        nb_h, nb_w, _, _ = samples.shape
        dims = torch.zeros((nb_h, nb_w), dtype=torch.int32, device=samples.device)
        for i in range(nb_h):
            for j in range(nb_w):
                dims[i, j] = pca_effective_dim(samples[i, j], self.threshold)
        return dims


# ---------------------------------------------------------------------------
# MLE (Levina-Bickel) — fixed k
# ---------------------------------------------------------------------------

@register("mle")
class MLEDimEstimator:
    """
    Levina-Bickel MLE at a fixed number of neighbours k.

    Parameters
    ----------
    k         : number of nearest neighbours (default 10)
    n_anchors : random query points per patch position (default 1000)
    fixnan    : mask NaN/Inf values in the log-ratio (default True)
    """

    def __init__(self, k: int = 10, n_anchors: int = 1000, fixnan: bool = True):
        self.k         = k
        self.n_anchors = n_anchors
        self.fixnan    = fixnan

    def _get_dists(self, samples: torch.Tensor) -> torch.Tensor:
        n_samples   = samples.shape[2]
        n_anchors   = min(self.n_anchors, n_samples)
        idx         = torch.randperm(n_samples)[:n_anchors]
        anchors     = samples[:, :, idx, :]
        return patch_topk_dists(anchors, samples, k=self.k + 1, remove_self=True)

    def __call__(self, samples: torch.Tensor) -> torch.Tensor:
        """samples: (Ph, Pw, N, D)  →  dims: (Ph, Pw)"""
        dists, _ = compute_mle(self._get_dists(samples), k=self.k, fixnan=self.fixnan)
        return dists


# ---------------------------------------------------------------------------
# MLE (Levina-Bickel) — averaged over k
# ---------------------------------------------------------------------------

@register("mle_avg")
class MLEAvgDimEstimator(MLEDimEstimator):
    """
    Levina-Bickel MLE averaged over k in [kmin, k].

    Parameters
    ----------
    k         : upper bound for k averaging (default 10)
    kmin      : lower bound for k averaging (default 5)
    n_anchors : random query points per patch position (default 1000)
    fixnan    : mask NaN/Inf values in the log-ratio (default True)
    """

    def __init__(self, k: int = 10, kmin: int = 5, n_anchors: int = 1000, fixnan: bool = True):
        super().__init__(k=k, n_anchors=n_anchors, fixnan=fixnan)
        self.kmin = kmin

    def __call__(self, samples: torch.Tensor) -> torch.Tensor:
        """samples: (Ph, Pw, N, D)  →  dims: (Ph, Pw)"""
        return compute_mle_averaged_over_k(
            self._get_dists(samples), kmin=self.kmin, kmax=self.k, fixnan=self.fixnan
        )