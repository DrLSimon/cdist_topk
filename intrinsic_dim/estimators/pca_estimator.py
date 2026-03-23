"""PCA-based intrinsic dimension — pure computation functions."""

import torch


def pca_effective_dim(samples: torch.Tensor, threshold: float = 0.999) -> int:
    """
    samples: (N, D)
    Returns effective dimensionality as an int.
    """
    S = samples - samples.mean(dim=0)
    _, sv, _ = torch.linalg.svd(S, full_matrices=False)
    var_ratio = (sv ** 2) / (sv ** 2).sum()
    return int(torch.searchsorted(var_ratio.cumsum(dim=0), threshold).item()) + 1