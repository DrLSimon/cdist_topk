import torch

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
