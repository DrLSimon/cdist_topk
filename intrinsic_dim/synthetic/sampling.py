
import torch
from intrinsic_dim.synthetic.manifolds import MANIFOLDS, DENSITIES, _nn_noise, _random_orthonormal_basis

# ── combined sampler ──────────────────────────────────────────────────────────
def _sample_one(d: int, full_dim: int, n_samples: int,
                manifold: str, density: str) -> torch.Tensor:
    """Sample n_samples points from the given manifold/density combination."""
    m_entry     = MANIFOLDS[manifold]
    manifold_fn = m_entry["fn"]
    density_fn  = DENSITIES[density or m_entry["default_density"]]
    max_d       = m_entry["max_dim"](full_dim)

    assert d <= max_d, (
        f"Manifold '{manifold}' supports d <= {max_d} for full_dim={full_dim}, got d={d}"
    )

    latent_dims = {
        "linear":      d,
        "sphere":      min(d + 1, full_dim),
        "torus":       d,
        "swiss_roll":  max(d, 1),
        "poly":        d,
    }
    latent_dim = latent_dims[manifold]

    z              = density_fn(n_samples, latent_dim)
    
    coords, sigma1 = manifold_fn(z, d, full_dim)
    noise          = _nn_noise(n_samples, full_dim, d, sigma1)
    basis          = _random_orthonormal_basis(full_dim, coords.shape[-1])
    return coords @ basis.T #+ noise


# ── public API ────────────────────────────────────────────────────────────────

def get_max_dim(manifold: str, full_dim: int) -> int:
    """Return the maximum supported intrinsic dim for a manifold given full_dim."""
    return MANIFOLDS[manifold]["max_dim"](full_dim)


def list_manifolds() -> list:
    """Return the names of all registered manifolds."""
    return list(MANIFOLDS.keys())


def list_densities() -> list:
    """Return the names of all registered densities."""
    return list(DENSITIES.keys())


def list_distributions() -> list:
    """Return all valid (manifold, density) combinations as 'manifold:density' strings,
    plus bare manifold names which use the default density."""
    return list_manifolds()


def sample_patches(dimensions: torch.Tensor, patch_size: int, nb_channels: int,
                   n_samples: int = 1, manifold: str = "linear",
                   density: str = None) -> torch.Tensor:
    """
    Sample n_samples realisations of the patch grid.
    Returns (Ph, Pw, N, C*p*p) — same convention as images_to_patches.

    Args:
        manifold: one of list_manifolds()
        density:  one of list_densities(), or None to use the manifold's default
    """
    nb_h, nb_w = dimensions.shape
    full_dim = nb_channels * patch_size * patch_size

    assert dimensions.min().item() >= 1, \
        f"All dimensions must be >= 1, got min={dimensions.min().item()}"
    max_d = get_max_dim(manifold, full_dim)
    assert dimensions.max().item() <= max_d, \
        f"Manifold '{manifold}' supports d <= {max_d} for full_dim={full_dim}, " \
        f"got dimensions.max()={dimensions.max().item()}"
    assert manifold in MANIFOLDS, \
        f"Unknown manifold '{manifold}'. Available: {list_manifolds()}"
    assert density is None or density in DENSITIES, \
        f"Unknown density '{density}'. Available: {list_densities()}"

    out = torch.zeros(nb_h, nb_w, n_samples, full_dim)
    for i in range(nb_h):
        for j in range(nb_w):
            d = max(1, min(int(dimensions[i, j].item()), full_dim))
            out[i, j] = _sample_one(d, full_dim, n_samples, manifold, density)

    return out  # (Ph, Pw, N, C*p*p)
