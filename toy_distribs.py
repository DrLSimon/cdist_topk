import torch
import math


# ── helpers ───────────────────────────────────────────────────────────────────

def _random_orthonormal_basis(full_dim: int, d: int) -> torch.Tensor:
    """Returns (full_dim, d) orthonormal matrix sampled from the Haar measure."""
    Q, _ = torch.linalg.qr(torch.randn(full_dim, d))
    return Q[:, :d]


def _nn_noise(n_samples: int, full_dim: int, d: int, sigma_1: float) -> torch.Tensor:
    """
    Isotropic noise scaled well below the expected nearest-neighbour distance
    on the manifold: eps ~ sigma_1 / n^(1/d) * 1e-2
    """
    nn_dist_scale = sigma_1 / (n_samples ** (1.0 / d))
    return nn_dist_scale * 1e-2 * torch.randn(n_samples, full_dim)


def _spectral_sigma(d: int) -> torch.Tensor:
    """Decaying standard deviations: sigma_k = 1/sqrt(k)."""
    return 1.0 / torch.sqrt(torch.arange(1, d + 1, dtype=torch.float32))


def _embed(coords: torch.Tensor, full_dim: int, d: int, sigma_1: float = 1.0) -> torch.Tensor:
    """Project (n_samples, embed_dim) coords into R^full_dim via a random orthonormal basis."""
    n_samples, embed_dim = coords.shape
    basis = _random_orthonormal_basis(full_dim, embed_dim)
    noise = _nn_noise(n_samples, full_dim, d, sigma_1)
    return coords @ basis.T + noise


# ── density registry ──────────────────────────────────────────────────────────
# A density samples latent coordinates z of shape (n_samples, latent_dim).
# latent_dim is manifold-dependent and passed by the manifold at call time.
#
# Signature: density_fn(n_samples: int, latent_dim: int) -> Tensor (n_samples, latent_dim)

DENSITIES = {}

def register_density(name: str):
    def decorator(fn):
        DENSITIES[name] = fn
        return fn
    return decorator


@register_density("gaussian")
def density_gaussian(n_samples: int, latent_dim: int) -> torch.Tensor:
    """Isotropic standard Gaussian in R^latent_dim."""
    return torch.randn(n_samples, latent_dim)


@register_density("gaussian_spectral")
def density_gaussian_spectral(n_samples: int, latent_dim: int) -> torch.Tensor:
    """Gaussian with spectrally decaying variance: sigma_k = 1/sqrt(k)."""
    sigma = _spectral_sigma(latent_dim)
    return torch.randn(n_samples, latent_dim) * sigma


@register_density("uniform")
def density_uniform(n_samples: int, latent_dim: int) -> torch.Tensor:
    """Uniform in [0, 1]^latent_dim."""
    return torch.rand(n_samples, latent_dim)


@register_density("laplace")
def density_laplace(n_samples: int, latent_dim: int) -> torch.Tensor:
    """Isotropic Laplace(0, 1) in R^latent_dim."""
    return torch.distributions.Laplace(0.0, 1.0).sample((n_samples, latent_dim))


@register_density("mixture_gaussian")
def density_mixture_gaussian(n_samples: int, latent_dim: int, n_components: int = 4) -> torch.Tensor:
    """
    Mixture of n_components isotropic Gaussians with random centres in [-2, 2]^latent_dim.
    Centres are redrawn each call, giving a different mixture per patch.
    """
    centres = 4 * torch.rand(n_components, latent_dim) - 2      # (K, d)
    idx     = torch.randint(n_components, (n_samples,))          # (N,)
    return centres[idx] + 0.3 * torch.randn(n_samples, latent_dim)


# ── manifold registry ─────────────────────────────────────────────────────────
# A manifold maps latent coords z (n_samples, latent_dim) -> ambient coords
# (n_samples, embed_dim), where embed_dim <= full_dim.
# It also returns sigma_1: a characteristic scale for noise.
#
# Signature: manifold_fn(z, d, full_dim) -> (coords, sigma_1)

MANIFOLDS = {}

def register_manifold(name: str, default_density: str):
    """
    Register a manifold with its recommended default density.
    The default_density is used when the user does not specify one.
    """
    def decorator(fn):
        MANIFOLDS[name] = {"fn": fn, "default_density": default_density}
        return fn
    return decorator


@register_manifold("linear", default_density="gaussian_spectral")
def manifold_linear(z: torch.Tensor, d: int, full_dim: int):
    """
    Linear submanifold: z is used directly as latent coords.
    latent_dim = d, embed_dim = d.
    """
    basis  = _random_orthonormal_basis(full_dim, d)
    coords = z                                                    # (n_samples, d)
    return coords @ basis.T, z.std().item() or 1.0


@register_manifold("sphere", default_density="gaussian")
def manifold_sphere(z: torch.Tensor, d: int, full_dim: int):
    """
    S^d: normalise z ~ R^(d+1) to unit sphere.
    latent_dim = d+1, embed_dim = min(d+1, full_dim).
    """
    z_norm = z / z.norm(dim=-1, keepdim=True)
    embed_dim = min(z.shape[-1], full_dim)
    return z_norm[:, :embed_dim], 1.0


@register_manifold("torus", default_density="uniform")
def manifold_torus(z: torch.Tensor, d: int, full_dim: int):
    """
    (S^1)^d: interpret z in [0,1]^d as angles theta = 2*pi*z.
    latent_dim = d, embed_dim = min(2*d, full_dim).
    """
    thetas = 2 * math.pi * z                                     # (n_samples, d)
    coords = torch.stack([thetas.cos(), thetas.sin()], dim=-1).reshape(z.shape[0], 2 * d)
    embed_dim = min(2 * d, full_dim)
    return coords[:, :embed_dim], 1.0


@register_manifold("swiss_roll", default_density="uniform")
def manifold_swiss_roll(z: torch.Tensor, d: int, full_dim: int):
    """
    Swiss roll x R^(d-2): z[:,0] -> t (rescaled to [1.5pi, 4.5pi]),
    z[:,1:] -> extra linear dims for d>2.
    latent_dim = d, embed_dim = min(d, full_dim).
    """
    t = (1.5 + 3.0 * z[:, 0]) * math.pi                         # (n_samples,)
    roll = torch.stack([t * t.cos(), t * t.sin()], dim=-1)       # (n_samples, 2)
    if d == 1:
        coords = t.unsqueeze(-1)
    elif d == 2:
        coords = roll
    else:
        coords = torch.cat([roll, z[:, 1:d - 1]], dim=-1)        # (n_samples, d)
    embed_dim = min(coords.shape[-1], full_dim)
    return coords[:, :embed_dim], 1.0


@register_manifold("poly", default_density="uniform")
def manifold_poly(z: torch.Tensor, d: int, full_dim: int, degree: int = 3):
    """
    Veronese polynomial embedding: each dim of z is lifted to (z, z^2, ..., z^degree).
    latent_dim = d, embed_dim = min(d*degree, full_dim).
    """
    t = 2 * z - 1                                                # rescale [0,1]^d -> [-1,1]^d
    powers = torch.cat([t ** k for k in range(1, degree + 1)], dim=-1)
    embed_dim = min(d * degree, full_dim)
    return powers[:, :embed_dim], 1.0


@register_manifold("product_spheres", default_density="gaussian")
def manifold_product_spheres(z: torch.Tensor, d: int, full_dim: int):
    """
    (S^1)^d via pair-wise normalisation: split z into d pairs, normalise each.
    latent_dim = 2*d, embed_dim = min(2*d, full_dim).
    """
    n_samples = z.shape[0]
    pairs  = z.reshape(n_samples, d, 2)
    pairs  = pairs / pairs.norm(dim=-1, keepdim=True)            # normalise each pair -> S^1
    coords = pairs.reshape(n_samples, 2 * d)
    embed_dim = min(2 * d, full_dim)
    return coords[:, :embed_dim], 1.0


# ── combined sampler ──────────────────────────────────────────────────────────

# Legacy flat registry for backwards compatibility with --distrib
DISTRIBUTIONS = {}

def register(name: str, density: str = None):
    """
    Register a combined (manifold, density) pair under a single name.
    If density is None, uses the manifold's default density.
    """
    def decorator(fn):
        DISTRIBUTIONS[name] = fn
        return fn
    return decorator


def _sample_one(d: int, full_dim: int, n_samples: int,
                manifold: str, density: str) -> torch.Tensor:
    """Sample n_samples points from the given manifold/density combination."""
    m_entry   = MANIFOLDS[manifold]
    manifold_fn = m_entry["fn"]
    density_fn  = DENSITIES[density or m_entry["default_density"]]

    # Determine latent_dim from manifold
    latent_dims = {
        "linear":          d,
        "sphere":          min(d + 1, full_dim),
        "torus":           d,
        "swiss_roll":      max(d, 1),
        "poly":            d,
        "product_spheres": 2 * d,
    }
    latent_dim = latent_dims[manifold]

    z              = density_fn(n_samples, latent_dim)            # (n_samples, latent_dim)
    coords, sigma1 = manifold_fn(z, d, full_dim)                  # (n_samples, embed_dim)
    noise          = _nn_noise(n_samples, full_dim, d, sigma1)
    basis          = _random_orthonormal_basis(full_dim, coords.shape[-1])
    return coords @ basis.T + noise                               # (n_samples, full_dim)


# ── public API ────────────────────────────────────────────────────────────────

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
    assert dimensions.max().item() <= full_dim, \
        f"Max intrinsic dim {dimensions.max().item()} exceeds full_dim={full_dim} " \
        f"(patch_size={patch_size}, nb_channels={nb_channels})"
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
