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
    nn_dist_scale = sigma_1 / math.sqrt(full_dim)
    return nn_dist_scale * torch.randn(n_samples, full_dim)


def _spectral_sigma(d: int) -> torch.Tensor:
    """Decaying standard deviations: sigma_k = 1/sqrt(k)."""
    return 1.0 / torch.sqrt(torch.arange(1, d + 1, dtype=torch.float32))


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
# Each manifold function returns (coords, sigma_1, max_dim) where:
#   coords:  (n_samples, embed_dim) — ambient coordinates before final projection
#   sigma_1: characteristic scale for off-manifold noise
#   max_dim: callable (full_dim) -> int, max supported intrinsic dim
#
# The decorator splits this into two entries in MANIFOLDS:
#   "fn":             (z, d, full_dim) -> (coords, sigma_1)
#   "max_dim":        (full_dim) -> int
#   "default_density": str

MANIFOLDS = {}

def register_manifold(name: str, default_density: str):
    """
    Register a manifold. The decorated function must return
    (coords, sigma_1, max_dim_fn) where max_dim_fn is a callable (full_dim) -> int.
    The decorator exposes fn(z, d, full_dim) -> (coords, sigma_1)
    and max_dim(full_dim) -> int as separate entries in MANIFOLDS.
    """
    def decorator(fn):
        # extract max_dim_fn by peeking at a minimal call
        _, _, max_dim_fn = fn(torch.zeros(1, 1), 1, 1)

        def _fn(z, d, full_dim):
            coords, sigma_1, _ = fn(z, d, full_dim)
            return coords, sigma_1

        MANIFOLDS[name] = {"fn": _fn, "max_dim": max_dim_fn,
                           "default_density": default_density}
        return fn
    return decorator


@register_manifold("linear", default_density="gaussian_spectral")
def manifold_linear(z: torch.Tensor, d: int, full_dim: int):
    """
    Linear submanifold: z is used directly as latent coords.
    latent_dim = d, embed_dim = d.
    """
    sigma1 = 1e-3
    return z, sigma1, lambda fd: fd


@register_manifold("sphere", default_density="gaussian")
def manifold_sphere(z: torch.Tensor, d: int, full_dim: int):
    """
    S^d: normalise z ~ R^(d+1) to unit sphere.
    latent_dim = d+1, embed_dim = min(d+1, full_dim).
    """
    z_norm    = z / z.norm(dim=-1, keepdim=True)
    embed_dim = min(z.shape[-1], full_dim)
    return z_norm[:, :embed_dim], 1e-3, lambda fd: fd - 1


@register_manifold("torus", default_density="uniform")
def manifold_torus(z: torch.Tensor, d: int, full_dim: int):
    """
    (S^1)^d: interpret z in [0,1]^d as angles theta = 2*pi*z.
    latent_dim = d, embed_dim = 2*d.
    Max supported d = full_dim // 2.
    """
    thetas = 2 * math.pi * z
    coords = torch.stack([thetas.cos(), thetas.sin()], dim=-1).reshape(z.shape[0], 2 * d)
    return coords, 1e-3, lambda fd: fd // 2


@register_manifold("swiss_roll", default_density="uniform")
def manifold_swiss_roll(z: torch.Tensor, d: int, full_dim: int):
    """
    Product of d independent 1D swiss roll curves, each mapping t -> (t·cos t, t·sin t)
    with t = (1.5 + 3*z)*pi. Each curve is intrinsically 1D embedded in R^2,
    so embed_dim = 2*d and max supported d = full_dim // 2.
    """
    pieces = []
    for k in range(d):
        t = (1.5 + 3.0 * z[:, k]) * math.pi               # (n_samples,)
        pieces.append(torch.stack([t * t.cos(), t * t.sin()], dim=-1))  # (n, 2)

    coords    = torch.cat(pieces, dim=-1)                  # (n_samples, 2*d)
    embed_dim = min(2 * d, full_dim)
    return coords[:, :embed_dim], 1e-3, lambda fd: fd // 2


@register_manifold("poly", default_density="uniform")
def manifold_poly(z: torch.Tensor, d: int, full_dim: int, degree: int = 3):
    """
    Veronese polynomial embedding: each dim of z is lifted to (z, z^2, ..., z^degree).
    latent_dim = d, embed_dim = min(d*degree, full_dim).
    """
    t      = 2 * z - 1
    powers = torch.cat([t ** k for k in range(1, degree + 1)], dim=-1)
    embed_dim = min(d * degree, full_dim)
    return powers[:, :embed_dim], 1e-2, lambda fd: fd // degree
