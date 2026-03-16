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
    on the manifold: ε ~ sigma_1 / n^(1/d) * 1e-2
    """
    nn_dist_scale = sigma_1 / (n_samples ** (1.0 / d))
    return nn_dist_scale * 1e-2 * torch.randn(n_samples, full_dim)


def _spectral_sigma(d: int) -> torch.Tensor:
    """Decaying standard deviations: sigma_k = 1/sqrt(k)."""
    return 1.0 / torch.sqrt(torch.arange(1, d + 1, dtype=torch.float32))


def _embed(coords: torch.Tensor, full_dim: int, d: int, noise_sigma: float = 1.0) -> torch.Tensor:
    """Project (n_samples, embed_dim) coords into R^full_dim via a random orthonormal basis."""
    n_samples, embed_dim = coords.shape
    basis = _random_orthonormal_basis(full_dim, embed_dim)
    noise = _nn_noise(n_samples, full_dim, d, noise_sigma)
    return coords @ basis.T + noise


# ── registry ──────────────────────────────────────────────────────────────────

DISTRIBUTIONS = {}

def register(name):
    def decorator(fn):
        DISTRIBUTIONS[name] = fn
        return fn
    return decorator


# ── linear manifold ───────────────────────────────────────────────────────────

@register("linear")
def sample_linear(d: int, full_dim: int, n_samples: int) -> torch.Tensor:
    """
    d-dim linear submanifold via random orthonormal basis and spectrally
    decaying Gaussian latent coordinates. Intrinsic dim = d.
    """
    basis = _random_orthonormal_basis(full_dim, d)
    sigma = _spectral_sigma(d)
    z     = torch.randn(n_samples, d) * sigma
    noise = _nn_noise(n_samples, full_dim, d, sigma[0].item())
    return z @ basis.T + noise


# ── sphere ────────────────────────────────────────────────────────────────────

@register("sphere")
def sample_sphere(d: int, full_dim: int, n_samples: int) -> torch.Tensor:
    """
    Uniform samples on S^d (the d-sphere), which is a d-dim manifold embedded
    in R^(d+1), then projected into R^full_dim. Intrinsic dim = d.
    If d+1 > full_dim, clamps to S^(full_dim-1).
    """
    embed_dim = min(d + 1, full_dim)
    z = torch.randn(n_samples, embed_dim)
    z = z / z.norm(dim=-1, keepdim=True)
    return _embed(z, full_dim, d)


# ── torus (product of circles) ────────────────────────────────────────────────

@register("torus")
def sample_torus(d: int, full_dim: int, n_samples: int) -> torch.Tensor:
    """
    d-dim manifold as a product of d circles: (S^1)^d embedded in R^(2d)
    then projected into R^full_dim. Each S^1 factor contributes 2 coords
    (cos θ_k, sin θ_k). If 2*d > full_dim the embedding is clamped to full_dim
    columns, which still captures the d-dim structure.
    """
    thetas = 2 * math.pi * torch.rand(n_samples, d)
    coords = torch.stack([thetas.cos(), thetas.sin()], dim=-1).reshape(n_samples, 2 * d)
    embed_dim = min(2 * d, full_dim)
    return _embed(coords[:, :embed_dim], full_dim, d)


def list_distributions() -> list:
    """Return the names of all registered distributions."""
    return list(DISTRIBUTIONS.keys())


# ── patch grid sampler ────────────────────────────────────────────────────────

def sample_patches(dimensions: torch.Tensor, patch_size: int, nb_channels: int,
                   n_samples: int = 1, distrib: str = "linear") -> torch.Tensor:
    """
    Sample n_samples realisations of the patch grid using the given distribution.
    Returns (Ph, Pw, N, C*p*p) — same convention as images_to_patches.
    """
    nb_h, nb_w = dimensions.shape
    full_dim = nb_channels * patch_size * patch_size

    assert dimensions.min().item() >= 1, \
        f"All dimensions must be >= 1, got min={dimensions.min().item()}"
    assert dimensions.max().item() <= full_dim, \
        f"Max intrinsic dim {dimensions.max().item()} exceeds full_dim={full_dim} " \
        f"(patch_size={patch_size}, nb_channels={nb_channels})"
    assert distrib in DISTRIBUTIONS, \
        f"Unknown distribution '{distrib}'. Available: {list(DISTRIBUTIONS.keys())}"

    sample_fn = DISTRIBUTIONS[distrib]
    out = torch.zeros(nb_h, nb_w, n_samples, full_dim)

    for i in range(nb_h):
        for j in range(nb_w):
            d = max(1, min(int(dimensions[i, j].item()), full_dim))
            out[i, j] = sample_fn(d, full_dim, n_samples)

    return out  # (Ph, Pw, N, C*p*p)
# ── swiss roll ────────────────────────────────────────────────────────────────

@register("swiss_roll")
def sample_swiss_roll(d: int, full_dim: int, n_samples: int) -> torch.Tensor:
    """
    d-dim manifold as a cartesian product of a 2D swiss roll and a (d-2)-dim
    linear subspace. Intrinsic dim = d. Coords clamped to full_dim if needed.

    For d=1: 1D spiral arc.
    For d=2: classic swiss roll.
    For d>2: swiss roll × R^(d-2).
    """
    t    = (1.5 + 3.0 * torch.rand(n_samples)) * math.pi
    roll = torch.stack([t * t.cos(), t * t.sin()], dim=-1)   # (n_samples, 2)

    if d == 1:
        coords = t.unsqueeze(-1)                              # (n_samples, 1)
    elif d == 2:
        coords = roll                                         # (n_samples, 2)
    else:
        extra  = torch.randn(n_samples, d - 2)               # (n_samples, d-2)
        coords = torch.cat([roll, extra], dim=-1)             # (n_samples, d)

    embed_dim = min(coords.shape[-1], full_dim)
    return _embed(coords[:, :embed_dim], full_dim, d)


# ── polynomial curve / surface ────────────────────────────────────────────────

@register("poly")
def sample_poly(d: int, full_dim: int, n_samples: int, degree: int = 3) -> torch.Tensor:
    """
    d-dim nonlinear manifold via a degree-`degree` Veronese-style embedding:
    each of the d input dims is lifted to (t, t^2, ..., t^degree), giving
    embed_dim = d*degree (clamped to full_dim if needed). Intrinsic dim = d.
    """
    t      = 2 * torch.rand(n_samples, d) - 1
    powers = torch.cat([t ** k for k in range(1, degree + 1)], dim=-1)  # (n_samples, d*degree)
    embed_dim = min(d * degree, full_dim)
    return _embed(powers[:, :embed_dim], full_dim, d)


# ── product of spheres ────────────────────────────────────────────────────────

@register("product_spheres")
def sample_product_spheres(d: int, full_dim: int, n_samples: int) -> torch.Tensor:
    """
    d-dim manifold as (S^1)^d embedded in R^(2d) then projected to R^full_dim.
    If 2*d > full_dim the embedding is clamped to full_dim columns.
    """
    thetas = 2 * math.pi * torch.rand(n_samples, d)
    coords = torch.stack([thetas.cos(), thetas.sin()], dim=-1).reshape(n_samples, 2 * d)
    embed_dim = min(2 * d, full_dim)
    return _embed(coords[:, :embed_dim], full_dim, d)
