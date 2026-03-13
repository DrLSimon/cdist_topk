import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from utils import images_to_patches, patches_to_images


# ── Core ─────────────────────────────────────────────────────────────────────

def _make_patch_bases(dimensions, full_dim):
    nb_h, nb_w = dimensions.shape
    bases = {}
    for i in range(nb_h):
        for j in range(nb_w):
            d = max(1, min(int(dimensions[i, j].item()), full_dim))
            Q, _ = torch.linalg.qr(torch.randn(full_dim, d))
            bases[(i, j)] = Q[:, :d]
    return bases


def sample_patches(dimensions, patch_size, nb_channels, n_samples=1):
    """
    Sample n_samples realisations of the patch grid.
    Returns (Ph, Pw, N, C*p*p) — same convention as images_to_patches.
    """
    nb_h, nb_w = dimensions.shape
    full_dim = nb_channels * patch_size * patch_size

    assert dimensions.min().item() >= 1, \
        f"All dimensions must be >= 1, got min={dimensions.min().item()}"
    assert dimensions.max().item() <= full_dim, \
        f"Max intrinsic dim {dimensions.max().item()} exceeds full_dim={full_dim} " \
        f"(patch_size={patch_size}, nb_channels={nb_channels})"

    bases = _make_patch_bases(dimensions, full_dim)
    out   = torch.zeros(nb_h, nb_w, n_samples, full_dim)

    for i in range(nb_h):
        for j in range(nb_w):
            d     = max(1, min(int(dimensions[i, j].item()), full_dim))
            basis = bases[(i, j)]                                   # (full_dim, d)
            sigma = 1.0 / torch.sqrt(torch.arange(1, d + 1, dtype=torch.float32))
            z     = torch.randn(n_samples, d) * sigma               # (n_samples, d)
            noise = 1e-3 * torch.randn(n_samples, full_dim)
            out[i, j, :, :] = z @ basis.T + noise

    return out  # (nb_h, nb_w, n_samples, full_dim)

# ── PCA utils ────────────────────────────────────────────────────────────────

def pca_effective_dim(samples: np.ndarray, threshold: float = 0.95):
    """
    samples: (n_samples, full_dim)
    Returns (effective_dim, singular_values).
    """
    S = samples - samples.mean(axis=0)
    _, sv, _ = np.linalg.svd(S, full_matrices=False)
    var_ratio = (sv ** 2) / (sv ** 2).sum()
    return int(np.searchsorted(np.cumsum(var_ratio), threshold)) + 1, sv


def compute_pca_dims(samples: torch.Tensor, threshold: float = 0.95):
    """
    Compute PCA effective dimensionality for every patch position.

    Args:
        samples:   (Ph, Pw, N, C*p*p) — same convention as images_to_patches
        threshold: cumulative variance threshold (default 0.95)

    Returns:
        pca_dims: (nb_h, nb_w) np.ndarray of effective dims
        spectra:  dict {(i, j): singular_values}
    """
    nb_h, nb_w, _, _ = samples.shape
    pca_dims = np.zeros((nb_h, nb_w))
    spectra  = {}
    for i in range(nb_h):
        for j in range(nb_w):
            pca_dims[i, j], spectra[(i, j)] = pca_effective_dim(
                samples[i, j, :, :].numpy(), threshold
            )
    return pca_dims, spectra


# ── Plot ─────────────────────────────────────────────────────────────────────

def plot_submanifold_test(patch_size=8, nb_channels=1, n_samples=512):
    # max dim = 64 <= full_dim = 8*8*1 = 64 ✓
    dims = torch.tensor([
        [1,  2,  4,  8],
        [2,  4,  8, 16],
        [4,  8, 16, 32],
        [1,  8, 32, 64],
    ])

    full_dim = nb_channels * patch_size * patch_size
    nb_h, nb_w = dims.shape

    samples = sample_patches(dims, patch_size, nb_channels, n_samples=n_samples)
    threshold = 0.9999  # use a very high threshold to capture all variance
    pca_dims, spectra = compute_pca_dims(samples, threshold=threshold)

    fig = plt.figure(figsize=(16, 10))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    _plot_heatmap(fig.add_subplot(gs[0, 0]), dims.numpy(),  "Target intrinsic dim",        dims)
    _plot_heatmap(fig.add_subplot(gs[0, 1]), pca_dims,      f"PCA-estimated dim ({threshold*100}% var)", dims)
    _plot_scatter(fig.add_subplot(gs[0, 2]), dims.numpy().flatten(), pca_dims.flatten())
    _plot_spectra(fig.add_subplot(gs[1, 0]), spectra, dims, nb_h, nb_w)
    _plot_patch_grid(fig.add_subplot(gs[1, 1]), samples[:, :, 0, :], patch_size, nb_channels, nb_h, nb_w)
    _plot_residuals(fig.add_subplot(gs[1, 2]), dims.numpy().flatten(), pca_dims.flatten())

    fig.suptitle(
        f"Submanifold patch test  |  patch_size={patch_size}  "
        f"nb_channels={nb_channels}  full_dim={full_dim}  n_samples={n_samples}",
        fontsize=12,
    )
    plt.tight_layout()
    plt.savefig("submanifold_test.png", dpi=150, bbox_inches="tight")
    plt.show()


# ── Axes helpers ─────────────────────────────────────────────────────────────

def _plot_heatmap(ax, data, title, dims_ref):
    nb_h, nb_w = data.shape
    im = ax.imshow(data, cmap="viridis", aspect="auto",
                   vmin=dims_ref.min().item(), vmax=dims_ref.max().item())
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("patch col"); ax.set_ylabel("patch row")
    plt.colorbar(im, ax=ax, fraction=0.046)
    for i in range(nb_h):
        for j in range(nb_w):
            ax.text(j, i, str(int(data[i, j])),
                    ha="center", va="center", color="white", fontsize=9)


def _plot_scatter(ax, target, estimated):
    ax.scatter(target, estimated, alpha=0.8, edgecolors="k", linewidths=0.5)
    lim = [0, max(target.max(), estimated.max()) + 2]
    ax.plot(lim, lim, "r--", linewidth=1, label="y = x")
    ax.set_xlabel("Target dim"); ax.set_ylabel("PCA-estimated dim")
    ax.set_title("Target vs estimated", fontsize=11)
    ax.legend(fontsize=9)


def _plot_spectra(ax, spectra, dims, nb_h, nb_w):
    cmap = plt.cm.viridis
    vmin, vmax = dims.min().item(), dims.max().item()
    for i in range(nb_h):
        for j in range(nb_w):
            sv = spectra[(i, j)]
            var_ratio = (sv ** 2) / (sv ** 2).sum()
            color = cmap((dims[i, j].item() - vmin) / (vmax - vmin + 1e-8))
            ax.plot(np.arange(1, len(var_ratio) + 1), var_ratio,
                    color=color, alpha=0.6, linewidth=1)
    ax.set_xlabel("Principal component"); ax.set_ylabel("Var explained ratio")
    ax.set_title("Singular value spectra (all patches)", fontsize=11)
    plt.colorbar(plt.cm.ScalarMappable(cmap=cmap,
                 norm=plt.Normalize(vmin=vmin, vmax=vmax)),
                 ax=ax, fraction=0.046, label="target dim")


def _plot_patch_grid(ax, single_sample, patch_size, nb_channels, nb_h, nb_w):
    # single_sample: (nb_h, nb_w, full_dim) — one sample across all patch positions
    grid = np.zeros((nb_h * patch_size, nb_w * patch_size))
    for i in range(nb_h):
        for j in range(nb_w):
            tile = single_sample[i, j].numpy().reshape(nb_channels, patch_size, patch_size)[0]
            tile = (tile - tile.min()) / (tile.max() - tile.min() + 1e-8)
            grid[i*patch_size:(i+1)*patch_size,
                 j*patch_size:(j+1)*patch_size] = tile
    ax.imshow(grid, cmap="gray", aspect="auto")
    ax.set_title("Patch visualisation (ch 0, sample 0)", fontsize=11)
    ax.set_xticks([(j + 0.5) * patch_size for j in range(nb_w)])
    ax.set_yticks([(i + 0.5) * patch_size for i in range(nb_h)])
    ax.set_xticklabels(range(nb_w)); ax.set_yticklabels(range(nb_h))


def _plot_residuals(ax, target, estimated):
    residuals = estimated - target
    ax.bar(range(len(residuals)), residuals, color="slategray", edgecolor="k", linewidth=0.5)
    ax.axhline(0, color="red", linewidth=1, linestyle="--")
    ax.set_xlabel("Patch index (row-major)"); ax.set_ylabel("PCA dim − target dim")
    ax.set_title("Residuals", fontsize=11)

if __name__ == "__main__":
    plot_submanifold_test(patch_size=8, nb_channels=1, n_samples=512)