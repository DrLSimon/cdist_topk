import torch
import numpy as np
import warnings
import argparse
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


from dimension import compute_mle_dims, compute_pca_dims, compute_mle_dims_variance, compute_mle_dims_sample_variance
from toy_distribs import sample_patches, list_manifolds, list_densities, get_max_dim


def make_dims_grid(full_dim: int, max_d: int, grid: str) -> torch.Tensor:
    """
    Build a 2D tensor of target intrinsic dimensions.

    grid="auto"   : 4x4 grid with a fixed spread from 1 to max_d (default behaviour)
    grid="linear" : 1xN grid sweeping 1..max_d in N equal steps (N=16 by default)
    grid="powers" : 1xN grid of powers of 2 up to max_d
    grid="1,3,7,…": comma-separated list of integers, arranged as a 1-row tensor
    """
    if grid == "auto":
        # Reproduce the original 4x4 layout, clamped to max_d
        raw = torch.tensor([
            [ 1,  2,  3,  4],
            [ 5,  6,  7,  8],
            [11, 14, 18, 23],
            [26, 29, 32, max_d],
        ])
        return raw.clamp(max=max_d)

    if grid == "linear":
        n = 16
        vals = torch.linspace(1, max_d, n).round().long().clamp(min=1, max=max_d)
        return vals.unsqueeze(0)  # (1, n)

    if grid == "powers":
        vals = []
        v = 1
        while v <= max_d:
            vals.append(v)
            v *= 2
        if vals[-1] != max_d:
            vals.append(max_d)
        return torch.tensor(vals).clamp(max=max_d).unsqueeze(0)  # (1, n)

    # fallback: treat grid as a comma-separated list
    try:
        vals = [int(x.strip()) for x in grid.split(",")]
        return torch.tensor(vals).clamp(min=1, max=max_d).unsqueeze(0)
    except ValueError:
        raise ValueError(
            f"Unknown grid spec '{grid}'. "
            "Use 'auto', 'linear', 'powers', or a comma-separated list of ints."
        )


def plot_submanifold_test(patch_size=8, nb_channels=1, n_samples=25000, k_mle=10,
                          n_anchors=1000, manifold="linear", density=None,
                          grid="powers", pca_threshold=0.999,
                          n_trials=10, n_subsample=1000,
                          variance_mode="aggregated"):
    """
    variance_mode: "aggregated"  — trial-to-trial variability of the aggregated estimate
                   "per_sample"  — average per-anchor variance across trials (captures
                                   noise due to reference subsampling per data point)
    """
    full_dim = nb_channels * patch_size * patch_size
    max_d    = get_max_dim(manifold, full_dim)

    dims     = make_dims_grid(full_dim, max_d, grid)
    nb_h, nb_w = dims.shape

    samples = sample_patches(dims, patch_size, nb_channels, n_samples=n_samples,
                              manifold=manifold, density=density)

    pca_dims                   = compute_pca_dims(samples, threshold=pca_threshold)
    pca_dims_np                = pca_dims.cpu().numpy()
    mle_dims, mle_avg_dims     = compute_mle_dims(samples, k=k_mle, n_anchors=n_anchors)
    mle_dims_var = (
        compute_mle_dims_variance(
            samples, k=k_mle, n_anchors=n_anchors,
            n_subsample=n_subsample, n_trials=n_trials)
        if variance_mode == "aggregated" else
        compute_mle_dims_sample_variance(
            samples, k=k_mle, n_anchors=n_anchors,
            n_subsample=n_subsample, n_trials=n_trials)
    )
    std_label = f"STD MLE dims ({variance_mode}, k={k_mle})"
    mle_dims_std_np            = mle_dims_var.sqrt().cpu().numpy()
    mle_dims_np                = mle_dims.cpu().numpy()
    mle_avg_dims_np            = mle_avg_dims.cpu().numpy()
    target                     = dims.cpu().numpy().flatten()

    fig = plt.figure(figsize=(18, 14))
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.5, wspace=0.35)

    distrib_label = f"{manifold}:{density or 'default'}"

    # row 0: target heatmap, STD heatmap, scatter
    _plot_heatmap(fig.add_subplot(gs[0, 0]), dims.cpu().numpy(), "Target intrinsic dim", dims)
    _plot_heatmap(fig.add_subplot(gs[0, 1]), mle_dims_std_np, std_label, dims)
    _plot_scatter_multi(fig.add_subplot(gs[0, 2]), target, {
        f"PCA ({pca_threshold:.0%})": pca_dims_np.flatten(),
        f"MLE k={k_mle}":             mle_dims_np.flatten(),
        f"MLE avg k=3..{k_mle}":      mle_avg_dims_np.flatten(),
    })

    # row 1: PCA, MLE, MLE avg heatmaps
    _plot_heatmap(fig.add_subplot(gs[1, 0]), pca_dims_np,     f"PCA-estimated dim ({pca_threshold:.0%} var)", dims)
    _plot_heatmap(fig.add_subplot(gs[1, 1]), mle_dims_np,     f"MLE dim (k={k_mle})",                        dims)
    _plot_heatmap(fig.add_subplot(gs[1, 2]), mle_avg_dims_np, f"MLE avg dim (k=3..{k_mle})",                 dims)

    # row 2: residuals
    _plot_residuals(fig.add_subplot(gs[2, 0]), target, pca_dims_np.flatten(),     "PCA residuals")
    _plot_residuals(fig.add_subplot(gs[2, 1]), target, mle_dims_np.flatten(),     f"MLE residuals (k={k_mle})")
    _plot_residuals(fig.add_subplot(gs[2, 2]), target, mle_avg_dims_np.flatten(), f"MLE avg residuals (k=3..{k_mle})")

    fig.suptitle(
        f"Submanifold patch test  |  distrib={distrib_label}  patch_size={patch_size}  "
        f"nb_channels={nb_channels}  full_dim={full_dim}  n_samples={n_samples}  "
        f"n_anchors={n_anchors}  pca_threshold={pca_threshold:.0%}  k_mle={k_mle}  "
        f"var={variance_mode}",
        fontsize=12,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*tight_layout.*")
        plt.tight_layout()

    density_label = density or "default"
    fname = f"{manifold}__{density_label}__k{k_mle}__fd{full_dim}__grid{grid}__var{variance_mode}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    print(f"Saved to {fname}")
    plt.show()


# ── Axes helpers ──────────────────────────────────────────────────────────────

def _plot_heatmap(ax, data, title, dims_ref):
    nb_h, nb_w = data.shape
    im = ax.imshow(data, cmap="viridis", aspect="auto",
                   vmin=dims_ref.min().item(), vmax=dims_ref.max().item())
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("patch col"); ax.set_ylabel("patch row")
    plt.colorbar(im, ax=ax, fraction=0.046)
    for i in range(nb_h):
        for j in range(nb_w):
            val = data[i, j]
            label = "nan" if np.isnan(val) else f'{val:1.1f}'
            ax.text(j, i, label, ha="center", va="center", color="white", fontsize=9)


def _plot_scatter_multi(ax, target, estimators: dict):
    colors = ["steelblue", "tomato", "seagreen"]
    lim_max = target.max()
    for (label, estimated), color in zip(estimators.items(), colors):
        ax.scatter(target, estimated, alpha=0.75, edgecolors="k",
                   linewidths=0.4, color=color, label=label)
        lim_max = max(lim_max, estimated.max())
    lim = [0, lim_max + 2]
    ax.plot(lim, lim, "r--", linewidth=1)
    ax.set_xlabel("Target dim"); ax.set_ylabel("Estimated dim")
    ax.set_title("Target vs estimated (all methods)", fontsize=11)
    ax.legend(fontsize=9)


def _plot_residuals(ax, target, estimated, title="Residuals"):
    residuals = estimated - target
    xs = range(len(residuals))
    ax.bar(xs, residuals, color="slategray", edgecolor="k", linewidth=0.5)
    ax.axhline(0, color="red", linewidth=1, linestyle="--")
    ax.set_xticks(list(xs))
    ax.set_xticklabels(target.astype(int), rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("Target dim"); ax.set_ylabel("estimated − target dim")
    ax.set_title(title, fontsize=11)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Submanifold patch dimensionality test")
    parser.add_argument("--manifold",       type=str,   default="linear",
                        choices=list_manifolds())
    parser.add_argument("--density",        type=str,   default=None,
                        choices=list_densities())
    parser.add_argument("--patch_size",     type=int,   default=8)
    parser.add_argument("--nb_channels",    type=int,   default=1)
    parser.add_argument("--n_samples",      type=int,   default=25000)
    parser.add_argument("--n_anchors",      type=int,   default=1000)
    parser.add_argument("--k_mle",          type=int,   default=10)
    parser.add_argument("--grid",           type=str,   default="powers",
                        help="Dim grid: 'auto' (4x4 default), 'linear' (1x16 sweep), "
                             "'powers' (powers of 2), or comma-separated ints e.g. '1,4,16,64'")
    parser.add_argument("--pca_threshold",  type=float, default=0.999,
                        help="Cumulative variance threshold for PCA (default: 0.999)")
    parser.add_argument("--n_trials",       type=int,   default=10,
                        help="Number of trials for MLE variance estimation")
    parser.add_argument("--n_subsample",    type=int,   default=1000,
                        help="Subsample size per trial for MLE variance estimation")
    parser.add_argument("--variance_mode",   type=str,   default="aggregated",
                        choices=["aggregated", "per_sample"],
                        help="'aggregated': trial-to-trial variability of the aggregated estimate; "
                             "'per_sample': average per-anchor variance across trials")
    args = parser.parse_args()

    plot_submanifold_test(
        manifold=args.manifold,
        density=args.density,
        patch_size=args.patch_size,
        nb_channels=args.nb_channels,
        n_samples=args.n_samples,
        n_anchors=args.n_anchors,
        k_mle=args.k_mle,
        grid=args.grid,
        pca_threshold=args.pca_threshold,
        n_trials=args.n_trials,
        n_subsample=args.n_subsample,
        variance_mode=args.variance_mode,
    )