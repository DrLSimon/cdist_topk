import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import numpy as np
import warnings
import argparse
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.gridspec import GridSpecFromSubplotSpec


from intrinsic_dim import get_estimator
from intrinsic_dim.estimators.mle_variance import compute_mle_dims_variance, compute_mle_dims_sample_variance
from intrinsic_dim.estimators.diagnostics import check_poisson_regime
from intrinsic_dim.synthetic.sampling import sample_patches, list_manifolds, list_densities, get_max_dim


def make_dims_grid(max_d: int) -> torch.Tensor:
    """Powers-of-2 grid from 1 to max_d, returned as a (1, N) tensor."""
    vals = []
    v = 1
    while v <= max_d:
        vals.append(v)
        v *= 2
    if vals[-1] != max_d:
        vals.append(max_d)
    return torch.tensor(vals).unsqueeze(0)  # (1, N)


def plot_submanifold_test(full_dim=64, n_samples=25000, k_mle=10, unbiased=True,
                          n_anchors=1000, manifold="linear", density=None,
                          pca_threshold=0.999, n_trials=10, n_subsample=1000,
                          extra_debug_plot=None):
    """
    extra_debug_plot: None               — 2-column layout, no debug
                      "variance"         — both estimator and per-sample STD stacked
                      "poisson_validity" — Poisson regime diagnostics stacked
    """
    max_d  = get_max_dim(manifold, full_dim)
    dims   = make_dims_grid(max_d)
    target = dims.cpu().numpy().flatten()

    samples = sample_patches(dims, patch_size=1, nb_channels=full_dim, n_samples=n_samples,
                              manifold=manifold, density=density)
    pca_estimator          = get_estimator("pca", threshold=pca_threshold)
    mle_estimator          = get_estimator("mle", k=k_mle, n_anchors=n_anchors, unbiased=unbiased)
    mle_avg_estimator      = get_estimator("mle_avg", k=k_mle, n_anchors=n_anchors, unbiased=unbiased)

    pca_dims               = pca_estimator(samples)
    pca_dims_np            = pca_dims.cpu().numpy()
    mle_dims               = mle_estimator(samples)
    mle_dims_np            = mle_dims.cpu().numpy()
    mle_avg_dims           = mle_avg_estimator(samples)
    mle_avg_dims_np        = mle_avg_dims.cpu().numpy()

    # ── extra debug data ──────────────────────────────────────────────────────
    # each entry: (data_np, title, is_bool)
    extra_panels = []

    if extra_debug_plot == "variance":
        var_est = compute_mle_dims_variance(
            samples, k=k_mle, n_anchors=n_anchors,
            n_subsample=n_subsample, n_trials=n_trials, unbiased=unbiased)
        var_ps = compute_mle_dims_sample_variance(
            samples, k=k_mle, n_anchors=n_anchors,
            n_subsample=n_subsample, n_trials=n_trials)
        extra_panels = [
            (var_est.sqrt().cpu().numpy(), f"STD MLE (estimator, k={k_mle})", False, dims),
            (var_ps.sqrt().cpu().numpy(),  f"STD MLE (per sample, k={k_mle})", False, dims),
        ]

    elif extra_debug_plot == "poisson_validity":
        is_valid, stats = check_poisson_regime(samples, k=k_mle, n_anchors=n_anchors)
        # failure reason: 0=valid, 1=k_over_n, 2=r_ratio, 3=ks_pvalue (first failing criterion)
        reason = np.zeros(dims.shape[1], dtype=int)
        for j in range(dims.shape[1]):
            if not is_valid[0, j]:
                if stats['k_over_n'][0, j] >= 0.01:
                    reason[j] = 1
                elif stats['r_ratio'][0, j] >= 2.0:
                    reason[j] = 2
                else:
                    reason[j] = 3
        extra_panels = [
            (is_valid.float().cpu().numpy(),   "Poisson valid",    True,  None),
            (reason[np.newaxis, :].astype(float), "Failure reason", "reason", None),
            (stats['r_ratio'].cpu().numpy(),   "r_k / r_1",        False, dims),
            (stats['ks_pvalue'].cpu().numpy(), "KS p-value",       False, dims),
        ]

    # ── layout ────────────────────────────────────────────────────────────────
    # 3 columns: col0=scatter+residuals, col1=heatmaps, col2=debug (or hidden)
    n_heatmap_rows = max(4, len(extra_panels))
    has_debug      = extra_debug_plot is not None
    ncols          = 3 if has_debug else 2
    col_widths     = [2, 1, 1] if has_debug else [2, 1]

    fig = plt.figure(figsize=(6 * sum(col_widths), 14))
    gs  = gridspec.GridSpec(1, ncols, figure=fig, width_ratios=col_widths, wspace=0.35)

    # col 0: scatter on top, residuals on bottom (nested grid, equal height)
    gs0 = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[0], hspace=0.45)
    ax_scatter = fig.add_subplot(gs0[0])
    ax_res_gs  = GridSpecFromSubplotSpec(1, 3, subplot_spec=gs0[1], wspace=0.4)
    ax_res     = [fig.add_subplot(ax_res_gs[i]) for i in range(3)]

    _plot_scatter_multi(ax_scatter, target, {
        f"PCA ({pca_threshold:.0%})": pca_dims_np.flatten(),
        f"MLE k={k_mle}":             mle_dims_np.flatten(),
        f"MLE avg k=5..{k_mle}":      mle_avg_dims_np.flatten(),
    })
    _plot_residuals(ax_res[0], target, pca_dims_np.flatten(),     "PCA residuals")
    _plot_residuals(ax_res[1], target, mle_dims_np.flatten(),     f"MLE residuals (k={k_mle})")
    _plot_residuals(ax_res[2], target, mle_avg_dims_np.flatten(), f"MLE avg residuals (k=3..{k_mle})")

    # col 1: heatmaps stacked (target, PCA, MLE, MLE avg)
    heatmap_data = [
        (dims.cpu().numpy(),  "Target intrinsic dim",           dims),
        (pca_dims_np,         f"PCA ({pca_threshold:.0%} var)", dims),
        (mle_dims_np,         f"MLE dim (k={k_mle})",           dims),
        (mle_avg_dims_np,     f"MLE avg dim (k=3..{k_mle})",    dims),
    ]
    gs1 = GridSpecFromSubplotSpec(n_heatmap_rows, 1, subplot_spec=gs[1], hspace=0.6)
    for row, (data_np, title, dims_ref) in enumerate(heatmap_data):
        _plot_heatmap(fig.add_subplot(gs1[row]), data_np, title, dims_ref)
    for row in range(len(heatmap_data), n_heatmap_rows):
        fig.add_subplot(gs1[row]).set_visible(False)

    # col 2: debug panels stacked (aligned with col 1 rows)
    n_heatmap_rows = max(4, len(extra_panels))
    if has_debug:
        gs2 = GridSpecFromSubplotSpec(n_heatmap_rows, 1, subplot_spec=gs[2], hspace=0.6)
        for row, (data_np, title, kind, dims_ref) in enumerate(extra_panels):
            ax = fig.add_subplot(gs2[row])
            if kind is True:
                _plot_bool_heatmap(ax, data_np, title, target)
            elif kind == "reason":
                _plot_reason_heatmap(ax, data_np, title, target)
            else:
                _plot_heatmap(ax, data_np, title, dims_ref)
        for row in range(len(extra_panels), n_heatmap_rows):
            fig.add_subplot(gs2[row]).set_visible(False)

    distrib_label = f"{manifold}:{density or 'default'}"
    fig.suptitle(
        f"Submanifold patch test  |  distrib={distrib_label}  full_dim={full_dim}  "
        f"n_samples={n_samples}  n_anchors={n_anchors}  "
        f"pca_threshold={pca_threshold:.0%}  k_mle={k_mle}"
        + (f"  debug={extra_debug_plot}" if extra_debug_plot else ""),
        fontsize=12,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*tight_layout.*")
        plt.tight_layout()

    density_label = density or "default"
    debug_label   = f"__{extra_debug_plot}" if extra_debug_plot else ""
    unbias_label   = "__biased" if not unbiased else ""
    fname = f"{manifold}__{density_label}__k{k_mle}{unbias_label}__fd{full_dim}{debug_label}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    print(f"Saved to {fname}")
    plt.show()


# ── Axes helpers ──────────────────────────────────────────────────────────────

def _heatmap_xticks(ax, target, nb_w):
    ax.set_xticks(range(nb_w))
    ax.set_xticklabels(target.astype(int), rotation=45, ha="right", fontsize=7)
    ax.set_xlabel("target dim", fontsize=8)
    ax.set_yticks([])


def _plot_heatmap(ax, data, title, dims_ref):
    nb_h, nb_w = data.shape
    target_vals = dims_ref.flatten().cpu().numpy() if isinstance(dims_ref, torch.Tensor) else dims_ref.flatten()
    im = ax.imshow(data, cmap="viridis", aspect="auto",
                   vmin=dims_ref.min().item() if isinstance(dims_ref, torch.Tensor) else dims_ref.min(),
                   vmax=dims_ref.max().item() if isinstance(dims_ref, torch.Tensor) else dims_ref.max())
    ax.set_title(title, fontsize=9)
    _heatmap_xticks(ax, target_vals, nb_w)
    plt.colorbar(im, ax=ax, fraction=0.046)
    for j in range(nb_w):
        val = data[0, j]
        ax.text(j, 0, "nan" if np.isnan(val) else f'{val:1.1f}',
                ha="center", va="center", color="white", fontsize=8)


def _plot_reason_heatmap(ax, data, title, target):
    """Categorical heatmap showing the first failing Poisson criterion per patch."""
    # 0=valid(green), 1=k_over_n(red), 2=r_ratio(orange), 3=ks_pvalue(blue)
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    cmap   = ListedColormap(["#2ca02c", "#d62728", "#ff7f0e", "#1f77b4"])
    labels = ["valid", "k/n too large", "r_ratio too large", "KS p-val too small"]
    nb_h, nb_w = data.shape
    ax.imshow(data, cmap=cmap, aspect="auto", vmin=0, vmax=3)
    ax.set_title(title, fontsize=9)
    _heatmap_xticks(ax, target, nb_w)
    legend_patches = [Patch(color=cmap(i/3), label=labels[i]) for i in range(4)]
    ax.legend(handles=legend_patches, fontsize=6, loc="upper left",
              bbox_to_anchor=(0, -0.35), ncol=2)


def _plot_bool_heatmap(ax, data, title, target):
    nb_h, nb_w = data.shape
    ax.imshow(data, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
    ax.set_title(title, fontsize=9)
    _heatmap_xticks(ax, target, nb_w)
    for j in range(nb_w):
        ax.text(j, 0, "✓" if data[0, j] > 0.5 else "✗",
                ha="center", va="center", color="black", fontsize=10)


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
    parser.add_argument("--manifold",         type=str,   default="linear",
                        choices=list_manifolds())
    parser.add_argument("--density",          type=str,   default=None,
                        choices=list_densities())
    parser.add_argument("--full_dim",         type=int,   default=64)
    parser.add_argument("--n_samples",        type=int,   default=25000)
    parser.add_argument("--n_anchors",        type=int,   default=1000)
    parser.add_argument("--k_mle",            type=int,   default=10)
    parser.add_argument("--no-unbiased", dest="unbiased", action="store_false", help="Disable bias correction for MLE estimator")
    parser.add_argument("--pca_threshold",    type=float, default=0.999)
    parser.add_argument("--n_trials",         type=int,   default=10)
    parser.add_argument("--n_subsample",      type=int,   default=1000)
    parser.add_argument("--extra_debug_plot", type=str,   default=None,
                        choices=["variance", "poisson_validity"])
    args = parser.parse_args()

    plot_submanifold_test(
        manifold=args.manifold,
        density=args.density,
        full_dim=args.full_dim,
        n_samples=args.n_samples,
        n_anchors=args.n_anchors,
        k_mle=args.k_mle,
        unbiased=args.unbiased,
        pca_threshold=args.pca_threshold,
        n_trials=args.n_trials,
        n_subsample=args.n_subsample,
        extra_debug_plot=args.extra_debug_plot,
    )