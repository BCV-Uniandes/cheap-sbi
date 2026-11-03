"""
paper_figures.py
----------------
Every figure of the camera-ready, in one consistent style: Computer Modern type,
boxed axes with inward major and minor ticks on all four sides, black-edged markers,
shaded uncertainty bands, dotted reference lines, panel labels inside the axes.

Figures are drawn at their final printed size (NeurIPS text width = 5.5 in), so the
fonts in the PDF match the caption size and nothing gets rescaled in LaTeX. Each
figure is written as a vector PDF (for the paper) and a 300 dpi PNG (for slides).

    premise_validation   Fig. 1   computed here from telegraph.py (no training)
    money_plot           Fig. 2   from results_main.npz   (run_experiment.py)
    calibration_curves   Fig. 3   from results_main.npz   (run_experiment.py)
    budget_rule          Fig. 4   from budget_curves.npz  (run_budget.py)
    sbc_histograms       App. A   from results_main.npz   (run_experiment.py)

Usage (re-plots from saved data, no retraining):
    python paper_figures.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

TEXTWIDTH = 5.5                     # inches

RC = {
    "font.family": "serif",
    "font.serif": ["cmr10", "Computer Modern Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True,
    "axes.unicode_minus": False,
    "font.size": 8.5,
    "axes.labelsize": 8.5,
    "axes.titlesize": 8.5,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.2,
    "axes.linewidth": 0.8,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True,
    "xtick.minor.visible": True, "ytick.minor.visible": True,
    "xtick.major.size": 3.5, "ytick.major.size": 3.5,
    "xtick.minor.size": 1.8, "ytick.minor.size": 1.8,
    "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    "xtick.minor.width": 0.5, "ytick.minor.width": 0.5,
    "xtick.major.pad": 3, "ytick.major.pad": 3,
    "legend.frameon": True, "legend.fancybox": False,
    "legend.edgecolor": "black", "legend.framealpha": 1.0,
    "legend.borderpad": 0.45, "legend.handlelength": 2.0,
    "patch.linewidth": 0.6,
    "lines.linewidth": 1.3, "lines.markersize": 4.2,
    "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.015,
    "pdf.fonttype": 42, "ps.fonttype": 42,
}

# controls in blues, Gaussian surrogates in reds
COL = {"cme": "#2b5797", "nb": "#7aa6d8", "lna": "#d26a70", "cle": "#8e1b24"}
MK = {"cme": "o", "nb": "D", "lna": "^", "cle": "s"}
NAME = {"cme": "CME (control)", "nb": "NB (non-Gaussian control)",
        "lna": "LNA (Gaussian surrogate)", "cle": "CLE (Gaussian surrogate)"}
ORDER = ["cme", "nb", "lna", "cle"]
GREY_BAND = "0.88"
PARAMS_TEX = [r"$k_{\mathrm{on}}$", r"$k_{\mathrm{off}}$", r"$k_{\mathrm{syn}}$"]


def use_style():
    plt.rcParams.update(RC)


def panel_label(ax, text, x=0.04, y=0.955, ha="left", va="top", **kw):
    ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va, fontsize=9, **kw)


def marker_kw(sim, ms=4.2):
    return dict(marker=MK[sim], ms=ms, mfc=COL[sim], mec="black", mew=0.6)


def save(fig, stem):
    fig.savefig(stem + ".pdf")
    fig.savefig(stem + ".png", dpi=300)
    plt.close(fig)
    print(f"[fig] {stem}.pdf / .png")


# ----------------------------------------------------------------------
# shared statistics
# ----------------------------------------------------------------------
def fano_bins(fano, n_bins=6):
    """Equal-count bins of the empirical Fano factor of the shared test set."""
    edges = np.quantile(fano, np.linspace(0, 1, n_bins + 1)); edges[-1] += 1e-6
    idx = np.clip(np.digitize(fano, edges) - 1, 0, n_bins - 1)
    centers = np.array([np.median(fano[idx == b]) for b in range(n_bins)])
    centers = np.maximum(centers, 0.1)          # all-zero snapshots have Fano 0
    return edges, centers, idx


def bootstrap_bins(vals, idx, n_bins, fn, n_boot=1000, seed=1):
    rng = np.random.default_rng(seed)
    m = np.full(n_bins, np.nan); lo = m.copy(); hi = m.copy()
    for b in range(n_bins):
        v = vals[idx == b]
        if len(v) < 4:
            continue
        m[b] = fn(v)
        boot = [fn(v[rng.integers(0, len(v), len(v))]) for _ in range(n_boot)]
        lo[b], hi[b] = np.percentile(boot, [2.5, 97.5])
    return m, lo, hi


def in_central(u, level):
    return np.abs(u - 0.5) <= level / 2


def load_results(path="results_main.npz"):
    d = np.load(path)
    res = {s: (d[f"bias_{s}"], d[f"ranks_{s}"], int(d["n_post"])) for s in ORDER
           if f"ranks_{s}" in d}
    return res, d["theta_test"], d["emp_fano"]


# ----------------------------------------------------------------------
# Fig. 1: the CLE against the exact CME along a burst-frequency sweep
# ----------------------------------------------------------------------
def fig_premise(out="premise_validation", k_off=1.0, k_syn=40.0, n_cells=20000,
                seed=0, bursty_until=0.5):
    """CME curves are exact (closed form); CLE points come from the batched
    Euler-Maruyama simulator used to train the NPE (dt = 0.02), 20,000 cells each."""
    from scipy.special import hyp1f1
    from telegraph import exact_mean, exact_fano, sample_cle_batched

    use_style()
    kon_pts = np.geomspace(0.025, 20.0, 15)
    kon_line = np.geomspace(0.02, 25.0, 300)
    ex = {"mean": exact_mean(kon_line, k_off, k_syn),
          "fano": exact_fano(kon_line, k_off, k_syn),
          "zero": hyp1f1(kon_line, kon_line + k_off, -k_syn)}
    ex_pts = {"mean": exact_mean(kon_pts, k_off, k_syn),
              "fano": exact_fano(kon_pts, k_off, k_syn),
              "zero": hyp1f1(kon_pts, kon_pts + k_off, -k_syn)}
    rng = np.random.default_rng(seed)
    cle = {"mean": [], "fano": [], "zero": []}
    for k in kon_pts:
        c = sample_cle_batched(np.full(n_cells, k), np.full(n_cells, k_off),
                               np.full(n_cells, k_syn), rng).astype(float)
        cle["mean"].append(c.mean()); cle["fano"].append(c.var() / c.mean())
        cle["zero"].append(np.mean(c == 0))
    cle = {k: np.array(v) for k, v in cle.items()}

    print("\n--- Fig. 1 values quoted in Section 5.1 ---")
    print(f"  bursty end k_on={kon_pts[0]:.3f}: CME mean={ex_pts['mean'][0]:.2f} "
          f"Fano={ex_pts['fano'][0]:.1f} zero={ex_pts['zero'][0]:.2f} | CLE mean="
          f"{cle['mean'][0]:.2f} Fano={cle['fano'][0]:.2f} zero={cle['zero'][0]:.3f}")
    print(f"  fast end k_on={kon_pts[-1]:.0f}: CME mean={ex_pts['mean'][-1]:.1f} CLE mean="
          f"{cle['mean'][-1]:.1f}  ({100*(1-cle['mean'][-1]/ex_pts['mean'][-1]):.0f}% lower)")

    fig, axes = plt.subplots(1, 3, figsize=(TEXTWIDTH + 0.21, 1.66))
    keys = ["mean", "fano", "zero"]
    ylabels = [r"mean mRNA $\mathbb{E}[m]$", "Fano factor", r"zero fraction $P(m{=}0)$"]
    for j, (ax, k) in enumerate(zip(axes, keys)):
        ax.axvspan(1e-3, bursty_until, color=GREY_BAND, lw=0, zorder=0)
        ax.plot(kon_line, ex[k], "-", color=COL["cme"], lw=1.4, zorder=2)
        ax.plot(kon_pts, ex_pts[k], ls="none", **marker_kw("cme"), zorder=3)
        ax.plot(kon_pts, cle[k], "--", color=COL["cle"], lw=1.2, zorder=2)
        ax.plot(kon_pts, cle[k], ls="none", **marker_kw("cle", ms=3.8), zorder=3)
        ax.set_xscale("log"); ax.set_xlim(0.018, 28)
        ax.set_xlabel(r"burst frequency $k_{\mathrm{on}}$")
        ax.set_ylabel(ylabels[j])
    axes[0].set_ylim(0, 42); axes[1].set_ylim(0, 22.5); axes[2].set_ylim(-0.03, 1.0)
    panel_label(axes[0], "(a)")
    panel_label(axes[1], "(b)", x=0.955, ha="right")
    panel_label(axes[2], "(c)", x=0.955, ha="right")
    axes[1].text(0.06, 0.05, "bursty", transform=axes[1].transAxes, fontsize=7.5,
                 color="0.35", ha="left", va="bottom")
    handles = [Line2D([], [], color=COL["cme"], lw=1.4, **marker_kw("cme")),
               Line2D([], [], color=COL["cle"], lw=1.2, ls="--", **marker_kw("cle", ms=3.8))]
    axes[2].legend(handles, ["CME", "CLE"], loc="center right", handlelength=1.6,
                   bbox_to_anchor=(0.985, 0.5))
    fig.tight_layout(w_pad=0.9)
    save(fig, out)


# ----------------------------------------------------------------------
# Fig. 2: posterior-mean error and k_on coverage against the empirical Fano
# ----------------------------------------------------------------------
def fig_money(results, emp_fano, out="money_plot", n_bins=6):
    use_style()
    edges, centers, idx = fano_bins(emp_fano, n_bins)
    cov_fn = lambda v: np.mean((v >= 0.05) & (v <= 0.95))
    fig, ax = plt.subplots(1, 2, figsize=(TEXTWIDTH, 2.2))
    for sim in ORDER:
        if sim not in results:
            continue
        bias, ranks, n_post = results[sim]
        for a, vals, fn in [(ax[0], np.linalg.norm(bias, axis=1), np.mean),
                            (ax[1], ranks[:, 0] / n_post, cov_fn)]:
            m, lo, hi = bootstrap_bins(vals, idx, n_bins, fn)
            a.fill_between(centers, lo, hi, color=COL[sim], alpha=0.16, lw=0, zorder=1)
            a.plot(centers, m, "-", color=COL[sim], lw=1.3, zorder=2)
            a.plot(centers, m, ls="none", **marker_kw(sim), zorder=3)
    ax[1].axhline(0.9, ls=":", color="0.35", lw=0.9, zorder=0)
    ax[0].set_ylabel(r"posterior-mean error ($\log_{10}$)")
    ax[1].set_ylabel(r"90% CI coverage of $k_{\mathrm{on}}$")
    ax[1].set_ylim(-0.03, 1.03); ax[0].set_ylim(bottom=0)
    lo_x = min(0.8, 0.8 * np.min(centers))
    for a in ax:
        a.set_xscale("log"); a.set_xlim(lo_x, 1.25 * np.max(centers))
        a.set_xlabel("empirical Fano factor of the snapshot")
    panel_label(ax[0], "(a)"); panel_label(ax[1], "(b)", y=0.06, va="bottom")
    handles = [Line2D([], [], color=COL[s], lw=1.3, **marker_kw(s)) for s in ORDER if s in results]
    fig.legend(handles, [NAME[s] for s in ORDER if s in results], loc="upper center",
               ncol=4, bbox_to_anchor=(0.5, 1.065), columnspacing=1.2, handletextpad=0.5)
    fig.tight_layout(w_pad=1.6)
    save(fig, out)


# ----------------------------------------------------------------------
# Fig. 3: reliability (expected-coverage) curves
# ----------------------------------------------------------------------
def fig_calibration(results, out="calibration_curves"):
    """Three square reliability panels across the full text width, with the same
    one-row legend on top as Fig. 2."""
    use_style()
    levels = np.linspace(0.0, 1.0, 51)
    fig, ax = plt.subplots(1, 3, figsize=(TEXTWIDTH + 0.1, 2.05), sharey=True)
    for sim in ORDER:
        if sim not in results:
            continue
        _, ranks, n_post = results[sim]
        u = ranks / n_post
        cov = np.array([[in_central(u[:, j], l).mean() for j in range(3)] for l in levels])
        for j in range(3):
            ax[j].plot(levels, cov[:, j], "-", color=COL[sim], lw=1.3, zorder=2)
            ax[j].plot(levels[5::10], cov[5::10, j], ls="none", **marker_kw(sim, ms=3.6), zorder=3)
    for j, a in enumerate(ax):
        a.plot([0, 1], [0, 1], ls=":", color="0.35", lw=0.9, zorder=0)
        a.set_xlim(0, 1); a.set_ylim(0, 1); a.set_aspect("equal")
        a.set_xticks([0, 0.5, 1]); a.set_yticks([0, 0.5, 1])
        a.set_xlabel("nominal level")
        panel_label(a, f"({'abc'[j]}) " + PARAMS_TEX[j])
    ax[0].set_ylabel("empirical coverage")
    sims = [s for s in ORDER if s in results]
    handles = [Line2D([], [], color=COL[s], lw=1.3, **marker_kw(s)) for s in sims]
    fig.legend(handles, [NAME[s] for s in sims], loc="upper center", ncol=4,
               bbox_to_anchor=(0.5, 1.07), columnspacing=1.2, handletextpad=0.5)
    fig.tight_layout(w_pad=0.6)
    save(fig, out)


# ----------------------------------------------------------------------
# Fig. 4: Fano-guided versus random routing
# ----------------------------------------------------------------------
def fig_budget(fracs, guided, random, out="budget_rule"):
    use_style()
    fig, ax = plt.subplots(figsize=(0.57 * TEXTWIDTH, 2.25))   # sits beside its caption
    ax.axhline(0.9, ls=":", color="0.35", lw=0.9, zorder=0)
    ax.fill_between(fracs, random, guided, color=COL["cle"], alpha=0.10, lw=0, zorder=1)
    ax.plot(fracs, random, "--", color="0.45", lw=1.2, zorder=2)
    ax.plot(fracs, random, ls="none", marker="s", ms=3.0, mfc="0.65", mec="black",
            mew=0.5, zorder=3, markevery=2)
    ax.plot(fracs, guided, "-", color=COL["cle"], lw=1.3, zorder=2)
    ax.plot(fracs, guided, ls="none", marker="o", ms=3.4, mfc=COL["cle"], mec="black",
            mew=0.5, zorder=3, markevery=2)
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(0.4, 1.0)
    ax.set_xlabel("fraction of queries sent to exact")
    ax.set_ylabel(r"90% coverage of $k_{\mathrm{on}}$")
    handles = [Line2D([], [], color=COL["cle"], lw=1.3, marker="o", ms=3.4,
                      mfc=COL["cle"], mec="black", mew=0.5),
               Line2D([], [], color="0.45", ls="--", lw=1.2, marker="s", ms=3.0,
                      mfc="0.65", mec="black", mew=0.5)]
    ax.legend(handles, ["Fano-guided", "random"], loc="lower right")
    ax.text(0.02, 0.905, "nominal", transform=ax.get_yaxis_transform(), ha="left",
            va="bottom", fontsize=7, color="0.35")
    fig.tight_layout()
    save(fig, out)


# ----------------------------------------------------------------------
# Appendix A: SBC rank histograms
# ----------------------------------------------------------------------
def fig_sbc(results, out="sbc_histograms", nb_bins=10):
    use_style()
    sims = [s for s in ORDER if s in results]
    fig, axes = plt.subplots(len(sims), 3, figsize=(TEXTWIDTH, 1.05 * len(sims) + 0.35),
                             sharex=True, sharey=True)
    axes = np.atleast_2d(axes)
    ymax = 0
    for r, sim in enumerate(sims):
        _, ranks, n_post = results[sim]
        n_test = ranks.shape[0]
        exp = n_test / nb_bins
        band = 2.0 * np.sqrt(exp * (1 - 1.0 / nb_bins))
        for c in range(3):
            a = axes[r, c]
            a.axhspan(exp - band, exp + band, color=GREY_BAND, lw=0, zorder=0)
            a.axhline(exp, ls=":", color="0.35", lw=0.8, zorder=1)
            h, _, _ = a.hist(ranks[:, c] / n_post, bins=nb_bins, range=(0, 1),
                             color=COL[sim], edgecolor="black", lw=0.5, zorder=2)
            ymax = max(ymax, h.max())
            if r == 0:
                a.set_title(PARAMS_TEX[c])
            if c == 0:
                a.set_ylabel(sim.upper())
            a.minorticks_off(); a.tick_params(top=True, right=True)
    for a in axes[-1]:
        a.set_xlabel("normalized SBC rank")
    axes[0, 0].set_ylim(0, 1.08 * ymax)
    fig.tight_layout(h_pad=0.4, w_pad=0.6)
    save(fig, out)


# ----------------------------------------------------------------------
def main():
    fig_premise()
    if os.path.exists("results_main.npz"):
        res, _, emp_fano = load_results()
        fig_money(res, emp_fano); fig_calibration(res); fig_sbc(res)
    else:
        print("[skip] results_main.npz not found: run run_experiment.py first")
    if os.path.exists("budget_curves.npz"):
        d = np.load("budget_curves.npz")
        fig_budget(d["fracs"], d["guided"], d["random"])
    else:
        print("[skip] budget_curves.npz not found: run run_budget.py first")


if __name__ == "__main__":
    main()