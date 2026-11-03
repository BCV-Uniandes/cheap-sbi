"""
run_budget.py  (camera-ready version)
-------------------------------------
End-to-end demonstration of the routing rule. A population of inference queries
(test observations spanning burstiness) must be served with either a cheap,
misspecified estimator (LNA-trained) or, for a fraction f of queries, the exact
CME-trained estimator. We compare two policies:

  * Fano-guided: route the highest empirical-Fano queries to the exact estimator.
  * Random: route a random fraction f to the exact estimator.

We report overall 90% CI coverage (for k_on) as a function of f, save the curves to
budget_curves.npz and draw Fig. 4 in the paper style (paper_figures.py).

Camera-ready changes: seeded training and test set, curves saved to disk, figure
drawn by paper_figures.fig_budget. The routing logic is unchanged.

Usage:
    python run_budget.py --n_sims 8000 --n_cells 500 --n_test 600
"""

import argparse
import numpy as np
import torch
import warnings
warnings.filterwarnings("ignore")

import run_experiment as R
import paper_figures as F


def per_obs(posterior, x_test, tt, n_post):
    """Coverage indicator (k_on in central 90% CI) per observation."""
    cov = np.empty(len(tt))
    for i in range(len(tt)):
        s = R._flow_sample(posterior, x_test[i], n_post).numpy()
        rank = (s[:, 0] < tt[i, 0]).mean()
        cov[i] = 1.0 if 0.05 <= rank <= 0.95 else 0.0
    return cov


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_sims", type=int, default=8000)
    p.add_argument("--n_cells", type=int, default=500)
    p.add_argument("--n_test", type=int, default=600)
    p.add_argument("--n_post", type=int, default=1000)
    p.add_argument("--cheap", type=str, default="lna", choices=["lna", "cle"])
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    # shared, seeded test set (same construction as run_experiment.py)
    tt, x_test = R.make_test_set(args.n_test, args.n_cells, seed=999)
    emp_fano = x_test[:, 2].numpy()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    post_exact = R.train("cme", args.n_sims, args.n_cells, args.seed)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    post_cheap = R.train(args.cheap, args.n_sims, args.n_cells, args.seed)
    cov_exact = per_obs(post_exact, x_test, tt, args.n_post)
    cov_cheap = per_obs(post_cheap, x_test, tt, args.n_post)

    order = np.argsort(-emp_fano)                     # high Fano first
    fracs = np.linspace(0, 1, 21)
    N = args.n_test
    guided, random = [], []
    rng2 = np.random.default_rng(0)
    for f in fracs:
        k = int(round(f * N))
        sel = np.zeros(N, bool); sel[order[:k]] = True
        guided.append(np.mean(np.where(sel, cov_exact, cov_cheap)))
        accs = []
        for _ in range(30):
            s = np.zeros(N, bool); s[rng2.choice(N, k, replace=False)] = True
            accs.append(np.mean(np.where(s, cov_exact, cov_cheap)))
        random.append(np.mean(accs))
    guided = np.array(guided); random = np.array(random)

    print(f"\ncheap surrogate = {args.cheap.upper()}")
    print(f"coverage at f=0 (all cheap):  {guided[0]:.2f}")
    print(f"coverage at f=1 (all exact):  {guided[-1]:.2f}")
    for ft in [0.25, 0.5, 0.75]:
        j = int(round(ft * (len(fracs) - 1)))
        print(f"  f={fracs[j]:.2f}:  Fano-guided={guided[j]:.2f}   random={random[j]:.2f}"
              f"   (gap {guided[j]-random[j]:+.2f})")

    def budget_to(target, curve):
        idx = np.where(curve >= target)[0]
        return fracs[idx[0]] if len(idx) else np.nan
    for tgt in [0.80, 0.88]:
        bg, br = budget_to(tgt, guided), budget_to(tgt, random)
        sv = f"{br/bg:.1f}x" if bg and bg > 0 else "n/a"
        print(f"budget to reach {tgt:.2f} coverage:  Fano-guided={bg:.2f}  random={br:.2f}  ({sv})")

    np.savez("budget_curves.npz", fracs=fracs, guided=guided, random=random)
    print("[data] saved budget_curves.npz")
    F.fig_budget(fracs, guided, random)


if __name__ == "__main__":
    main()
