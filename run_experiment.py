"""
run_experiment.py  (camera-ready version)
-----------------------------------------
End-to-end driver for the misspecification experiment.

  1. Build & CACHE training sets from the CME, CLE, LNA and NB simulators.
  2. Train one NPE per simulator (seeded).
  3. Evaluate ALL estimators on ONE shared test set of exact CME snapshots.
  4. Save everything to results_main.npz and draw, in the paper style
     (paper_figures.py):
       money_plot.pdf          Fig. 2  (error and k_on coverage vs empirical Fano)
       calibration_curves.pdf  Fig. 3  (reliability / expected-coverage curves)
       sbc_histograms.pdf      App. A  (SBC rank histograms)
     plus perbin_kon.csv, and print Table 1 and the per-bin numbers of Sec. 5.2.

Usage:
    python run_experiment.py --n_sims 8000 --n_cells 500 --n_test 400 --seed 0
    python paper_figures.py          # re-draw the figures later, no retraining

CAMERA-READY FIX. The submitted version drew theta_test inside evaluate() from
torch's global RNG, which training had already advanced, so every estimator saw a
DIFFERENT test set; money_plot() then binned all of them with the empirical Fano of
the LAST estimator's test set (NB). Only the NB curve was binned correctly; the CME,
LNA and CLE curves were scattered across bins at random, which flattened them. Here
the test set is drawn once, from its own generator, and shared by every estimator.
Aggregate numbers (Table 1, seeds, n_cells, ablation, budget) were not affected.

Sampling note: a CLE-trained posterior queried on CME data is out-of-distribution
and places mass outside the prior box, so sbi's rejection sampler accepts ~0% and
hangs. We sample every trained flow DIRECTLY (no prior rejection), the control
included, so all estimators are evaluated on the same object.
"""

import argparse, csv, os, time
import numpy as np
import torch
import warnings
warnings.filterwarnings("ignore")

from sbi.inference import NPE

from sbi_experiment import make_prior, simulate_batch, theta_to_rates, LOW, HIGH
from telegraph import exact_fano
import paper_figures as F

CACHE = "cache"
os.makedirs(CACHE, exist_ok=True)
PARAM_NAMES = ["k_on", "k_off", "k_syn"]
SIMS = ["cme", "cle", "lna", "nb"]


# ----------------------------------------------------------------------
# training
# ----------------------------------------------------------------------
def cached_training_set(simulator, n_sims, n_cells, seed):
    path = f"{CACHE}/train_{simulator}_{n_sims}_{n_cells}_{seed}.npz"
    if os.path.exists(path):
        d = np.load(path)
        print(f"[cache] loaded {path}")
        return torch.as_tensor(d["theta"], dtype=torch.float32), \
               torch.as_tensor(d["x"], dtype=torch.float32)
    print(f"[sim] generating {n_sims} '{simulator}' sims (cached after) ...")
    rng = np.random.default_rng(seed)
    prior = make_prior()
    theta = prior.sample((n_sims,))
    t = time.time()
    x = simulate_batch(theta.numpy(), simulator, n_cells, rng)
    print(f"      done in {time.time()-t:.0f}s")
    np.savez(path, theta=theta.numpy(), x=x)
    return theta, torch.as_tensor(x, dtype=torch.float32)


def train(simulator, n_sims, n_cells, seed):
    theta, x = cached_training_set(simulator, n_sims, n_cells, seed)
    inf = NPE(prior=make_prior())
    inf.append_simulations(theta, x)
    de = inf.train()
    return inf.build_posterior(de)


def _flow_sample(posterior, x_row, n_post):
    """Sample the trained flow directly, bypassing prior-rejection sampling."""
    s = posterior.posterior_estimator.sample((n_post,), condition=x_row.unsqueeze(0))
    return s.reshape(n_post, -1).detach()


# ----------------------------------------------------------------------
# shared test set + evaluation
# ----------------------------------------------------------------------
def make_test_set(n_test, n_cells, seed=999):
    """ONE test set of exact CME snapshots, drawn from its own generators so it
    does not depend on how much of torch's global RNG training consumed."""
    g = torch.Generator().manual_seed(seed)
    low = torch.as_tensor(LOW, dtype=torch.float32)
    high = torch.as_tensor(HIGH, dtype=torch.float32)
    theta = low + (high - low) * torch.rand((n_test, 3), generator=g)   # uniform box
    tt = theta.numpy().astype(np.float64)
    x = simulate_batch(tt, "cme", n_cells, np.random.default_rng(seed))
    return tt, torch.as_tensor(x, dtype=torch.float32)


def evaluate_on(posterior, tt, x_test, n_post=1000):
    """Posterior-mean error and SBC ranks of one estimator on a GIVEN test set."""
    n_test = len(tt)
    post_means = np.empty((n_test, 3))
    ranks = np.empty((n_test, 3))
    for i in range(n_test):
        s = _flow_sample(posterior, x_test[i], n_post).numpy()
        post_means[i] = s.mean(0)
        ranks[i] = (s < tt[i]).sum(0)
    return post_means - tt, ranks


def evaluate(posterior, n_test, n_cells, seed, n_post=1000):
    """Backward-compatible wrapper used by run_seeds.py and run_ncells.py.
    Returns tt, bias, ranks, n_post, emp_fano for a test set built from `seed`."""
    tt, x_test = make_test_set(n_test, n_cells, seed)
    bias, ranks = evaluate_on(posterior, tt, x_test, n_post)
    return tt, bias, ranks, n_post, x_test[:, 2].numpy()


# ----------------------------------------------------------------------
# metrics and tables
# ----------------------------------------------------------------------
def sbc_calibration_error(ranks, n_post):
    """Mean |empirical CDF - uniform| of SBC ranks (0 = perfectly calibrated)."""
    u = (ranks + 0.5) / (n_post + 1)
    grid = np.linspace(0, 1, 50)
    errs = []
    for j in range(u.shape[1]):
        ecdf = np.searchsorted(np.sort(u[:, j]), grid) / len(u)
        errs.append(np.mean(np.abs(ecdf - grid)))
    return np.array(errs)


def coverage_90(ranks, n_post):
    """Empirical coverage of the central 90% credible interval, per parameter."""
    u = ranks / n_post
    return np.mean((u >= 0.05) & (u <= 0.95), axis=0)


def print_table(results):
    """Table 1: posterior-mean abs. error and 90% coverage per parameter."""
    print("\n--- Table 1 (posterior-mean abs. error / 90% coverage per parameter) ---")
    for sim, (bias, ranks, n_post) in results.items():
        ab = np.abs(bias).mean(0); cov = coverage_90(ranks, n_post)
        print(f"  {sim.upper():3s}  err " +
              " ".join(f"{n}={ab[i]:.2f}" for i, n in enumerate(PARAM_NAMES)) +
              "  | cov90 " +
              " ".join(f"{n}={cov[i]:.2f}" for i, n in enumerate(PARAM_NAMES)))


def perbin_report(results, emp_fano, csv_out="perbin_kon.csv", n_bins=6):
    """k_on 90% coverage per empirical-Fano sextile of the SHARED test set, with
    bootstrap 95% intervals: the numbers behind Fig. 2(b) and Section 5.2."""
    edges, _, idx = F.fano_bins(emp_fano, n_bins)
    cov_fn = lambda v: np.mean((v >= 0.05) & (v <= 0.95))
    rows = []
    print("\n--- k_on 90% coverage per empirical-Fano sextile (shared test set) ---")
    print("  edges: " + "  ".join(f"{e:.2f}" for e in edges))
    print("  n/bin: " + "  ".join(str(int((idx == b).sum())) for b in range(n_bins)))
    for sim in F.ORDER:
        if sim not in results:
            continue
        _, ranks, n_post = results[sim]
        m, lo, hi = F.bootstrap_bins(ranks[:, 0] / n_post, idx, n_bins, cov_fn)
        for b in range(n_bins):
            rows.append([sim, b, f"{edges[b]:.2f}", f"{edges[b+1]:.2f}",
                         int((idx == b).sum()), f"{m[b]:.3f}", f"{lo[b]:.3f}", f"{hi[b]:.3f}"])
        print(f"  {sim.upper():3s}: " + "  ".join(f"{v:.2f}" for v in m) +
              f"    (least bursty {m[0]:.2f} -> most bursty {m[-1]:.2f})")
    with open(csv_out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sim", "bin", "fano_lo", "fano_hi", "n", "cov_kon", "ci_lo", "ci_hi"])
        w.writerows(rows)
    print(f"[table] saved {csv_out}")


def save_results(results, theta_test, emp_fano, path="results_main.npz"):
    payload = {"theta_test": theta_test, "emp_fano": emp_fano,
               "n_post": next(iter(results.values()))[2]}
    for sim, (bias, ranks, _) in results.items():
        payload[f"bias_{sim}"] = bias; payload[f"ranks_{sim}"] = ranks
    np.savez(path, **payload)
    print(f"[data] saved {path}")


# ----------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_sims", type=int, default=8000)
    p.add_argument("--n_cells", type=int, default=500)
    p.add_argument("--n_test", type=int, default=400)
    p.add_argument("--n_post", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    # ONE test set, shared by every estimator (the camera-ready fix)
    theta_test, x_test = make_test_set(args.n_test, args.n_cells, seed=999)
    emp_fano = x_test[:, 2].numpy()

    results = {}
    for sim in SIMS:
        print(f"\n===== {sim.upper()} =====")
        torch.manual_seed(args.seed); np.random.seed(args.seed)
        post = train(sim, args.n_sims, args.n_cells, args.seed)
        bias, ranks = evaluate_on(post, theta_test, x_test, args.n_post)
        results[sim] = (bias, ranks, args.n_post)
        ce = sbc_calibration_error(ranks, args.n_post)
        cov = coverage_90(ranks, args.n_post)
        print("  mean |error| (log10): " +
              "  ".join(f"{n}={np.abs(bias)[:,i].mean():.3f}" for i, n in enumerate(PARAM_NAMES)))
        print("  SBC calib error:      " +
              "  ".join(f"{n}={ce[i]:.3f}" for i, n in enumerate(PARAM_NAMES)))
        print("  90% CI coverage:      " +
              "  ".join(f"{n}={cov[i]:.2f}" for i, n in enumerate(PARAM_NAMES)))

    ana = np.array([exact_fano(*r) for r in theta_to_rates(theta_test)])
    corr = np.corrcoef(np.log(ana), np.log(np.clip(emp_fano, 1e-3, None)))[0, 1]
    print(f"\n[diagnostic] corr(log analytic Fano, log empirical Fano) = {corr:.3f}")

    save_results(results, theta_test, emp_fano)
    print_table(results)
    perbin_report(results, emp_fano)
    F.fig_money(results, emp_fano)
    F.fig_calibration(results)
    F.fig_sbc(results)


if __name__ == "__main__":
    main()
