# Predicting When Cheap Simulators Break Neural Inference of Bursty Gene Expression

Code for the paper presented at the **LatinX in AI Workshop at NeurIPS 2026** (non-archival).

Rafael Velásquez and Pablo Arbeláez, Universidad de los Andes, Colombia.

Amortized simulation-based inference (SBI) is often trained on a cheap approximate
simulator rather than on the exact model that generated the data. Using the two-state
(telegraph) model of gene expression, whose exact stationary distribution is known in
closed form, we train neural posterior estimators (NPE) on the exact chemical master
equation (CME) and on three cheap surrogates (chemical Langevin equation, linear noise
approximation and a moment-matched negative binomial), evaluate them all on exact data,
and show that the empirical Fano factor of a snapshot flags where a Gaussian surrogate
breaks posterior calibration.

## Repository contents

| File | What it does |
|---|---|
| `telegraph.py` | Telegraph model: exact Poisson-Beta sampler (CME), CLE, LNA and NB surrogates, closed-form moments, summary statistics |
| `sbi_experiment.py` | Prior, batched simulation of snapshots, summary-statistic features |
| `run_experiment.py` | Trains the four estimators, evaluates them on one shared exact test set, prints Table 1 and the per-bin coverage, saves `results_main.npz` |
| `run_budget.py` | Fano-guided versus random routing of queries (Section 5.5), saves `budget_curves.npz` |
| `run_ablation.py` | Moments-only feature ablation (Section 5.2) |
| `run_seeds.py` | Robustness across five random seeds (Section 5.2) |
| `run_ncells.py` | Robustness to the snapshot size (Section 5.6) |
| `paper_figures.py` | Draws every figure in the paper style from the saved results, and computes Figure 1 |
| `reproduce.sh` | Runs everything in order |

## Installation

```bash
conda create -n cheap-sbi python=3.10
conda activate cheap-sbi
pip install -r requirements.txt
```

Tested with Python 3.10 and `sbi` 0.26.1. A GPU is not required.

## Reproducing the paper

```bash
./reproduce.sh
```

or step by step:

| Paper item | Command |
|---|---|
| Table 1, Figures 2 and 3, Appendix A | `python run_experiment.py --n_sims 8000 --n_cells 500 --n_test 400 --seed 0` |
| Figure 4 | `python run_budget.py --n_sims 8000 --n_cells 500 --n_test 600` |
| Figure 1 (and a redraw of all figures) | `python paper_figures.py` |
| Moments-only ablation | `python run_ablation.py --n_sims 8000 --n_cells 500 --n_test 400` |
| Seed robustness | `python run_seeds.py --n_sims 8000 --n_cells 500 --n_test 400 --seeds 0 1 2 3 4` |
| Snapshot-size robustness | `python run_ncells.py --n_sims 8000 --n_cells 100 200 500 2000 --sims cme cle` |

Training sets are cached in `cache/` after the first run. Generating the CLE training
set is the slowest step (about 6 to 8 minutes for 8,000 simulations on one machine);
each estimator then trains in a few minutes. Figures are written as vector PDFs and
300 dpi PNGs.

Training uses fixed seeds, but the training of normalizing flows is not bit-for-bit
deterministic across hardware and library versions, so re-runs can differ from the
reported numbers in the second decimal.

## Notes

* Every estimator is evaluated on the same held-out set of exact CME snapshots, and every
  trained flow, the control included, is sampled directly rather than by rejection
  within the prior box (misspecified flows place mass outside it).
* Burstiness is binned by the **empirical** Fano factor of each snapshot, a quantity
  available at inference time, in equal-count sextiles of the shared test set.

## Citation

```bibtex
@inproceedings{velasquez2026predicting,
  title     = {Predicting When Cheap Simulators Break Neural Inference of Bursty Gene Expression},
  author    = {Vel{\'a}squez, Rafael and Arbel{\'a}ez, Pablo},
  booktitle = {LatinX in AI Workshop at the 40th Conference on Neural Information Processing Systems (NeurIPS 2026)},
  year      = {2026},
  note      = {Non-archival}
}
```

## License

MIT, see `LICENSE`.
