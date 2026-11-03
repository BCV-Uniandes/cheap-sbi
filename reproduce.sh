#!/usr/bin/env bash
# Regenerates every figure and table of the paper. The CLE training set is the
# bottleneck (about 6 to 8 minutes for 8,000 simulations) and is cached in cache/.
set -euo pipefail

# Table 1, Figure 2, Figure 3, Appendix A (and the per-bin numbers of Section 5.2)
python run_experiment.py --n_sims 8000 --n_cells 500 --n_test 400 --seed 0

# Figure 4 (routing rule, Section 5.5)
python run_budget.py --n_sims 8000 --n_cells 500 --n_test 600

# Figure 1 (CME versus CLE sweep); also redraws Figures 2 to 4 from the saved data
python paper_figures.py

# Moments-only ablation (Section 5.2)
python run_ablation.py --n_sims 8000 --n_cells 500 --n_test 400

# Robustness to random seeds (Section 5.2)
python run_seeds.py --n_sims 8000 --n_cells 500 --n_test 400 --seeds 0 1 2 3 4

# Robustness to snapshot size (Section 5.6)
python run_ncells.py --n_sims 8000 --n_cells 100 200 500 2000 --sims cme cle
