# RepresentationOptimizationDichotomy
A Representation-Optimization Dichotomy for Lie-Algebraic Policy Optimization
# Lie-Algebraic Policy Optimization: Experimental Validation

Code for the experiments in:

"A Representation-Optimization Dichotomy for Lie-Algebraic Policy Optimization"
Sooraj K.C. and Vivek Mishra, Alliance University
Submitted to SIAM Journal on Mathematics of Data Science (SIMODS)

---

## What this code does

This script reproduces all tables, figures, and numerical claims in Section 9
(Numerical Illustrations) of the paper. It runs policy gradient experiments on
SO(3) and SE(3) environments and measures Fisher-metric alignment, convergence
rates, computational efficiency, and method comparisons.

The experiments correspond to paper sections and outputs as follows:

| Exp | What it tests                                  | Section | Tables / Figures         |
|-----|------------------------------------------------|---------|--------------------------|
| 1   | Fisher-metric alignment and isotropy tracking  | 9.2     | Table 4, Fig 1           |
| 2   | Computational efficiency timing                | 9.4     | Table 5                  |
| 3   | Anisotropy ablation with controlled noise      | 9.2     | Fig 2 left               |
| 4   | Sample efficiency: Lie policy vs ambient       | 9.2     | Fig 2 right              |
| 5   | Convergence rate (deterministic)               | 9.3     | Fig 3 left               |
| 5b  | Alignment vs Fisher isotropy deviation         | 9.2     | (supporting data)        |
| 6   | Scalability as J increases                     | 9.4     | Table 6                  |
| 7   | SE(3) non-compact algebra comparison           | 9.6     | Table 7, Fig 4           |
| 8   | Symmetry-violation robustness                  | 9.5     | Supplement S11           |
| 9   | Method comparison: LPG vs ambient PG vs NatGrad| 9.7     | Table 8, Fig 5           |
| 10  | Convergence rate (stochastic)                  | 9.3     | Fig 3 right              |

---

## Setup

Python 3.8 or later is recommended.

Install dependencies:

    pip install -r requirements.txt

---

## Running experiments

Run all experiments:

    python experiments_unified_simods.py

Run selected experiments (for example, 1, 3, and 7 only):

    python experiments_unified_simods.py 1 3 7

Output figures are saved to the figs_simods/ folder. Numerical results are
printed to stdout.

---

## Output files

All figures are saved as PNG files at 300 DPI in figs_simods/:

  exp1_fisher_alignment_hist.png        -- alignment distribution histogram (Fig 1 left)
  isotropy_tracking.png                 -- eps_F, kappa, and alignment over training (Fig 1 right)
  exp1_alignment_vs_eps.png             -- alignment vs Fisher isotropy deviation
  exp5_eps_vs_alignment.png             -- isotropy deviation vs alignment sweep
  controlled_anisotropy.png             -- anisotropy ablation results (Fig 2 left)
  exp3_learning_curves.png              -- Lie vs ambient learning curves (Fig 2 right)
  exp3_bar_stats.png                    -- bar chart of AUC statistics
  exp4_convergence_loglog.png           -- deterministic convergence log-log plot (Fig 3 left)
  exp4_convergence_loglog_stochastic.png-- stochastic convergence log-log plot (Fig 3 right)
  exp2_timing.png                       -- Fisher inversion vs projection timing
  exp2_speedup.png                      -- speedup ratio across joint counts
  exp_se3_divergence.png                -- SE(3) divergence with/without radius projection (Fig 4)
  exp9_method_comparison.png            -- LPG vs ambient PG vs natural gradient (Fig 5 left)
  exp9_wallclock_comparison.png         -- same comparison on wall-clock time axis (Fig 5 right)

---

## Code structure

Everything is in a single file. The main components are:

  SO3, SE3                              -- Lie group utilities (hat, vee, exp, log, project)
  SO3JEnv, SE3Env                       -- RL environments for rotation and pose tracking
  LieLinearPolicy, ControlledNoisePolicy,
  DiagFeaturePolicy, AmbientLinearPolicy -- policy classes
  train_reinforce, estimate_fisher,
  compute_fisher_metrics                -- training and Fisher estimation utilities
  exp1 through exp9, exp5b              -- individual experiment functions
  run_all                               -- entry point that runs selected or all experiments

---

## Reproducibility

All experiments use fixed random seeds. The global seed is set to 42 at
startup, and each experiment additionally seeds NumPy per trial. Results may
vary slightly across hardware due to floating-point differences in LAPACK
routines.

Expected runtime on a standard laptop CPU:

  Experiments 1, 2, 3, 6: a few minutes each
  Experiments 4, 5, 7, 8, 9, 10: 10 to 40 minutes each depending on hardware
  Full run (all experiments): approximately 2 to 4 hours

---

## Citation

If you use this code, please cite the paper once it is published. For now,
you can reference the SIMODS submission.

---

## Contact

Sooraj K.C. -- ksoorajPHD23@sam.alliance.edu.in
Department of Pure and Applied Mathematics, Alliance University, Bengaluru, India
