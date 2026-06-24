# DimeNet++ MP formation energy Fastfood results

350-epoch DimeNet++ formation energy Fastfood random-subspace sweep on the Materials Project dataset (wrapper-v3, `dimenet_run_eform_v3.py`).

- `summary_mae_runtime.csv` — validation/test MAE (eV) and runtime for every dimension/seed run (18 runs).
- `predictions/` — per-crystal test-set predictions for each run, named by dimension, model seed, split seed, and epoch count.
- `logs/` — one representative training log per intrinsic-dimension fraction (lowest model seed), 8 dims: 10%, 15%, 20%, 45%, 50%, 60%, 80%, 100%.

Intrinsic-dimension fractions swept: 10%, 15%, 20%, 45%, 50%, 60%, 80%, 100%. Seeds: model 123/456 (+789 for 45% and 60%), split 1123/1456/1789.
