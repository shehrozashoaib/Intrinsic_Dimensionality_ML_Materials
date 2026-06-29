# DimeNet++ matbench log_kvrh Fastfood results — 2 layers (2 interaction blocks)

150-epoch DimeNet++ log_kvrh Fastfood random-subspace sweep on the full matbench log_kvrh dataset, with `num_blocks=2` (wrapper-v3, `dimenet_run_kvrh_v3.py`, gradient clipping `clipnorm=1.0`).

- `summary_mae_runtime.csv` — validation/test MAE and runtime for every dimension/seed run (36 runs = 12 dims x 3 seeds).
- `predictions/` — per-crystal test-set predictions for each run, named by num_blocks, dimension, model seed, split seed, and epoch count.
- `logs/` — one representative training log per intrinsic-dimension fraction (lowest model seed), 12 dims: 100%, 80%, 70%, 65%, 50%, 45%, 20%, 10%, 8%, 5%, 2%, 1%.

Intrinsic-dimension fractions swept: 100%, 80%, 70%, 65%, 50%, 45%, 20%, 10%, 8%, 5%, 2%, 1%. Seeds: model/split = 123/1123, 456/1456, 789/1789.
