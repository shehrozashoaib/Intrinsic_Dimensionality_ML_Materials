# ALIGNN MP band gap Fastfood results

This folder contains the completed 350-epoch ALIGNN band-gap Fastfood intrinsic-dimensionality sweep on the MP band-gap dataset (now full: 18 runs).

`summary_mae_runtime.csv` reports validation/test MAE and runtime for each dimension/seed. The `predictions/` directory contains per-crystal test-set predictions for each run, named by dimension, data seed, model seed, split seed, and epoch count. The `logs/`, `metadata/`, and `histories/` folders contain supporting run artifacts.

Intrinsic-dimension fractions swept: 100, 80, 60, 50, 45, 20, 10, 5 %. Seeds: model 123/456 (+789 for 45 % and 60 %), split 123/456/789. Runs: 18 (the seed-123 set plus the seed-456/789 runs added once the sweep finished).
