# ALIGNN MP formation energy Fastfood results

This folder contains the 350-epoch ALIGNN formation-energy Fastfood sweep on the MP dataset. `summary_mae_runtime.csv` reports validation/test MAE and runtime for each dimension/seed (18 runs). The `predictions/` directory contains per-crystal test-set predictions for each run, named by dimension, model seed, split seed, and epoch count; `logs/` holds one representative training log per intrinsic-dimension fraction (lowest model seed).

Intrinsic-dimension fractions swept: 100, 80, 60, 50, 45, 20, 15, 10 %. Seeds: model 123/456 (+789 for 45 % and 60 %), split 1123/1456/1789.
