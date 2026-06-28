# ALIGNN MP formation energy — Fastfood intrinsic-dimension sweep on data subsets

This study repeats the ALIGNN formation-energy Fastfood sweep on **disjoint random data
subsets** of the Materials Project dataset, to see how the intrinsic-dimension / test-MAE
curve shifts with training-set size.

## Partitions (13, disjoint)
- `eform_p10_1`, `eform_p10_2`, `eform_p10_3` — three **10 %** subsets
- `eform_p01_1` … `eform_p01_10` — ten **1 %** subsets

## Sweep
- Intrinsic-dimension fractions: **100, 80, 60, 50, 45, 20, 15, 10 %** of model parameters
  (`id_dim` is a float fraction; optimizer = Adam, 350 epochs).
- Seeds: **123, 456** per (partition × dim).
- Total: 8 dims × 13 partitions × 2 seeds = **208 runs**, all `success`.

## Layout
- `summary_mae_runtime.csv` — one row per run: validation/test MAE, runtime, timestamps,
  and the relative path to that run's test-set predictions.
- `<partition>/predictions/` — per-crystal test-set predictions for every run (both seeds).
- `<partition>/logs/` — one representative training log per dimension (lowest seed, 123).

Raw run directories (`results_alignn_eform_partitions_fastfood/`) are git-ignored; only this
curated tree is committed.
