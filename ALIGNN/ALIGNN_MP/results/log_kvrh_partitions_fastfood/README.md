# ALIGNN matbench log_kvrh — Fastfood intrinsic-dimension sweep on half-dataset subsets

ALIGNN intrinsic-dimensionality (Fastfood) sweep for the Matbench **log_kvrh** task on two
**disjoint 50% halves** of the dataset, to probe how the intrinsic-dimension / test-MAE curve
behaves at half the training-set size.

## Partitions (2, disjoint halves)
- `kvrh_half_1`, `kvrh_half_2` — two disjoint 50% subsets of the 10,987-structure log_kvrh set
  (built by `alignn/build_log_kvrh_idprop.py`, shuffle seed 0: 5,493 / 5,494 records). ALIGNN
  then does its own internal 80/10/10 split within each half.

## Sweep
- Intrinsic-dimension fractions: **100, 80, 70, 65, 50, 45, 20, 10, 8, 5, 2, 1 %** of model
  parameters (`id_dim` is a float fraction; optimizer = Adam-family, 120 epochs).
- Seeds: **123, 456** per (half × dim).
- Total: 12 dims × 2 halves × 2 seeds = **48 runs**, all `success`.

## Layout
- `summary_mae_runtime.csv` — one row per run: validation/test MAE, runtime, timestamps, and
  the relative path to that run's test-set predictions.
- `<partition>/predictions/` — per-crystal test-set predictions for every run (both seeds).
- `<partition>/logs/` — one representative training log per dimension (lowest seed, 123).

Raw run directories (`results_alignn_log_kvrh_partitions_fastfood/`) are git-ignored; only this
curated tree is committed. Data source: matbench log_kvrh structures recovered from the CGCNN
raw split pickle and converted to ALIGNN JARVIS `id_prop.json`.
