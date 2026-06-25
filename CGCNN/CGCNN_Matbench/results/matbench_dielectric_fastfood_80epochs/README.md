# CGCNN Matbench dielectric Fastfood results (Adam, percentages)

80-epoch CGCNN Fastfood intrinsic-dimension sweep on **matbench_dielectric** (refractive
index n; 4764 structures). **Corrected re-test:** `--id-dim` passed as **fractions** (a true %
of the ~88k-param model) with the **Adam** optimizer.

> The earlier Matbench runs passed bare-integer dims, which CGCNN reads as *absolute*
> parameter counts (100…1 ≈ 0.001–0.1 % of D) so they collapsed to ~identical error. This
> version sweeps real percentages, so the intrinsic-dimension axis is meaningful.

- `summary_mae_runtime.csv` — val/test MAE + runtime for every dim/seed run (36 = 12 dims × 3 seeds).
- `predictions/` — per-crystal test-set predictions, one CSV per run.
- `logs/` — training log per run.

Dims (fractions): 1.0, 0.8, 0.7, 0.65, 0.5, 0.45, 0.2, 0.1, 0.08, 0.05, 0.02, 0.01.
Seeds: matched `data_seed = model_seed ∈ {123, 456, 789}` (3 distinct 80/10/10 splits).
Model: n_conv=4, atom_fea_len=56, h_fea_len=64, n_h=3; batch size 512.

Note: dielectric is a noisy Matbench target — mean test MAE stays roughly flat (~0.48–0.61)
across dims, with no strong intrinsic-dimension trend.
