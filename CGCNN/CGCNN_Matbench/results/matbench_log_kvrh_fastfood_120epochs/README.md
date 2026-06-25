# CGCNN Matbench log_kvrh Fastfood results (Adam, percentages)

120-epoch CGCNN Fastfood intrinsic-dimension sweep on **matbench_log_kvrh** (log₁₀ of bulk
modulus K_VRH, log₁₀ GPa; 10987 structures). **Corrected re-test:** `--id-dim` as **fractions**
+ **Adam** (see the dielectric README for why the old bare-integer dims collapsed).

- `summary_mae_runtime.csv` — val/test MAE + runtime for every dim/seed run (36 = 12 dims × 3 seeds).
- `predictions/` — per-crystal test-set predictions, one CSV per run.
- `logs/` — training log per run.

Dims (fractions): 1.0, 0.8, 0.7, 0.65, 0.5, 0.45, 0.2, 0.1, 0.08, 0.05, 0.02, 0.01.
Seeds: matched `data_seed = model_seed ∈ {123, 456, 789}`. Model: n_conv=4, atom=56, h=64, n_h=3; batch 512.

Monotonic intrinsic-dimension curve: mean test MAE ≈ 0.070 (100 %) flat to ~45 %, rising to
≈ 0.106 at 1 %.
