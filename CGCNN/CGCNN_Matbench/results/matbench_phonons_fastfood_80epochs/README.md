# CGCNN Matbench phonons Fastfood results (Adam, percentages)

80-epoch CGCNN Fastfood intrinsic-dimension sweep on **matbench_phonons** (last phonon DOS
peak, cm⁻¹; 1265 structures). **Corrected re-test:** `--id-dim` as **fractions** + **Adam**
(see the dielectric README for why the old bare-integer dims collapsed).

- `summary_mae_runtime.csv` — val/test MAE + runtime for every dim/seed run (36 = 12 dims × 3 seeds).
- `predictions/` — per-crystal test-set predictions, one CSV per run.
- `logs/` — training log per run.

Dims (fractions): 1.0, 0.8, 0.7, 0.65, 0.5, 0.45, 0.2, 0.1, 0.08, 0.05, 0.02, 0.01.
Seeds: matched `data_seed = model_seed ∈ {123, 456, 789}`. Model: n_conv=4, atom=56, h=64, n_h=3; batch 512.

Clean monotonic intrinsic-dimension curve: mean test MAE ≈ 79 (100 %) flat down to ~45 %,
then rises sharply below ~20 % to ≈ 292 at 1 % (a 3.7× spread).
