# Validation trajectories at dim = 100% (AdamW vs SGD)

Per-epoch validation curves for the **full-subspace (100%)** formation-energy runs, for every
model whose SGD variant ran successfully. Purpose: show the validation-curve **fluctuation**
(the seed-to-seed noise ball) at 100% under each optimizer, alongside the final seed MAEs.

## Files
```
val_logs_100pct/
  SUMMARY.md                       # per-seed final val/test MAE + plateau fluctuation
  <Model>_eform_<AdamW|SGD>.csv    # long format: model, optimizer, seed, epoch, <metric>
```

## Per-epoch metric (differs by model's logging)
- **ALIGNN, CGCNN** — validation **MAE** per epoch.
- **DimeNet++** — validation **loss (MSE)** per epoch; MAE is not tracked per epoch, so its
  final val/test **MAE** (from metadata) is reported in SUMMARY but the trajectory CSV is MSE.

`epoch` is the validation-checkpoint index (ALIGNN logs more checkpoints on some runs, so its
row count can exceed the nominal 350). **Fluctuation** = std of the metric over the last 50
checkpoints (the plateau).

## Coverage note
All runs are one-cycle LR + best-model eval (matched AdamW vs SGD). **CGCNN AdamW has only 1
seed (5789)** with a recoverable per-epoch log — the other seeds' training logs were not
preserved (their metadata/MAEs survive and are in `results/` / `results/SGD_runs/`). CGCNN SGD
has 3 seeds; ALIGNN/DimeNet++ have 4 (AdamW) and 2 (SGD).

## What the fluctuations show
At 100%, SGD's plateau is *steadier* than AdamW's for ALIGNN (std ~0.0012–0.0022 vs
~0.0038–0.0055); DimeNet++ under one-cycle is very stable for both (std ≈ 0).
