# No-wrapper baselines

Control runs that train the **full parameter space directly** — no random-subspace
(Fastfood) wrapper at all. They anchor the intrinsic-dimension curves: a wrapped run at
`id_dim = 100%` should approach the no-wrapper MAE. Scattered across the DimeNet++ result
tree (dedicated no-wrapper dir, the optimizer-control arm, and the wrapper-construction
study's `base` arm); identified by `wrapper == False` or `train_mode == "base"`.

All runs are **DimeNet++, full dataset, best-model eval**. No CGCNN or ALIGNN no-wrapper
runs exist. Two NaN/failed runs (band-gap SGD seed 789, eform SGD seed 456) are excluded.

## Layout
```
no_wrapper_baselines/
  SUMMARY.md                # per-seed val+test MAE and mean±std, per (model,task,optimizer)
  all_no_wrapper_runs.csv   # every run, one row
  formation_energy.csv  band_gap.csv  log_kvrh.csv   # per-task
```

## Columns
`model, task, dataset_size, optimizer, lr_schedule, restore_best, id_dim_percent,
model_seed, split_seed, val_mae, test_mae, source_dir`

## Configs present
| task | optimizer | LR schedule |
|---|---|---|
| formation energy | AdamW, SGD | constant |
| band gap | AdamW, SGD | constant |
| log-K_VRH | AdamW | one-cycle |

`optimizer` is taken from metadata where recorded; the no-wrapper eform and log-K_VRH runs
did not record it and used the runner default (AdamW), confirmed from their launch scripts.
