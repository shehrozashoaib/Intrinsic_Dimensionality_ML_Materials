# SGD sweeps

Companion to the main (AdamW) [`results/`](../) release: the same intrinsic-dimension
experiments re-run with **plain SGD** (momentum 0.9) as a pure optimizer swap. Config matches
the published AdamW methodology exactly (best-model eval, same schedule), differing only in the
optimizer. Learning rate is 1e-3 (the AdamW value) everywhere **except CGCNN log-K_VRH** (see note):

| sweep | LR schedule | eval |
|---|---|---|
| `eform` (formation energy, full dataset) | one-cycle | best-model (restore-best) |
| `log_kvrh_full` (full dataset, all 3 models) | ALIGNN one-cycle / CGCNN multistep / DimeNet++ constant | best-model |
| `log_kvrh_partitions` (full / quarter / tenth) | constant | best-model |
| `log_kvrh_convwidth` (conv width 1x/2x/4x) | constant | best-model |

## Layout
```
SGD_runs/
  SUMMARY.md                       # per-seed val & test MAE (all sweeps)
  eform/<Model>.csv                # CGCNN, ALIGNN, DimeNetPP
  log_kvrh_full/<Model>.csv        # CGCNN, ALIGNN, DimeNetPP (full-dataset log-K_VRH)
  log_kvrh_partitions/<Model>_<full|quarter|tenth>.csv
  log_kvrh_convwidth/<Model>_<width>.csv   # ALIGNN h64/128/256; DimeNetPP ies64/128/256/370/982
```
`DimeNetPP_full` under partitions is the ies64 (1x width) run — the full-dataset baseline.

## ⚠️ CGCNN log-K_VRH SGD
Plain SGD collapses on this task at lr 1e-3 **and** 1e-2 (mean-predictor, test ≈ 0.29 flat).
It needs **lr = 0.1** to train at all — and even then only learns at high dim (dim100 test ≈ 0.17,
~2.7× worse than AdamW's 0.06) and **collapses to ≈0.29 below ~dim 20** (those low-dim rows are the
degenerate mean-predictor). ALIGNN and DimeNet++ log-K_VRH SGD learned fully at lr 1e-3.

## Columns
`id_dim_percent, model_seed, split_seed, val_mae, test_mae, optimizer, lr_schedule, restore_best, source_dir`
— one row per run; `val_mae` is the best-val MAE, `test_mae` the test MAE. NaN and 2-epoch smoke
runs are excluded.

## Notes
- SGD tracks the AdamW intrinsic-dimension shape but sits slightly higher (worse), most at low dim.
- Quarter/tenth partitions pool the two disjoint splits (`_1`, `_2`) as extra seeds.
- Figures comparing these against AdamW are in the paper's SGD-vs-AdamW panels.
