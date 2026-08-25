# SGD sweeps

Companion to the main (AdamW) [`results/`](../) release: the same intrinsic-dimension
experiments re-run with **plain SGD** (momentum 0.9, learning rate 1e-3) as a pure
optimizer swap. Config matches the published AdamW methodology exactly, differing only
in the optimizer:

| sweep | LR schedule | eval |
|---|---|---|
| `eform` (formation energy, full dataset) | one-cycle | best-model (restore-best) |
| `log_kvrh_partitions` (full / quarter / tenth) | constant | best-model |
| `log_kvrh_convwidth` (conv width 1x/2x/4x) | constant | best-model |

## Layout
```
SGD_runs/
  SUMMARY.md                       # per-dim mean ± std of val & test MAE (all sweeps)
  eform/<Model>.csv                # CGCNN, ALIGNN, DimeNetPP
  log_kvrh_partitions/<Model>_<full|quarter|tenth>.csv
  log_kvrh_convwidth/<Model>_<width>.csv   # ALIGNN h64/128/256; DimeNetPP ies64/128/256/370/982
```
`DimeNetPP_full` under partitions is the ies64 (1x width) run — the full-dataset baseline.

## Columns
`id_dim_percent, model_seed, split_seed, val_mae, test_mae, optimizer, lr_schedule, restore_best, source_dir`
— one row per run; `val_mae` is the best-val MAE, `test_mae` the test MAE. NaN and 2-epoch smoke
runs are excluded.

## Notes
- SGD tracks the AdamW intrinsic-dimension shape but sits slightly higher (worse), most at low dim.
- Quarter/tenth partitions pool the two disjoint splits (`_1`, `_2`) as extra seeds.
- Figures comparing these against AdamW are in the paper's SGD-vs-AdamW panels.
