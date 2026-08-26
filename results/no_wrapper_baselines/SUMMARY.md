# No-wrapper baselines — MAE summary

Control runs trained on the **full parameter space** (no random-subspace / Fastfood wrapper),
so `id_dim = 100%` here means *all* weights are trained directly. All are **full-dataset**,
**best-model eval**. Each block is one (model, task, optimizer) group; `val` = best-val MAE,
`test` = test MAE. AdamW runs where the optimizer field was unrecorded are the runner default.

## DimeNet++ · bandgap · adamw (schedule=none, no wrapper, full dataset)

| seed | val MAE | test MAE |
|-----:|:--------|:---------|
| 123 | 0.3162 | 0.3325 |
| 234 | 0.3467 | 0.3438 |
| 456 | 0.3138 | 0.3149 |
| 789 | 0.3276 | 0.3376 |
| **mean±std** | **0.3261 ± 0.0130** | **0.3322 ± 0.0107** |

## DimeNet++ · bandgap · sgd (schedule=none, no wrapper, full dataset)

| seed | val MAE | test MAE |
|-----:|:--------|:---------|
| 123 | 0.4394 | 0.4567 |
| 234 | 0.4041 | 0.3975 |
| 456 | 0.3848 | 0.3906 |
| **mean±std** | **0.4094 ± 0.0226** | **0.4149 ± 0.0297** |

## DimeNet++ · eform · adamw (schedule=none, no wrapper, full dataset)

| seed | val MAE | test MAE |
|-----:|:--------|:---------|
| 123 | 0.0458 | 0.0459 |
| 234 | 0.0443 | 0.0459 |
| 456 | 0.0450 | 0.0468 |
| 789 | 0.0463 | 0.0451 |
| **mean±std** | **0.0454 ± 0.0008** | **0.0459 ± 0.0006** |

## DimeNet++ · eform · sgd (schedule=none, no wrapper, full dataset)

| seed | val MAE | test MAE |
|-----:|:--------|:---------|
| 123 | 0.0694 | 0.0708 |
| 234 | 0.0651 | 0.0657 |
| 789 | 0.0633 | 0.0635 |
| **mean±std** | **0.0660 ± 0.0026** | **0.0667 ± 0.0031** |

## DimeNet++ · log_kvrh · adamw (schedule=onecycle, no wrapper, full dataset)

| seed | val MAE | test MAE |
|-----:|:--------|:---------|
| 123 | 0.0693 | 0.0730 |
| 234 | 0.0714 | 0.0695 |
| 456 | 0.0775 | 0.0753 |
| 789 | 0.0683 | 0.0694 |
| **mean±std** | **0.0716 ± 0.0036** | **0.0718 ± 0.0025** |
