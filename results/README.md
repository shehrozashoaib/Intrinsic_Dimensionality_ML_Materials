# Reconciled results — val + test metrics behind the figures

Every value plotted in the intrinsic-dimension figures is a **best-val** MAE/AUC (band gap and
some Matbench panels plot the paired **test** value). This folder recovers, for **each plotted
point**, the underlying run's *both* metrics — validation and test — by matching the plotted
number back to its run `metadata.json`.

## Layout
```
results/<Model>/<task>/<partition>.csv
```
`<Model>` = ALIGNN | DimeNetPP | CGCNN  ·  `<partition>` = full | quarter | one-tenth
(quarter / one-tenth exist only for the dataset-size study: formation energy, band gap, log-K_VRH).

## Columns
| column | meaning |
|---|---|
| `id_dim_percent` | subspace dimension d as % of D |
| `seed` | model/data seed of the matched run |
| `val` | validation metric (best_val_mae / val_mae / best_val_auc) |
| `test` | test metric (test_mae / test_auc) |
| `metric` | `mae` or `auc` |
| `plotted_value` | the number as it appears in the figure |
| `matched_on` | whether the plotted value equalled the run's `val` or `test` |
| `confidence` | `exact` (<0.5% rel.), `close` (<5%), or `unmatched` |
| `abs_residual` | |plotted − matched metric| |
| `source_dir` | run directory the metrics came from |

## One row per plotted point → dropped seeds stay dropped
Rows mirror the figures exactly: if a seed was dropped from a curve, it is absent here too.

## Coverage
1866 exact + 42 close + 113 unmatched  (of 2021 plotted points).
Per-series counts are in `coverage_manifest.csv`. Unmatched points are genuinely unrecoverable
from the current run tree, in three groups:
- **ALIGNN band-gap / formation-energy *quarter*** — no 25%-subset runs exist in the tree
  (only full, 10% and 1% partitions were kept), so quarter rows are empty.
- **ALIGNN band-gap *full*, older seeds** — a few early seeds predate the retained runs.
- **5% formation-energy placeholders** (DimeNet++, CGCNN) — those figure points were copies of the
  10% values, so they have no 5% run to match.
Everything else reconciles to the exact run (residual ≈ 0).
