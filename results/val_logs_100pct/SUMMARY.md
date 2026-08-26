# Validation trajectories at dim = 100% — AdamW vs SGD (formation energy)

Per-epoch validation curves for the full-subspace (100%) runs, one file per (model,optimizer).
**Fluctuation** = std of the validation metric over the last 50 epochs (the plateau) — it
quantifies the seed-to-seed noise ball at 100%. Per-epoch metric: ALIGNN/CGCNN = val MAE;
DimeNet++ = val loss (MSE, MAE not tracked per epoch). Final val/test MAE are from metadata.

## CGCNN · AdamW  (metric per epoch: val_mae)

| seed | final val MAE | final test MAE | plateau fluct. (last-50 std) | plateau range |
|-----:|:--------------|:---------------|:-----------------------------|:--------------|
| 5789 | 0.0530 | 0.0537 | 0.00121 | 0.0530–0.0590 |
| **mean** | **0.0530** (±0.0000 across seeds) | | | |

## CGCNN · SGD  (metric per epoch: val_mae)

| seed | final val MAE | final test MAE | plateau fluct. (last-50 std) | plateau range |
|-----:|:--------------|:---------------|:-----------------------------|:--------------|
| 1653 | 0.0800 | 0.0816 | 0.00438 | 0.0800–0.1050 |
| 253 | 0.0780 | 0.0817 | 0.00447 | 0.0780–0.0980 |
| 768 | 0.0840 | 0.0850 | 0.00865 | 0.0840–0.1420 |
| **mean** | **0.0807** (±0.0025 across seeds) | | | |

## ALIGNN · AdamW  (metric per epoch: val_mae)

| seed | final val MAE | final test MAE | plateau fluct. (last-50 std) | plateau range |
|-----:|:--------------|:---------------|:-----------------------------|:--------------|
| 123 | 0.0481 | 0.0535 | 0.00555 | 0.0481–0.0730 |
| 321 | 0.0469 | 0.0477 | 0.00434 | 0.0469–0.0697 |
| 456 | 0.0476 | 0.0557 | 0.00420 | 0.0476–0.0645 |
| 654 | 0.0488 | 0.0467 | 0.00382 | 0.0488–0.0714 |
| **mean** | **0.0478** (±0.0007 across seeds) | | | |

## ALIGNN · SGD  (metric per epoch: val_mae)

| seed | final val MAE | final test MAE | plateau fluct. (last-50 std) | plateau range |
|-----:|:--------------|:---------------|:-----------------------------|:--------------|
| 123 | 0.0520 | 0.0532 | 0.00117 | 0.0520–0.0575 |
| 456 | 0.0516 | 0.0537 | 0.00220 | 0.0516–0.0630 |
| **mean** | **0.0518** (±0.0002 across seeds) | | | |

## DimeNet++ · AdamW  (metric per epoch: val_loss(MSE))

| seed | final val MAE | final test MAE | plateau fluct. (last-50 std) | plateau range |
|-----:|:--------------|:---------------|:-----------------------------|:--------------|
| 123 | 0.0594 | 0.0596 | 0.00001 | 0.0084–0.0085 |
| 234 | 0.0595 | 0.0614 | 0.00001 | 0.0088–0.0088 |
| 456 | 0.0576 | 0.0593 | 0.00001 | 0.0075–0.0075 |
| 789 | 0.0619 | 0.0614 | 0.00002 | 0.0103–0.0104 |
| **mean** | **0.0596** (±0.0015 across seeds) | | | |

## DimeNet++ · SGD  (metric per epoch: val_loss(MSE))

| seed | final val MAE | final test MAE | plateau fluct. (last-50 std) | plateau range |
|-----:|:--------------|:---------------|:-----------------------------|:--------------|
| 123 | 0.0675 | 0.0686 | 0.00001 | 0.0102–0.0103 |
| 456 | 0.0698 | 0.0725 | 0.00002 | 0.0106–0.0107 |
| **mean** | **0.0687** (±0.0011 across seeds) | | | |
