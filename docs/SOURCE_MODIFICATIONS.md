# Source-code modifications

This project wraps three published GNN codebases (CGCNN, ALIGNN, DimeNet++) with a
random-subspace ("intrinsic dimension") training wrapper, and adds the optimizer /
learning-rate-schedule / best-model-checkpoint machinery needed for the study. This file lists
every change made to the upstream model code so results are reproducible and the diffs are clear.

The **random-subspace wrapper** trains only a `d`-dimensional vector `z`; the full parameter
vector is `theta = theta_0 + P z`, with `theta_0` frozen at initialization and `P` a fixed random
projection. `P` is the **non-orthonormal Fastfood** transform `P = (1/s) H G Pi H B` by default
(`B` random +/-1 diagonal, `Pi` random permutation, `G` Gaussian diagonal, `H` Walsh-Hadamard),
applied implicitly via two Fast Walsh-Hadamard transforms. Dense and orthonormalized variants are
also implemented (used only in the projection-construction control).

---

## CGCNN  (`CGCNN/CGCNN_{MP,Matbench}/`)

**`cgcnn/subspace.py`** *(new file)* — the random-subspace wrapper.
- `RandomSubspaceWrapper` supporting `method={dense, fastfood}`; only `z` is trainable.
- Fastfood applied as two FWHTs with `O(D)` buffers (no dense `P` materialized).

**`main.py`** — training driver.
- `--subspace-method {none,dense,fastfood}` and `--id-dim` (fraction of `D` to train).
- `--lr-schedule {multistep,onecycle,cosine}` via a `build_scheduler()` helper. **Default
  `multistep`** (upstream behaviour: x0.1 at `--lr-milestones`, default `[100]`), so previously
  produced results are unaffected. `--pct-start` exposes the OneCycle warm-up fraction.
- `--optim {SGD,Adam}` (Adam default).
- **Best-model test evaluation** is upstream CGCNN behaviour: `model_best.pth.tar` is saved on the
  best validation metric and reloaded before the final test pass. No change needed.

> **Task-specific setting (embedded in the bash scripts):** CGCNN on **Matbench dielectric** under
> OneCycle needs `--lr 0.1` (100x the value used elsewhere); at `1e-3` it collapses to a constant
> predictor. See `CGCNN/CGCNN_Matbench/scripts/run_cgcnn_dielectric_onecycle_sweep.sh`.

---

## ALIGNN  (`ALIGNN/ALIGNN_MP/alignn/`)

**`train_alignn.py`** — training entry point.
- Random-subspace wrapper (`--method {dense,fastfood}`, `--id_dim`) around the ALIGNN model.
- Classification fix: sets `config.model.classification = True` and `energy_mult_natoms = False`
  when a `classification_threshold` is given (otherwise the atomwise head crashes with an
  NLLLoss / natoms-broadcast shape error).

**`train.py`** — training loop / evaluation.
- **Restore-best-checkpoint fix (opt-in via `ALIGNN_RESTORE_BEST=1`).** Upstream sets
  `best_model = net`, which is a Python *alias*: `net` keeps training, so the reported test metric
  is actually the **final-epoch (overfit)** model and `best_model.pt` is never read back. The added
  block reloads `best_model.pt` before the test pass. Left opt-in so older runs stay reproducible;
  the bash scripts export `ALIGNN_RESTORE_BEST=1`.
- Classification prediction writer emits the class-1 probability (`exp(logsoftmax)[:,1]`) so
  `Test ROCAUC` is a true probabilistic AUC, and also prints `Test ACC`.

> ALIGNN's LR schedule is set in its JSON configs (`configs/config_*.json`): `scheduler: onecycle`,
> `learning_rate: 1e-3`.

---

## DimeNet++  (`DimeNetPP/DimeNetPP_{MP,Matbench}/dimenetpp_code_only/`)

**`wrapper_tensorflow_v3.py`** *(new file)* — the TensorFlow random-subspace wrapper.
- `SubspaceProjectedGradTFV3`: `theta = theta_0 + P z`, only `z` trainable.
- Fastfood via static Walsh-Hadamard transforms; `method`, `orthonormal`, `full_rotation`,
  `dense_block_cols` options (dense/orthonormal used only in the projection control).

**`dimenet_run_{eform,bandgap,kvrh,dielectric,is_metal,phonons}_v3.py`** — per-task runners.
- `--train_mode {base,wrapped}` (`base` trains all parameters directly, no projection — the
  no-wrapper control).
- `--optimizer {adamw,sgd}`, `--momentum`, `--lr` (max_lr for schedules).
- `--lr_schedule {none,onecycle,cosine}` + `--pct_start`. **Default `none` (constant lr)** — the
  main-experiment setting; OneCycle is used for the LR-schedule study.
- `--restore_best` (ModelCheckpoint on best `val_loss` + explicit `load_weights` before predict)
  and `--patience` (EarlyStopping). Without `--restore_best` the final-epoch model is tested.
- `--num_blocks` (interaction blocks) and `--int_emb_size` (interaction bottleneck).
- Dielectric runner prints `[pred_spread]` (std of test-set predictions) so a collapsed
  constant-predictor is flagged automatically.

> DimeNet++ runs on TensorFlow (V100/A100); CGCNN and ALIGNN on PyTorch (RTX-2080-Ti/A100).
> torch 2.11+cu128 has an SM-arch floor of sm_75, so V100 (sm_70) cannot run CGCNN/ALIGNN.
