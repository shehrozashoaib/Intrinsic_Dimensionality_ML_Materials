# CGCNN Matbench Intrinsic-Dimensionality Experiments

This directory contains the CGCNN part of the intrinsic-dimensionality experiments for materials-property prediction. The code trains a Crystal Graph Convolutional Neural Network (CGCNN) on three Matbench regression tasks while constraining optimization to a low-dimensional random subspace with a Fastfood projection.

> **Corrected re-test (current results).** The intrinsic dimension is now swept as a
> **percentage of the model's parameters** — `--id-dim` is passed as a **fraction**
> (`1.0`, `0.8`, …, `0.01`), since CGCNN's `subspace.py` treats an *integer* `id_dim` as an
> *absolute* parameter count. The earlier scripts passed bare integers (100…1), which were
> 0.001–0.1 % of the ~88k-param model and therefore collapsed to ~identical error. The
> optimizer is **Adam** (SGD freezes at low intrinsic dimension). Curated per-task results
> with logs, predictions and summaries live under `results/matbench_<task>_fastfood_<epochs>epochs/`.

## Goal

The goal is to study how many trainable intrinsic parameters are needed for a materials graph neural network to recover useful performance. The base model is a CGCNN with about 87.6k parameters. During intrinsic-dimension runs, the full parameter vector is frozen at initialization and training only updates a low-dimensional vector `z`; the model weights used in each forward pass are reconstructed as:

```text
theta = theta0 + P z
```

where `P` is either a dense Gaussian projection or an implicit Fastfood projection. The production sweeps here use Fastfood because it avoids storing a large dense `D x d` matrix.

## Contents

```text
CGCNN_Matbench/
  main.py                               # CGCNN training entrypoint
  prepare_matbench_splits.py             # create deterministic raw Matbench 80/10/10 splits
  materialize_cgcnn_tensors.py           # precompute CGCNN graph tensors once
  predict.py                             # original CGCNN prediction helper
  requirements.txt
  cgcnn/
    data.py                              # CIFData, TensorCIFData, collate/loaders
    model.py                             # CGCNN model
    subspace.py                          # dense/Fastfood intrinsic-dimension wrapper
  root_dir/
    atom_init.json                       # atom feature vectors needed by CIFData
  scripts/
    run_cgcnn_matbench_fastfood_sweep.sh    # CURRENT runner: fractions + Adam, all 12 dims x 3 seeds
    submit_matbench_task.slurm              # one A100 job = one full task sweep
    build_matbench_all.slurm                # build tensor caches for 3 tasks x 3 seeds
    build_matbench_tensors.slurm            # build a single (tasks, seed) cache
    confirm_dielectric_optim.slurm          # 100% vs 1% sanity check
    run_cgcnn_fastfood_phonons_dataseed123_modelseeds.sh    # (superseded: bare-int dims)
    run_cgcnn_fastfood_dielectric_dataseed123_modelseeds.sh # (superseded)
    run_cgcnn_fastfood_log_kvrh_dataseed123_modelseeds.sh   # (superseded)
  results/
    matbench_dielectric_fastfood_80epochs/  # README + summary_mae_runtime.csv + logs/ + predictions/
    matbench_phonons_fastfood_80epochs/
    matbench_log_kvrh_fastfood_120epochs/
    summary_csvs/
      matbench_phonons_fastfood_summary.csv
      matbench_dielectric_fastfood_summary.csv
      matbench_log_kvrh_fastfood_summary.csv
```

Curated results (per-run training logs, prediction CSVs, and summary CSVs) are tracked under
`results/matbench_<task>_fastfood_<epochs>epochs/`. Large cached tensors, raw Matbench pickle
files, model checkpoints, and the raw `results_matbench_*_fastfood/` run directories are not tracked.

## Datasets

The experiments target these Matbench v0.1 tasks:

- `matbench_phonons`
- `matbench_dielectric`
- `matbench_log_kvrh`

For these local experiments, results are not submitted to the Matbench leaderboard. Instead, fold 0 is used only as a convenient way to load all labeled structures. The script combines Matbench's fold-0 train/validation labels and test labels, shuffles them with `data_seed=123`, then creates a local 80/10/10 train/validation/test split.

Generated split sizes:

| Task | Train | Val | Test |
|---|---:|---:|---:|
| `matbench_phonons` | 1012 | 126 | 127 |
| `matbench_dielectric` | 3811 | 476 | 477 |
| `matbench_log_kvrh` | 8789 | 1098 | 1100 |

## Model

The sweep scripts use this CGCNN configuration:

```text
n_conv=4
atom_fea_len=56
h_fea_len=64
n_h=3
batch_size=512
```

For the cached Matbench tensor feature dimensions used here, this model has:

```text
87,577 total parameters
```

In a Fastfood run, trainable parameters equal the requested intrinsic dimension `d`.

## Intrinsic-Dimension Sweep

The Fastfood sweeps use (intrinsic dim as a **fraction** of the model's params):

```text
id_dim (fraction) = 1.0, 0.8, 0.7, 0.65, 0.5, 0.45, 0.2, 0.1, 0.08, 0.05, 0.02, 0.01
                    (i.e. 100%, 80%, ..., 1%)
optimizer  = Adam
seeds      = matched data_seed = model_seed in {123, 456, 789}  (3 distinct 80/10/10 splits)
```

Epochs:

```text
matbench_phonons:     80
matbench_dielectric:  80
matbench_log_kvrh:   120
```

Each shell script appends one row per run to a task-level summary CSV and removes checkpoint files after the run to avoid filling the disk. It keeps logs, metadata, and prediction CSVs in the runtime output directory, but those generated files are ignored by git.

The committed summary CSVs contain:

```text
task, wrapper, id_dim, data_seed, model_seed, epochs, batch_size,
n_conv, atom_fea_len, h_fea_len, n_h, output_dir, predictions_csv,
best_val_mae, test_mae, status, exit_code, start_time, end_time
```

## Environment Setup

The workstation used for these experiments had a conda environment named `py312`.

One way to create a similar environment:

```bash
bash requirements.sh
```


## Reproduce From Scratch

Start from this directory:

```bash
cd CGCNN/CGCNN_Matbench
conda activate py312
```

### 1. Create Raw Matbench Splits

This downloads/loads the three Matbench datasets, combines labeled fold-0 train and test data, shuffles, and writes local 80/10/10 raw split files.

```bash
python prepare_matbench_splits.py \
  --output-dir cached_matbench/raw \
  --seed 123 \
  --fold 0
```

### 2. Precompute CGCNN Tensors

This runs the graph construction in `cgcnn/data.py` once and saves reusable tensor datasets.

```bash
python materialize_cgcnn_tensors.py \
  --raw-dir cached_matbench/raw \
  --output-dir cached_matbench/tensors \
  --root-dir root_dir \
  --seed 123
```

This creates:

```text
cached_matbench/tensors/<task>/seed123/train.pt
cached_matbench/tensors/<task>/seed123/val.pt
cached_matbench/tensors/<task>/seed123/test.pt
```

These `.pt` files are large generated artifacts and are ignored by git.

### 3. Run One Cached Training Job

Example: one Fastfood run on phonons with intrinsic dimension 20.

```bash
python main.py \
  --cached-data-dir cached_matbench/tensors \
  --matbench-task matbench_phonons \
  --output-dir results_matbench_phonons_fastfood/debug_dim20_seed123 \
  --epochs 80 \
  --batch-size 512 \
  --n-conv 4 \
  --atom-fea-len 56 \
  --h-fea-len 64 \
  --n-h 3 \
  --subspace-method fastfood \
  --id-dim 20 \
  --data-seed 123 \
  --random-seed 123 \
  --print-freq 1000
```

`--data-seed` selects which cached tensor split to load. `--random-seed` controls model initialization and the random subspace projection. Keeping these separate allows the same dataset split to be evaluated under several model/subspace seeds.

### 4. Run Full Sweeps (current)

Build the caches for all 3 tasks × 3 seeds, then submit one A100 job per task (each runs all
12 dims × 3 seeds with fractions + Adam):

```bash
sbatch scripts/build_matbench_all.slurm                                   # caches: 3 tasks x 3 seeds
for t in dielectric phonons log_kvrh; do
  sbatch --export=ALL,CGCNN_MB_TASK="$t" scripts/submit_matbench_task.slurm
done
```

`scripts/run_cgcnn_matbench_fastfood_sweep.sh` (invoked by the job) for each run:

1. Loads cached tensors from `cached_matbench/tensors`.
2. Runs all intrinsic-dim **fractions** for matched `data_seed = model_seed ∈ {123, 456, 789}`, with **Adam**.
3. Skips a run whose `metadata.json` already has `status=success`.
4. Appends one row to a flock-guarded per-task summary CSV under `results_matbench_<task>_fastfood/`.
5. Renames the test prediction CSV and deletes `.pth.tar` checkpoints after each run.

The three superseded `run_cgcnn_fastfood_<task>_dataseed123_modelseeds.sh` scripts are kept for
reference but pass bare-integer dims (absolute counts) and predate the Adam default — do not use
them for the percentage sweep.

## Notes

- The cached tensor path is the recommended workflow for sweeps. It avoids repeatedly running expensive pymatgen neighbor searches.
- Some structures may warn that fewer than the requested number of neighbors were found within the default radius. This follows the original CGCNN behavior and pads missing neighbors.
- The repository copy intentionally excludes generated tensors and checkpoints. Regenerate them with the preprocessing scripts when needed.

