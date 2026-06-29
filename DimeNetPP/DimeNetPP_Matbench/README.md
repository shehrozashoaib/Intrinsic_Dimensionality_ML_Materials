# DimeNet++ Matbench Intrinsic-Dimensionality Experiments

DimeNet++ Matbench part of the intrinsic-dimensionality study for materials-property
prediction. Trains a DimeNet++ crystal model with a **Fastfood random-subspace wrapper**
(only the low-dimensional `z` is trained, projected into the full weight space) on the
**Matbench `log_kvrh`** task (log10 of the Voigt-Reuss-Hill bulk modulus), sweeping the
intrinsic dimension as a percentage of model parameters.

This study additionally compares **model depth**: the same sweep is run at **2 interaction
blocks (layers)** and **4 interaction blocks**, 3 seeds each.

Only code, shell/Slurm scripts, small config, and curated results are tracked. Large items
(the conda env, cached tensors, raw per-run outputs, datasets) are git-ignored. The conda
environment is shared with `DimeNetPP_MP/envs/pydimnet` (see `scripts/dimenet_env.sh`).

## Directory Layout

```text
DimeNetPP_Matbench/
  README.md
  requirements_dimenetpp.txt
  environment_dimenetpp.yml
  dimenetpp_code_only/
    dimenet_run_kvrh_v3.py        # wrapper-v3 training runner (log_kvrh); --num_blocks, --clipnorm
    wrapper_tensorflow_v3.py      # SubspaceProjectedGradTFV3 (Fastfood/dense wrapper)
    data_saving_parallel.py       # build-once + reshuffle, 16-worker parallel tensor cache
    torch_orthonormal_q_v3.py     # external PyTorch QR for dense-orthonormal runs (unused here)
  scripts/
    dimenet_env.sh                # env activation + LD_LIBRARY_PATH (points at shared MP env)
    build_tensors_parallel.slurm  # build log_kvrh tensor caches (3 split seeds)
    run_dimenetpp_fastfood_kvrh_sweep.sh  # 12-dim x 3-seed sweep for one layer count
    submit_kvrh_job.slurm         # one V100 job = full sweep for one NUM_BLOCKS
    smoke_kvrh_layers.slurm       # feasibility smoke: compile + 2 epochs at nb=2 and nb=4
    smoke_kvrh_nb4_long.slurm     # nb=4 convergence check (clipnorm)
    curate_kvrh_results.py        # raw -> curated results/ layout
  results/                        # curated, tracked
    log_kvrh_fastfood_2layer_150epochs/
    log_kvrh_fastfood_4layer_180epochs/
```

## Data

The `log_kvrh` structures are recovered from the CGCNN raw split pickle and exported to the
ALIGNN JARVIS `id_prop.json` (`ALIGNN/ALIGNN_MP/alignn/build_log_kvrh_idprop.py`), then built
into DimeNet++ ragged-tensor caches via `data_saving_parallel.py` (`--source eform_idprop`,
generic JARVIS reader) for 3 split seeds. 10,987 structures (10,933 survive graph cleaning).

## Run

```bash
cd DimeNetPP/DimeNetPP_Matbench
# 1. build the 3 split-seed tensor caches (CPU, ~16 workers)
sbatch --export=ALL,DIMENET_DATASET=log_kvrh scripts/build_tensors_parallel.slurm
# 2. submit each layer count as one bundled V100 job (full 12-dim x 3-seed sweep)
sbatch --export=ALL,NUM_BLOCKS=2,DIMENET_EPOCHS=150 scripts/submit_kvrh_job.slurm
sbatch --export=ALL,NUM_BLOCKS=4,DIMENET_EPOCHS=180 scripts/submit_kvrh_job.slurm
# 3. curate raw outputs into results/
python scripts/curate_kvrh_results.py
```

## Method notes

- **num_blocks (layers)** is a CLI arg (`--num_blocks`). Parameter count scales as
  `D = 31,195 + num_blocks x 52,490` (fixed embedding/basis/output backbone + per-block params),
  so nb=2 -> D=136,175 and nb=4 -> D=241,155; the intrinsic dim is `d = id_dim x D`.
- **Gradient clipping (`clipnorm=1.0`, default)** is required for the deeper model: nb=4 with
  plain AdamW explodes at initialization (val_loss ~1e5-1e6) and barely recovers; clipnorm tames
  the spike and the 4-layer sweep trains cleanly. Applied to all runs for consistency.
- Optimizer: AdamW(lr=1e-3, amsgrad), MSE on standardized targets. V100, batch 64.
- Epochs: **150** for 2 layers, **180** for 4 layers (extra budget so the deeper model is not
  under-trained, keeping the depth comparison fair).
- Seeds: model/split = 123/1123, 456/1456, 789/1789.

## Results summary (mean test MAE over 3 seeds)

Intrinsic dim %: 1, 2, 5, 8, 10, 20, 45, 50, 65, 70, 80, 100.

- **2-layer:** error falls from ~0.115 (1%) to ~0.070 (100%), saturating above ~45%.
- **4-layer:** ~0.151 (1%) to ~0.066 (100%). Above ~20% dim the deeper model wins; below ~2%
  it is worse (harder to compress a larger model into a tiny subspace). Crossover ~20%.
