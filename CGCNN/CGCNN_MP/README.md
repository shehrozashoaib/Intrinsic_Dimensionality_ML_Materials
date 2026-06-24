# CGCNN MP Intrinsic-Dimensionality Experiments

CGCNN part of the intrinsic-dimensionality experiments for materials-property
prediction. The sweeps train a CGCNN crystal model with a **Fastfood random-subspace
wrapper** (intrinsic-dimension training: only the low-dimensional `z` is trained, projected
into the full weight space via a Fastfood transform) on the Materials Project
**formation energy** and **band gap** targets.

Only code, shell/Slurm scripts, small config, and curated results are tracked. Large items
(the conda env, cached tensors, raw per-run outputs, datasets) are git-ignored.

## Directory Layout

```text
CGCNN_MP/
  README.md
  requirements.txt                 # pip dependencies (torch 2.11+cu128, pymatgen, ...)
  requirements.sh                  # env bootstrap helper
  main.py                          # training entry; cached-MP path + per-run metadata/CSV
  materialize_mp_tensors.py        # build per-seed {train,val,test}.pt tensor caches (16-worker)
  predict.py                       # standalone prediction helper (upstream CGCNN)
  cgcnn/
    __init__.py
    data.py                        # CIFData / collate; default graph featurization settings
    model.py                       # CrystalGraphConvNet
    subspace.py                    # Fastfood/dense subspace wrapper (id_dim: float=frac, int=abs)
  scripts/
    build_mp_tensors.slurm         # materialize tensor caches for a task (seeds 1123/1456/1789)
    run_cgcnn_fastfood_mp_sweep.sh # sweep driver (array-index -> (dim, seed) runs)
    submit_cgcnn_mp_all.sh         # submit the sweep as grouped Slurm arrays
    submit_cgcnn_mp_job.slurm      # one-GPU sweep job (A100)
    smoke_cgcnn_mp.slurm           # short full-dataset timing smoke
  results/
    eform_fastfood_formation_energy_350epochs/
    bandgap_fastfood_band_gap_350epochs/
```

## Method & Model

- **Wrapper:** Fastfood random subspace (Li et al. 2018, *Measuring the Intrinsic
  Dimension of Objective Landscapes*). `id_dim` passed as a **fraction** of the full
  parameter count `D` (e.g. `0.45` -> 45% of `D`); an integer would be read as an absolute
  count (`cgcnn/subspace.py`).
- **Optimizer:** **Adam** (default in `main.py`). SGD freezes at low intrinsic dimension,
  so all MP sweeps use Adam.
- **Model:** `n_conv=4`, `atom_fea_len=56`, `h_fea_len=64`, `n_h=3` (~87.6k trainable
  parameters in the full space), `batch_size=256`, **350 epochs**.
- **Data:** Materials Project, default `cgcnn/data.py` graph featurization. Tensors are
  materialized once per data seed into `cached_mp/tensors/mp_<task>/seed<seed>/{train,val,test}.pt`
  (80:10:10 split) and reused across all dimensions.
- **Hardware:** NVIDIA A100 (torch 2.11.0+cu128 dropped Volta/sm_70, so V100 is unusable).

## Intrinsic-dimension sweep

Fractions swept and seeds:

| Target | id-dim fractions | model/split seeds | runs |
|--------|------------------|-------------------|------|
| Formation energy | 100, 80, 60, 50, 45, 20, 15, 10 % | 123/1123, 456/1456 (+789/1789 for 45 & 60 %) | 18 |
| Band gap | 100, 80, 60, 50, 45, 20, 10, 5 % | 123/1123, 456/1456 (+789/1789 for 45 & 60 %) | 18 |

## Reproduce

```bash
# from CGCNN_MP/  (conda env `py312`, A100)
sbatch scripts/build_mp_tensors.slurm        # 1) build tensor caches (per task)
bash   scripts/submit_cgcnn_mp_all.sh         # 2) submit the Fastfood sweeps
```

Each run writes `metadata.json`, a per-crystal `<task>_test_results.csv`, and `train.log`
into `results_mp_<task>_fastfood/<run>/`; a flock-guarded `..._summary.csv` aggregates MAE
and runtime. The curated copies under `results/` are produced from those raw outputs.

## Results summary

Test MAE decreases monotonically with intrinsic dimension on both targets (best seed shown):

| id-dim | eform test MAE (eV/atom) | band gap test MAE (eV) |
|-------:|-------------------------:|-----------------------:|
| 100 %  | 0.046 | 0.292 |
| 80 %   | 0.049 | 0.277 |
| 60 %   | 0.046 | 0.290 |
| 50 %   | 0.050 | 0.298 |
| 45 %   | 0.047 | 0.296 |
| 20 %   | 0.060 | 0.334 |
| 15 %   | 0.064 |   –   |
| 10 %   | 0.071 | 0.373 |
| 5 %    |   –   | 0.428 |

See each `results/<exp>/summary_mae_runtime.csv` for every run (all seeds, val/test MAE, runtime).
