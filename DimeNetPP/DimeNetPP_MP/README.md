# DimeNet++ MP Intrinsic-Dimensionality Experiments

DimeNet++ part of the intrinsic-dimensionality experiments for materials-property
prediction. The sweeps train a DimeNet++ crystal model with a **Fastfood random-subspace
wrapper** (intrinsic-dimension training: only the low-dimensional `z` is trained, projected
into the full weight space) on the Materials Project **formation energy** and **band gap**
targets.

Only code, shell/Slurm scripts, small config, and curated results are tracked. Large items
(the conda env, cached tensors, raw per-run outputs, datasets) are git-ignored.

## Directory Layout

```text
DimeNetPP_MP/
  README.md
  requirements_dimenetpp.txt
  environment_dimenetpp.yml
  dimenetpp_code_only/
    data_saving_from_alignn_eform.py     # eform tensors from ALIGNN id_prop.json (serial)
    data_saving_bandgap_from_mp21.py     # band-gap tensors from mp21.jsonl (serial)
    data_saving_parallel.py              # build-once + reshuffle, 16-worker parallel (USED)
    dimenet_run_eform_v3.py              # wrapper-v3 training runner (eform)
    dimenet_run_v3.py                    # wrapper-v3 training runner (band gap, original)
    wrapper_tensorflow_v3.py             # SubspaceProjectedGradTFV3 (Fastfood/dense wrapper)
    torch_orthonormal_q_v3.py            # external PyTorch QR for dense-orthonormal runs
    bench_parallel_build.py              # serial-vs-multiprocessing build benchmark
    ISSUES_DENSE_FASTFOOD_WRAP_HISTORY.md# dense/fastfood wrapper debug history
  scripts/
    setup_dimenetpp_env.sh               # build the conda env
    dimenet_env.sh                       # activate env + set LD_LIBRARY_PATH (GPU fix)
    build_tensors_parallel.slurm         # parallel build (eform|bandgap) -> 3 seed caches
    build_eform_tensors.slurm            # serial fallback (per-seed array)
    build_bandgap_tensors.slurm          # serial fallback (per-seed array)
    bench_parallel_build.slurm           # run the build benchmark
    run_dimenetpp_fastfood_eform_sweep.sh# sweep driver (array-index -> run)
    submit_eform_all.sh                  # submit the sweep as a Slurm array
    submit_eform_job.slurm               # one-GPU sweep job
    submit_eform_bench.sh/.slurm         # V100-vs-A100 1-epoch timing
    submit_eform_smoke.slurm             # short timing smoke
    aggregate_summary.py                 # metadata.json -> summary CSV
    curate_results.py                    # raw outputs -> results/ layout (this folder)
  results/
    eform_fastfood_formation_energy_350epochs/
    bandgap_fastfood_band_gap_350epochs/
```

## Environment Setup

The env is a local conda prefix (`envs/pydimnet`, git-ignored) so it can be shipped to
compute nodes alongside the code.

```bash
# from DimeNetPP_MP/
bash scripts/setup_dimenetpp_env.sh        # CONDA_BASE defaults to /ibex/user/$USER/miniconda3
```

This creates `envs/pydimnet` (Python 3.12) and installs `requirements_dimenetpp.txt`
(TensorFlow 2.17.1 + CUDA wheels, jarvis-tools, pymatgen, numpy<2, scikit-learn, pyyaml …)
plus **`kgcnn==4.0.2`** and **`dm-tree`**.

Before any GPU run, source `scripts/dimenet_env.sh` — it sets `DIMENET_PYTHON` and, critically,
prepends the env-local NVIDIA wheel `lib` dirs to `LD_LIBRARY_PATH` (see issue 1 below).

## Environment issues we hit (and fixes)

1. **TensorFlow silently ran on CPU (~4× slowdown).** Logs showed `Cannot dlopen some GPU
   libraries` / `Visible GPUs: []`. The CUDA libraries ship *inside* the env
   (`envs/pydimnet/lib/python3.12/site-packages/nvidia/*/lib`) but those dirs were not on
   `LD_LIBRARY_PATH`, so the dynamic linker never found them. **Fix:** `dimenet_env.sh`
   derives the NVIDIA lib dirs from the env and prepends them to `LD_LIBRARY_PATH`. Every
   sweep/bench/smoke script sources it, so GPU runs must go through these scripts.

2. **kgcnn 2.1.0 is Keras-2 only and breaks under TF 2.17 / Keras 3.** The v3 run scripts use
   the Keras-3 config keys (`input_node_embedding`, `input_tensor_type`,
   `output_tensor_type`) and `tf.keras.Model` subclassing with a custom `train_step`. **Fix:**
   install **`kgcnn==4.0.2`** (the Keras-3-compatible line). It also needs **`dm-tree`**
   (`tree`) at runtime, which it does not pin — so we install it explicitly.

3. **`No module named 'yaml'`.** `kgcnn.data.crystal` (used by the tensor builders) imports
   `yaml`, which kgcnn does not pull in when installed without full deps. **Fix:** `pyyaml`
   is in `requirements_dimenetpp.txt`.

4. **Cached tensors deserialized onto the GPU and OOM'd the card.** The pickled caches hold TF
   tensors; loading them put the whole dataset on the default device (GPU). **Fix:**
   `dimenet_run_eform_v3.py` loads the cache inside `with tf.device("/CPU:0")` so the tensors
   live in host RAM and Keras streams per-batch to the GPU.

5. **`OverflowError: serializing a bytes object larger than 4 GiB` when saving caches.**
   `np.savez_compressed` pickles object arrays with protocol 3, which caps a single object at
   4 GiB; the full `x_train` ragged set overflows it. **Fix:** the builders save caches with
   **`pickle` protocol 5** (`*.pkl`), and the run scripts load via `pickle.load`.

6. **Tensor build was slow (single-threaded kgcnn graph build).** `set_range_periodic` +
   `set_angle` over 154k crystals is pure-CPU and single-threaded (~18 struct/s →
   ~2.3 h/seed). `bench_parallel_build.py` showed a 16-worker `spawn` pool reaches ~98 struct/s
   (**5.4×**). **Fix:** `data_saving_parallel.py` builds all graphs once with a process pool
   (`spawn`, because TF + `fork` deadlocks) and reshuffles into the 3 seed splits.

See `dimenetpp_code_only/ISSUES_DENSE_FASTFOOD_WRAP_HISTORY.md` for the dense/fastfood wrapper
and dense-orthonormal-100% debugging history.

## Datasets / Tensor Caches

Caches are built once per split seed (`1123`, `1456`, `1789`) and stored as
`cached_tensors_dimenetpp_<task>/splitseed<seed>/dimenetpp_<task>_cached_tensors.pkl`
(+ `scaler_<task>.pkl`). 80/10/10 train/val/test.

```bash
# parallel build-once + reshuffle (recommended)
sbatch --export=ALL,DIMENET_DATASET=eform    scripts/build_tensors_parallel.slurm
sbatch --export=ALL,DIMENET_DATASET=bandgap  scripts/build_tensors_parallel.slurm
```

- **eform** structures come from the ALIGNN `MP_json_eform/id_prop.json` (JARVIS atoms →
  pymatgen), target `formation_energy_per_atom`.
- **band gap** structures come from `mp21.jsonl` (pymatgen `structure` dicts), target
  `band_gap`.

## Timing / hardware

A 1-epoch V100-vs-A100 bench (`scripts/submit_eform_bench.sh`) showed the V100 (32 GB) was the
faster, better-queued option for this small model, so the sweeps ran on V100. CPU tensor builds
run on the `batch` partition.

## Fastfood Sweep

16 eform runs / 18 band-gap runs (band gap adds a 5% point), submitted as one Slurm array
(batch format, never an interactive whole-GPU allocation):

```bash
bash scripts/submit_eform_all.sh                 # eform array on V100
```

Intrinsic-dimension fractions: `5%`(band gap only)`,10,20,45,50,60,80,100`. Seeds: model
`123/456` (+`789` for 45% and 60%), split `1123/1456/1789`. Each array task writes its own
`metadata.json`; aggregate with `scripts/aggregate_summary.py`, then curate the GitHub layout
with `scripts/curate_results.py`.

## Results

See `results/<exp>/`:

- `summary_mae_runtime.csv` — val/test MAE (eV) + runtime per dimension/seed.
- `predictions/` — per-crystal test-set predictions for every run.
- `logs/` — one representative training log per intrinsic-dimension fraction.
