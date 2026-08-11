# ALIGNN — Materials Project tasks

ALIGNN (AtomWise model) with the Fastfood random-subspace wrapper, for the **Materials Project**
property-prediction tasks: **formation energy**, **band gap**, and **is_metal** (classification).
Matbench tasks (dielectric, log-K_VRH, phonons) live in `../ALIGNN_Matbench/`.

> **Environment and dataset setup are documented once in the [top-level README](../../README.md)**
> (conda env `py312` from `environments/`, datasets via `data/download_datasets.sh`). This file
> only covers ALIGNN-MP-specific usage. Do not follow per-folder env/data instructions elsewhere —
> the top-level README is the single source of truth.

## Layout

```text
ALIGNN_MP/
  configs/        config_eform.json, config_bandgap.json, config_is_metal.json (+ 2layer variants)
  alignn/         ALIGNN model code + the random-subspace wrapper (train_alignn.py, train.py, ...)
                  and MP data builders (convert_eform_fast.py, build_is_metal_idprop.py, build_mp_quarters.py)
  scripts/        run_alignn_fastfood_{eform,bandgap,is_metal}_sweep.sh, run_alignn_2layer_full.sh,
                  run_alignn_fastfood_partition_sweep.sh (dataset-size, eform+bandgap)
```

## Method

`theta = theta_0 + P z`, only the `d`-dim `z` trained; `P` is the non-orthonormal Fastfood
projection. LR schedule is `OneCycleLR` (set in `configs/config_*.json`). Best-checkpoint test
evaluation is enabled with `ALIGNN_RESTORE_BEST=1` (fixes the upstream `best_model = net` alias so
the test metric is the best-val model, not the final epoch).

## Reproduce

```bash
# 0) env + data: see top-level README (creates the id_prop.json datasets under alignn/)
# 1) intrinsic-dimension sweep (formation energy), best-model eval on:
cd ALIGNN/ALIGNN_MP
ALIGNN_RESTORE_BEST=1 EFORM_RUNS="1.0:100:123:1123 0.5:50:123:1123" \
  bash scripts/run_alignn_fastfood_eform_sweep.sh
```

`EFORM_RUNS` items are `id_dim:percent:model_seed:split_seed`; omit it to run the full schedule.
Band gap uses per-seed reshuffled datasets (`MP_json_seed<N>`) via `run_alignn_fastfood_bandgap_sweep.sh`.
