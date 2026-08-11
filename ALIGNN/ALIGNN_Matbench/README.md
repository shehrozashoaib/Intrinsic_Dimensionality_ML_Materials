# ALIGNN — Matbench tasks

ALIGNN (AtomWise model) with the Fastfood random-subspace wrapper, for the **Matbench**
property-prediction tasks: **dielectric**, **log-K_VRH**, and **phonons**. Materials Project tasks
(formation energy, band gap, is_metal) live in `../ALIGNN_MP/`.

> **Environment and dataset setup are documented once in the [top-level README](../../README.md)**
> (conda env `py312` from `environments/`, datasets via `data/download_datasets.sh`). This file
> only covers ALIGNN-Matbench-specific usage.

## Layout

```text
ALIGNN_Matbench/
  configs/        config_dielectric.json, config_log_kvrh*.json, config_phonons.json (+ 2layer),
                  configs/fixedtest/  (log-K_VRH fixed-test split configs)
  alignn/         ALIGNN model code + wrapper (shared copy) and Matbench data builders
                  (build_{dielectric,log_kvrh,phonons}_idprop.py, build_kvrh_{quarters,tenths,fixedtest}.py)
  scripts/        run_alignn_fastfood_{dielectric,log_kvrh,phonons}_sweep.sh,
                  run_alignn_convwidth_kvrh_sweep.sh (model-width study, Fig. 4),
                  run_alignn_kvrh_fixedtest_sweep.sh
```

## Method

`theta = theta_0 + P z`, only the `d`-dim `z` trained; `P` is the non-orthonormal Fastfood
projection. LR schedule is `OneCycleLR` (in `configs/config_*.json`); best-checkpoint test
evaluation via `ALIGNN_RESTORE_BEST=1`.

## Reproduce

```bash
# 0) env + data: see top-level README, then build the id_prop.json for the task, e.g.
cd ALIGNN/ALIGNN_Matbench/alignn
python build_log_kvrh_idprop.py           # -> MP_json_log_kvrh/id_prop.json
# 1) intrinsic-dimension sweep (log-K_VRH):
cd ..
ALIGNN_RESTORE_BEST=1 bash scripts/run_alignn_fastfood_log_kvrh_sweep.sh
# model-width study (Fig. 4):
bash scripts/run_alignn_convwidth_kvrh_sweep.sh
```
