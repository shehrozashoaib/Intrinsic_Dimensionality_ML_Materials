# Intrinsic Dimensionality of Materials Graph Neural Networks

Measuring the **trainable degrees of freedom** of materials GNNs with a random-subspace
("intrinsic dimension") wrapper, across three backbones and several property-prediction tasks.

We train each model inside a random `d`-dimensional subspace of its full parameter space
(`theta = theta_0 + P z`, only `z` trained) and sweep `d` from 100 % down to ~1 % of the model
size. The three backbones — **CGCNN**, **ALIGNN**, **DimeNet++** — are matched to a ~85k-parameter
budget so the comparison is at equal capacity.

> **This repository contains code only.** 

---

## Repository layout

The tree is organized **model → dataset source → task**:

```
CGCNN/
  CGCNN_MP/          # Materials Project tasks (formation energy, band gap, is_metal)
  CGCNN_Matbench/    # Matbench tasks (dielectric, log-K_VRH, phonons, is_metal)
ALIGNN/
  ALIGNN_MP/         # Materials Project tasks (formation energy, band gap, is_metal)
  ALIGNN_Matbench/   # Matbench tasks (dielectric, log-K_VRH, phonons)
DimeNetPP/
  DimeNetPP_MP/          # Materials Project tasks
  DimeNetPP_Matbench/    # Matbench tasks
environments/        # conda env exports with exact package versions
data/                # dataset download scripts (Zenodo + Matbench)
docs/                # source-code modification notes
```

Each `<model>/<dataset>/` has:
- `scripts/` — **bash `run_*.sh`** training scripts (the main entry points) and cache-build
  scripts. `.slurm` files are kept as optional cluster wrappers.
- model source (`cgcnn/`, `alignn/`, `dimenetpp_code_only/`) including the random-subspace wrapper.
- `configs/` (ALIGNN) — JSON model/optimizer configs.

---

## Quick start

```bash
git clone https://github.com/shehrozashoaib/Intrinsic_Dimensionality_ML_Materials.git
cd Intrinsic_Dimensionality_ML_Materials

# 1. environments (two of them — see below)
conda env create -f environments/cgcnn_alignn_py312.yml       # -> env: py312   (CGCNN, ALIGNN; PyTorch)
conda env create -f environments/dimenetpp_pydimnet.yml       # -> env: pydimnet (DimeNet++; TensorFlow)

# 2. datasets (formation energy + band gap from Zenodo; rest from Matbench)
bash data/download_datasets.sh

# 3. build caches (CGCNN + DimeNet++ tensorize once; ALIGNN builds graphs at run time)
#    see "Building caches" below

# 4. train (example: CGCNN formation-energy intrinsic-dimension sweep)
cd CGCNN/CGCNN_Matbench
CGCNN_TASK=eform bash scripts/run_cgcnn_eform_custom_sweep.sh
```

---

## Environments

Two conda environments are needed (the models use different deep-learning frameworks):

| Environment | File | Backbones | Key versions |
|---|---|---|---|
| `py312` | `environments/cgcnn_alignn_py312.yml` | CGCNN, ALIGNN | Python 3.12, torch 2.11+cu128, dgl 2.4, alignn 2026.5.20, pymatgen 2026.5.4, matbench 0.6 |
| `pydimnet` | `environments/dimenetpp_pydimnet.yml` | DimeNet++ | Python 3.x, tensorflow 2.17.1, numpy 1.26.4, pymatgen 2026.5.4 |

Exact pinned pip versions are in `environments/*_requirements.txt`.

> **Hardware note.** torch 2.11+cu128 has an SM-arch floor of **sm_75**, so CGCNN/ALIGNN run on
> RTX-2080-Ti or A100 (not V100). DimeNet++ (TensorFlow) runs on V100 or A100.

---

## Datasets

`bash data/download_datasets.sh` fetches everything:

- **Formation energy & band gap** — from our Zenodo record, **DOI
  [10.5281/zenodo.21871852](https://doi.org/10.5281/zenodo.21871852)**. Each is a gzipped JSON list
  of `{material_id, structure, target}` where `structure` is a pymatgen `Structure.as_dict()`.
- **Dielectric, log-K_VRH, phonons, is_metal** — from Matbench (`matbench==0.6`) via
  `data/fetch_matbench.py`, saved in the same format.

```python
import gzip, json
from pymatgen.core import Structure
data = json.load(gzip.open("data/MP_eform_dataset.json.gz"))
struct  = Structure.from_dict(data[0]["structure"])
target  = data[0]["formation_energy_per_atom"]
```

---

## Building caches

Training reads pre-tensorized caches (built once per dataset/split).

- **CGCNN** — `materialize_*_tensors.py` (driven by `scripts/build_*_tensors.slurm`) build the CGCNN
  graph caches under `cached_mp/` and `cached_matbench/`.
- **DimeNet++** — `scripts/build_*_tensors.slurm` / `build_tensors_parallel.slurm` build the
  cached tensors under `cached_tensors_dimenetpp_*`.
- **ALIGNN** — no pre-build; it constructs line graphs at run time from the `id_prop.json`
  datasets (created from the downloaded data).

Each build script has a plain-bash core and can be run directly or via `sbatch`.

---

## Running training

All sweeps are **bash scripts** in each `<model>/<dataset>/scripts/`. They loop over intrinsic
dimensions (fraction of `D`) and seeds and write per-run results. The subspace fraction, seed,
optimizer, LR schedule, and best-model evaluation are all set inside these scripts / the runner
flags (see `docs/SOURCE_MODIFICATIONS.md`), so no separate config editing is needed.

| Backbone | Example entry point |
|---|---|
| CGCNN | `CGCNN/CGCNN_Matbench/scripts/run_cgcnn_eform_custom_sweep.sh` |
| ALIGNN | `ALIGNN/ALIGNN_MP/scripts/run_alignn_fastfood_eform_sweep.sh` |
| DimeNet++ | `DimeNetPP/DimeNetPP_MP/scripts/run_dimenetpp_fastfood_eform_sweep.sh` |

**Learning-rate schedules are embedded per model/task** (they are not uniform):
- CGCNN: `MultiStepLR` (x0.1 at epoch 100) by default; OneCycle sweeps set `--lr-schedule onecycle`.
  **CGCNN on Matbench dielectric under OneCycle uses `--lr 0.1`** (100x; it collapses at 1e-3) —
  baked into `run_cgcnn_dielectric_onecycle_sweep.sh`.
- ALIGNN: `OneCycleLR` (set in `configs/config_*.json`); best-model eval via `ALIGNN_RESTORE_BEST=1`.
- DimeNet++: constant LR in the main experiments; `--lr_schedule onecycle` for the schedule study;
  best-model eval via `--restore_best`.

---

## Experiments and where the code lives

Each experiment reported in the paper maps to the scripts below. `<model>` is `CGCNN`, `ALIGNN`, or
`DimeNetPP`; `<dataset>` is the MP or Matbench sub-tree.

| Paper item | Experiment | Code |
|---|---|---|
| **Fig. 2** | Main intrinsic-dimension sweep — 3 models x 6 tasks (formation energy, band gap, is_metal, dielectric, log-K_VRH, phonons) | `<model>/<dataset>/scripts/run_*_fastfood_*_sweep.sh` |
| **Table 1** | Parameter-matched (~85k) configs and d=D reference | ALIGNN `configs/config_*.json`; CGCNN/DimeNet++ flags in the sweep scripts |
| **Fig. 3** | Dataset-size response — full / quarter / tenth subsets (ALIGNN + DimeNet++; eform, band gap, log-K_VRH) | build: `ALIGNN/ALIGNN_MP/alignn/build_mp_quarters.py, ALIGNN/ALIGNN_Matbench/alignn/build_kvrh_{quarters,tenths}.py`, `DimeNetPP/*/scripts/build_partition_tensors.slurm`; run: `run_alignn_fastfood_bandgap_sweep.sh` (per-seed splits), `DimeNetPP/DimeNetPP_Matbench/scripts/run_dimenetpp_kvrh_bestmodel_sweep.sh` |
| **Fig. 4** | Model-width sweep (conv channels), log-K_VRH | `ALIGNN/ALIGNN_Matbench/scripts/run_alignn_convwidth_kvrh_sweep.sh`, `DimeNetPP/DimeNetPP_Matbench/scripts/run_dimenetpp_fastfood_kvrh_convwidth_sweep.sh` |
| **Fig. S1** | Projection-construction control — Dense / Fastfood / dense-orth. / Fastfood-orth. (DimeNet++, log-K_VRH) | `DimeNetPP/DimeNetPP_Matbench/launch_wrapper_variants.sh`, `DimeNetPP/DimeNetPP_MP/dimenetpp_code_only/run_dense_{,orthonormal_}sweep_v3.sh` |
| Methods control | Optimizer / LR-schedule / no-wrapper controls (DimeNet++) | `run_dimenetpp_optimizer_test.sh`, `run_dimenetpp_arm_sweep.sh`, `run_dimenetpp_nowrapper_eform.sh`, `DimeNetPP/DimeNetPP_Matbench/scripts/run_dimenetpp_dielectric_arm.sh` |
| Methods control | Depth control — 2-layer configs | `configs/config_*_2layer.json`, `run_*_2layer_*.sh` (all three models) |
| Methods control | Phonon label-shuffle (memorization) | `*_phonons*` sweeps; `ALIGNN/ALIGNN_Matbench/alignn/build_phonons_idprop.py` |

The random-subspace wrapper itself: `cgcnn/subspace.py` (CGCNN), inside `alignn/train_alignn.py`
(ALIGNN), `dimenetpp_code_only/wrapper_tensorflow_v3.py` (DimeNet++). See
`docs/SOURCE_MODIFICATIONS.md`.

---

## The random-subspace wrapper

`theta = theta_0 + P z` with `theta_0` frozen and only the `d`-dim `z` trained. `P` is the
implicit **non-orthonormal Fastfood** transform by default (`O(D)` memory, applied as two Fast
Walsh-Hadamard transforms); dense and orthonormalized variants exist for the projection-construction
control. Implementations:
- CGCNN — `cgcnn/subspace.py`
- ALIGNN — inside `alignn/train_alignn.py`
- DimeNet++ — `dimenetpp_code_only/wrapper_tensorflow_v3.py`

---

## Reproducibility notes

- Every change made to the three upstream model codebases is documented in
  **`docs/SOURCE_MODIFICATIONS.md`** (wrapper, optimizer/schedule flags, best-model checkpoint fix).
- Seeds set both the model initialization and the train/val/test split; sweeps use multiple seeds
  per dimension to quantify seed-to-seed variance.
- Raw run outputs and caches are intentionally untracked (`.gitignore`); regenerate them with the
  scripts above. A **curated results release** — the validation + test metric behind every plotted
  point, organized as `results/<Model>/<task>/<partition>.csv` — is tracked under
  [`results/`](results/) (see `results/README.md`).
