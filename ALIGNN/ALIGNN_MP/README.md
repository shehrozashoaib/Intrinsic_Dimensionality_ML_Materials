# ALIGNN MP Intrinsic-Dimensionality Experiments

This directory contains the ALIGNN part of the intrinsic-dimensionality experiments for materials-property prediction. The current sweep targets Materials Project band gap prediction using an ALIGNN AtomWise model with a Fastfood random-subspace wrapper.

Only code, shell scripts, and small config files are included here. Large datasets such as `mp21.jsonl`, generated `MP_json` folders, checkpoints, and run outputs are intentionally excluded.

## Directory Layout

```text
ALIGNN_MP/
  README.md
  requirements.txt
  configs/
    config_bandgap.json
    config_eform.json
    config.json
  alignn/
    train_alignn.py
    train.py
    data.py
    dataset.py
    graphs.py
    config.py
    utils.py
    convert_jsonl_to_idprop.py
    convert_jsonl_to_idprop_formation_energy.py
    run_alignn_ff.py
    cli.py
  scripts/
    run_alignn_fastfood_bandgap_sweep.sh
```

## Environment Setup

Install the environment in this order.

```bash
conda create -n py312 python=3.12

# 1) activate your env
conda activate py312

# 2) add Jupyter kernel files for this env
python -m pip install --upgrade pip ipykernel
python -m ipykernel install --user --name py312 --display-name "Python 3.12 (py312)"

conda install alignn
conda install -c conda-forge jarvis-tools

pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install matminer pymatgen pandas numpy mp_api

conda install -c dglteam/label/th24_cu124 dgl
```

The scripts assume the conda environment is named `py312`.

## Band Gap Dataset

The band gap workflow uses a local `mp21.jsonl` file. Place it inside the `alignn/` directory before preprocessing.

Example transfer commands:

```bash
curl https://rclone.org/install.sh | sudo bash
rclone config file
rclone copy "file:mp21.jsonl" /workspace/alignn -P
```

For this repository layout, copy or move the file into:

```text
ALIGNN/ALIGNN_MP/alignn/mp21.jsonl
```

Then convert it into ALIGNN's `id_prop.json` format:

```bash
cd ALIGNN/ALIGNN_MP/alignn
conda run -n py312 python convert_jsonl_to_idprop.py \
  --jsonl mp21.jsonl \
  --out_dir MP_json \
  --id_key material_id \
  --target_key band_gap \
  --seed 123
```

The conversion script writes:

```text
alignn/MP_json/id_prop.json
```

`MP_json/` is generated data and is ignored by git.

## Formation Energy Dataset

For formation energy, use the Materials Project API instead of the local band-gap JSONL workflow.

Set your API key in the environment:

```bash
export MP_API_KEY="your_materials_project_api_key"
```

Then run:

```bash
cd ALIGNN/ALIGNN_MP/alignn
conda run -n py312 python convert_jsonl_to_idprop_formation_energy.py \
  --out_dir MP_json_eform \
  --id_key material_id \
  --target_key formation_energy_per_atom
```

The script reads `MP_API_KEY` from the environment. Do not hardcode or commit API keys.

## Model and Wrapper

`train_alignn.py` supports intrinsic-dimension training through `RandomSubspaceWrapper`.

Important CLI flags:

```text
--subspace_method fastfood
--id_dim <fraction-or-integer>
--id_enable True
--random_seed <model/subspace seed>
--split_seed <train/val/test split seed>
```

`--random_seed` controls model initialization and the random Fastfood projection. `--split_seed` controls train/validation/test splitting. The sweep script uses the same value for both within each data-seed condition.

By default, seeded runs set Python, NumPy, Torch, and CUDA RNG state without forcing PyTorch's slow deterministic CUDA kernels. If strict deterministic kernels are required, set `ALIGNN_DETERMINISTIC=1` in the environment before running training.

## Single Run Example

From `ALIGNN/ALIGNN_MP/alignn`:

```bash
conda run -n py312 python train_alignn.py \
  --root_dir MP_json \
  --config_name ../configs/config_bandgap.json \
  --target_key band_gap \
  --id_key material_id \
  --subspace_method fastfood \
  --id_dim 1.0 \
  --id_enable True \
  --epochs 350 \
  --output_dir ../results_alignn_bandgap_fastfood/debug_dim100_seed123 \
  --random_seed 123 \
  --split_seed 123
```

## Full Band Gap Sweep

From the repository root or this directory:

```bash
bash ALIGNN/ALIGNN_MP/scripts/run_alignn_fastfood_bandgap_sweep.sh
```

Or from inside `ALIGNN/ALIGNN_MP`:

```bash
bash scripts/run_alignn_fastfood_bandgap_sweep.sh
```

The sweep runs:

1. Existing `MP_json`, seed `123`: `100%, 80%, 60%, 50%, 45%, 20%, 10%, 5%`
2. Reshuffled `MP_json_seed456`, seed `456`: `100%, 80%, 60%, 50%, 45%, 20%, 10%, 5%`
3. Reshuffled `MP_json_seed789`, seed `789`: `60%, 45%`

All runs use:

```text
subspace_method = fastfood
epochs = 350
target_key = band_gap
id_key = material_id
config = configs/config_bandgap.json
```

The script writes one combined summary file:

```text
ALIGNN/ALIGNN_MP/results_alignn_bandgap_fastfood/alignn_bandgap_fastfood_summary.csv
```

Each run records:

```text
task, target_key, wrapper, id_dim, id_dim_percent, data_seed, model_seed,
split_seed, epochs, batch_size, root_dir, output_dir, predictions_csv,
history_json, best_val_mae, test_mae, duration_sec, status, exit_code,
start_time, end_time
```

`test_mae` is computed from `prediction_results_test_set.csv`, which is generated from ALIGNN's test loader. `best_val_mae` is recorded separately from `history_val_mae.json`.

The script removes checkpoint files after metadata is recorded to avoid filling the disk. It keeps logs, metadata JSON, prediction CSVs, validation history, and the summary CSV.

