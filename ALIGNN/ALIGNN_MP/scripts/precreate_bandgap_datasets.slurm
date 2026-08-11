#!/bin/bash --login
# Pre-create the seed456 / seed789 band-gap id_prop datasets from mp21.jsonl ONCE,
# so the per-run GPU jobs don't race creating them. CPU-only, quick.
# Submit from ALIGNN/ALIGNN_MP:  sbatch scripts/precreate_bandgap_datasets.slurm
#SBATCH --job-name=bg_precreate_ds
#SBATCH --partition=batch
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/bg_precreate_ds_%j.out
#SBATCH --error=logs/bg_precreate_ds_%j.err

set -euo pipefail
mkdir -p logs
source /ibex/user/$USER/miniconda3/etc/profile.d/conda.sh
conda activate py312
cd "${SLURM_SUBMIT_DIR:-$PWD}/alignn"

for seed in 456 789; do
  out="MP_json_seed${seed}"
  if [[ -f "${out}/id_prop.json" ]]; then
    echo "[precreate] ${out}/id_prop.json already exists, skipping"
    continue
  fi
  echo "[precreate] building ${out} from mp21.jsonl (seed=${seed})"
  python convert_jsonl_to_idprop.py \
    --jsonl mp21.jsonl \
    --out_dir "$out" \
    --id_key material_id \
    --target_key band_gap \
    --seed "$seed"
done
echo "[precreate] done $(date -Iseconds)"
ls -la MP_json_seed456/id_prop.json MP_json_seed789/id_prop.json
