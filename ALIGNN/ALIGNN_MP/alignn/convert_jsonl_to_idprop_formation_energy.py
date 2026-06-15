# convert_jsonl_to_idprop_mpapi.py
# usage:
#     python convert_jsonl_to_idprop_formation_energy.py --out_dir MP_json --id_key material_id --target_key formation_energy_per_atom
import json, os, argparse
import random
from tqdm import tqdm
from jarvis.core.atoms import pmg_to_atoms # <-- this is the correct converter

# --- 1. MP-API Imports and Configuration ---
import mp_api
from mp_api.client import MPRester
# Set MP_API_KEY in your shell before running this script.
API_KEY_ENV_VAR = 'MP_API_KEY'
RANDOM_SEED = 423
# -------------------------------------------


p = argparse.ArgumentParser()
# p.add_argument("--jsonl", required=True, help="Path to your JSONL file") # Removed: No longer reading a local JSONL file
p.add_argument("--out_dir", default="MP_json", help="Output directory")
p.add_argument("--id_key", default="material_id")
p.add_argument("--target_key", default="formation_energy_per_atom") # New default target key
p.add_argument("--limit", type=int, default=0, help="0 = all")
args = p.parse_args()

os.makedirs(args.out_dir, exist_ok=True)
records = []

## 2. Data Fetching and Preparation using MPRester

print("Connecting to Materials Project API...")
api_key = os.environ.get(API_KEY_ENV_VAR)
if not api_key:
    raise RuntimeError(f"Set {API_KEY_ENV_VAR} before running this script.")
mpr = MPRester(api_key)
results = mpr.materials.summary.search( fields=["material_id","structure","formation_energy_per_atom"])


# Filter out records where the target property is missing
filtered_results = [
    result for result in results if getattr(result, args.target_key) is not None
]

print(f"Fetched {len(filtered_results)} records from MP API.")

# Set seed and shuffle
print(f"Applying random seed {RANDOM_SEED} and shuffling...")
random.seed(RANDOM_SEED)
random.shuffle(filtered_results)
print("Shuffling complete.")


# --- 3. Process Data and Convert Structures ---

for idx, result in enumerate(tqdm(filtered_results, desc="Processing MP Data")):
    if args.limit and idx >= args.limit:
        break

    mid = getattr(result, args.id_key)
    y = getattr(result, args.target_key)
    pmg_struct = result.structure

    # Convert the pymatgen Structure -> JARVIS Atoms
    jv_atoms = pmg_to_atoms(pmg_struct).to_dict()

    records.append({
        args.id_key: mid,
        args.target_key: y,
        "atoms": jv_atoms
    })


## 4. Final Output

print(f"Processed {len(records)} records. Writing to file...")
out_path = os.path.join(args.out_dir, "id_prop.json")
with open(out_path, "w") as g:
    json.dump(records, g)

print(f"Wrote {len(records)} shuffled samples to {out_path}")