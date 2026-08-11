# convert_jsonl_to_idprop.py
# usage:
#    python convert_jsonl_to_idprop.py --jsonl mp21.jsonl --out_dir MP_json --id_key material_id --target_key band_gap
import json, os, argparse
import random  # <-- 1. IMPORT random
from tqdm import tqdm
from pymatgen.core import Structure
from jarvis.core.atoms import pmg_to_atoms  # <-- this is the correct converter

p = argparse.ArgumentParser()
p.add_argument("--jsonl", required=True, help="Path to your JSONL file")
p.add_argument("--out_dir", default="MP_json", help="Output directory")
p.add_argument("--id_key", default="material_id")
p.add_argument("--target_key", default="band_gap")
p.add_argument("--limit", type=int, default=0, help="0 = all")
p.add_argument("--seed", type=int, default=123, help="Seed used when shuffling records")
p.add_argument("--no_shuffle", action="store_true", help="Write records in source JSONL order")
args = p.parse_args()

os.makedirs(args.out_dir, exist_ok=True)
records = []

with open(args.jsonl, "r") as f:
    for idx, line in enumerate(tqdm(f, desc="Reading JSONL")):
        if args.limit and idx >= args.limit:
            break
        line = line.strip()
        if not line:
            continue

        row = json.loads(line)
        mid = row[args.id_key]
        y = float(row[args.target_key])

        # Convert the pymatgen Structure dict -> pymatgen Structure -> JARVIS Atoms
        pmg_struct = Structure.from_dict(row["structure"])
        jv_atoms = pmg_to_atoms(pmg_struct).to_dict()

        records.append({
            args.id_key: mid,
            args.target_key: y,
            "atoms": jv_atoms
        })

# <-- 2. SHUFFLE THE LIST
if args.no_shuffle:
    print(f"Read {len(records)} records. Writing without shuffling.")
else:
    print(f"Read {len(records)} records. Now shuffling with seed={args.seed}...")
    random.seed(args.seed)
    random.shuffle(records)
    print("Shuffling complete.")

out_path = os.path.join(args.out_dir, "id_prop.json")
with open(out_path, "w") as g:
    json.dump(records, g)

print(f"Wrote {len(records)} shuffled samples to {out_path}")