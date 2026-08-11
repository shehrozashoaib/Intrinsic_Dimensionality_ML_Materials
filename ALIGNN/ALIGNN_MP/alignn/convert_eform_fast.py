#!/usr/bin/env python
# convert_eform_fast.py
# Arrow-native replacement for convert_jsonl_to_idprop_formation_energy.py.
# The new mp_api client returns an Arrow-backed MPDataset; iterating it row by
# row (the old script's pattern) does one Arrow `take` per row and is
# pathologically slow over ~170k materials. This reads the needed columns in
# one bulk Arrow call, then only loops for the unavoidable structure conversion
# (with a visible progress bar).
#
# usage:
#   python convert_eform_fast.py --out_dir MP_json_eform \
#       --id_key material_id --target_key formation_energy_per_atom
import json, os, argparse, random
from tqdm import tqdm
from jarvis.core.atoms import pmg_to_atoms
from pymatgen.core import Structure
from mp_api.client import MPRester

API_KEY_ENV_VAR = "MP_API_KEY"
RANDOM_SEED = 423   # same as the original; harmless since the ALIGNN splitter
                    # reshuffles anyway (keep_data_order=False)


def rows_from_results(results, id_key, target_key):
    """Bulk-read material_id / structure / target columns via Arrow."""
    cols = [id_key, "structure", target_key]
    # results is an MPDataset; .pyarrow_dataset exposes the raw pyarrow dataset
    table = results.pyarrow_dataset.to_table(columns=cols)
    return table.to_pylist(maps_as_pydicts="strict")


def build(rows, id_key, target_key, limit):
    rows = [r for r in rows if r.get(target_key) is not None]
    print(f"{len(rows)} rows after dropping missing {target_key}.")

    random.seed(RANDOM_SEED)
    random.shuffle(rows)

    # Fail fast with a useful message if the structure schema isn't what we expect
    probe = rows[0]["structure"]
    try:
        Structure.from_dict(probe)
    except Exception:
        print("ERROR: could not parse the 'structure' column via Structure.from_dict.")
        print("  type:", type(probe))
        if isinstance(probe, dict):
            print("  keys:", list(probe.keys())[:20])
        else:
            print("  value head:", str(probe)[:300])
        raise SystemExit("Structure schema differs from expectation; paste the above.")

    records, n_bad = [], 0
    for idx, r in enumerate(tqdm(rows, desc="Converting structures")):
        if limit and idx >= limit:
            break
        try:
            s = Structure.from_dict(r["structure"])
            jv = pmg_to_atoms(s).to_dict()
        except Exception:
            n_bad += 1
            continue
        records.append({id_key: r[id_key], target_key: r[target_key], "atoms": jv})
    return records, n_bad


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default="MP_json")
    p.add_argument("--id_key", default="material_id")
    p.add_argument("--target_key", default="formation_energy_per_atom")
    p.add_argument("--limit", type=int, default=0, help="0 = all")
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print("Connecting to Materials Project API...")
    api_key = os.environ.get(API_KEY_ENV_VAR)
    if not api_key:
        raise RuntimeError(f"Set {API_KEY_ENV_VAR} before running this script.")
    mpr = MPRester(api_key)
    results = mpr.materials.summary.search(
        fields=["material_id", "structure", "formation_energy_per_atom"]
    )

    print("Reading columns via Arrow (bulk, fast path)...")
    rows = rows_from_results(results, args.id_key, args.target_key)
    print(f"Loaded {len(rows)} rows from the Arrow dataset.")

    records, n_bad = build(rows, args.id_key, args.target_key, args.limit)

    out_path = os.path.join(args.out_dir, "id_prop.json")
    with open(out_path, "w") as g:
        json.dump(records, g)
    print(f"Wrote {len(records)} samples to {out_path} (skipped {n_bad} unparseable).")


if __name__ == "__main__":
    main()
