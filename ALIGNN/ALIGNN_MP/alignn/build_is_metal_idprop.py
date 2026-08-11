#!/usr/bin/env python3
"""Build a unified ALIGNN id_prop.json for matbench_mp_is_metal (binary classification).

Source: CGCNN raw split pickle
  CGCNN/CGCNN_Matbench/cached_matbench/raw/matbench_mp_is_metal_seed123_raw.pkl
which holds pymatgen Structure objects + is_metal boolean targets (the official
106,113-entry matbench task). Take the FULL union of the raw splits and convert each
Structure -> JARVIS atoms dict (records: {material_id, is_metal 0/1, atoms}).

This single file feeds all three models:
  * ALIGNN   : --root_dir MP_json_is_metal --target_key is_metal --classification_threshold 0.5
  * DimeNet++: data_saving_parallel.py --source eform_idprop --target_key is_metal --classification
  * CGCNN    : materialize_mp_tensors.py --source is_metal_idprop
Each model does its own 80/10/10 reshuffle per seed.

Output: alignn/MP_json_is_metal/id_prop.json
"""
import json, os, pickle, warnings
warnings.filterwarnings("ignore")
from jarvis.core.atoms import pmg_to_atoms

ALIGNN_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_PKL = os.path.join(ALIGNN_DIR, "..", "..", "..", "CGCNN", "CGCNN_Matbench",
                       "cached_matbench", "raw", "matbench_mp_is_metal_seed123_raw.pkl")
TARGET_KEY = "is_metal"
ID_KEY = "material_id"


def main():
    with open(RAW_PKL, "rb") as f:
        payload = pickle.load(f)
    assert payload["task"] == "matbench_mp_is_metal", payload["task"]
    structures, targets = [], []
    for split in ("train", "val", "test"):
        structures += list(payload["splits"][split]["structures"])
        targets += list(payload["splits"][split]["targets"])
    records = []
    for i, (st, tg) in enumerate(zip(structures, targets)):
        records.append({
            ID_KEY: f"mb-ismetal-{i:06d}",
            TARGET_KEY: int(bool(tg)),
            "atoms": pmg_to_atoms(st).to_dict(),
        })
    out_dir = os.path.join(ALIGNN_DIR, "MP_json_is_metal")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "id_prop.json")
    with open(path, "w") as f:
        json.dump(records, f)
    labels = [r[TARGET_KEY] for r in records]
    n_metal = sum(labels)
    print(f"wrote {len(records)} records -> {path}")
    print(f"class balance: metal(1)={n_metal} ({100*n_metal/len(labels):.1f}%)  "
          f"non-metal(0)={len(labels)-n_metal} ({100*(len(labels)-n_metal)/len(labels):.1f}%)")


if __name__ == "__main__":
    main()
