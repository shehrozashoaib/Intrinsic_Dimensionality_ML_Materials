#!/usr/bin/env python3
"""Build ALIGNN id_prop.json for matbench_phonons (last phonon-DOS peak, cm^-1).

Source: CGCNN raw split pickle
  CGCNN/CGCNN_Matbench/cached_matbench/raw/matbench_phonons_seed123_raw.pkl
which holds pymatgen Structure objects + phonons targets. Take the FULL union (the per-seed
splits are the same underlying matbench fold-0 set) and convert each Structure -> JARVIS atoms
dict (records: {material_id, phonons, atoms}). ALIGNN does its own 80/10/10 split via --split_seed.

Output: alignn/MP_json_phonons/id_prop.json
"""
import json, os, pickle, warnings
warnings.filterwarnings("ignore")
from jarvis.core.atoms import pmg_to_atoms

ALIGNN_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_PKL = os.path.join(ALIGNN_DIR, "..", "..", "..", "CGCNN", "CGCNN_Matbench",
                       "cached_matbench", "raw", "matbench_phonons_seed123_raw.pkl")
TARGET_KEY = "phonons"
ID_KEY = "material_id"


def main():
    with open(RAW_PKL, "rb") as f:
        payload = pickle.load(f)
    assert payload["task"] == "matbench_phonons", payload["task"]
    structures, targets = [], []
    for split in ("train", "val", "test"):
        structures += list(payload["splits"][split]["structures"])
        targets += list(payload["splits"][split]["targets"])
    records = []
    for i, (st, tg) in enumerate(zip(structures, targets)):
        records.append({
            ID_KEY: f"mb-phon-{i:05d}",
            TARGET_KEY: float(tg),
            "atoms": pmg_to_atoms(st).to_dict(),
        })
    out_dir = os.path.join(ALIGNN_DIR, "MP_json_phonons")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "id_prop.json")
    with open(path, "w") as f:
        json.dump(records, f)
    tg = [r[TARGET_KEY] for r in records]
    print(f"wrote {len(records)} records -> {path}")
    print(f"target range: min={min(tg):.2f} max={max(tg):.2f} mean={sum(tg)/len(tg):.2f}")


if __name__ == "__main__":
    main()
