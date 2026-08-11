#!/usr/bin/env python3
"""Build ALIGNN id_prop.json datasets for matbench_log_kvrh.

Source: the CGCNN raw split pickle
  CGCNN/CGCNN_Matbench/cached_matbench/raw/matbench_log_kvrh_seed123_raw.pkl
which holds pymatgen Structure objects + log_kvrh targets (10,987 total). We take
the FULL union (train+val+test of any seed is the same underlying matbench fold-0
dataset, just shuffled) and convert each Structure -> JARVIS atoms dict, the format
ALIGNN's train_alignn.py expects (records: {material_id, log_kvrh, atoms}).

Outputs (ALIGNN does its own internal 80/10/10 split via --split_seed):
  alignn/MP_json_log_kvrh/id_prop.json         full dataset (Exp 1)
  alignn/partitions/kvrh_half_1/id_prop.json   disjoint 50% half (Exp 2)
  alignn/partitions/kvrh_half_2/id_prop.json   disjoint 50% half (Exp 2)
"""
import json, os, pickle, random, warnings
warnings.filterwarnings("ignore")

from jarvis.core.atoms import pmg_to_atoms

ALIGNN_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_PKL = os.path.join(
    ALIGNN_DIR, "..", "..", "..", "CGCNN", "CGCNN_Matbench",
    "cached_matbench", "raw", "matbench_log_kvrh_seed123_raw.pkl",
)
TARGET_KEY = "log_kvrh"
ID_KEY = "material_id"
HALF_SHUFFLE_SEED = 0


def load_full_records():
    with open(RAW_PKL, "rb") as f:
        payload = pickle.load(f)
    assert payload["task"] == "matbench_log_kvrh", payload["task"]
    structures, targets = [], []
    for split in ("train", "val", "test"):
        structures += list(payload["splits"][split]["structures"])
        targets += list(payload["splits"][split]["targets"])
    records = []
    for i, (st, tg) in enumerate(zip(structures, targets)):
        atoms = pmg_to_atoms(st).to_dict()
        records.append({
            ID_KEY: f"mb-kvrh-{i:05d}",
            TARGET_KEY: float(tg),
            "atoms": atoms,
        })
    return records


def write_idprop(records, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "id_prop.json")
    with open(path, "w") as f:
        json.dump(records, f)
    print(f"  wrote {len(records):5d} records -> {path}")


def main():
    records = load_full_records()
    print(f"[full] {len(records)} records converted")
    write_idprop(records, os.path.join(ALIGNN_DIR, "MP_json_log_kvrh"))

    # Two disjoint 50% halves (data variation for Exp 2). Fixed shuffle so the
    # split is reproducible; ALIGNN then splits 80/10/10 WITHIN each half.
    idx = list(range(len(records)))
    random.Random(HALF_SHUFFLE_SEED).shuffle(idx)
    cut = len(idx) // 2
    halves = {
        "kvrh_half_1": [records[i] for i in idx[:cut]],
        "kvrh_half_2": [records[i] for i in idx[cut:]],
    }
    part_root = os.path.join(ALIGNN_DIR, "partitions")
    for name, recs in halves.items():
        write_idprop(recs, os.path.join(part_root, name))

    # manifest for provenance
    manifest = {
        "task": "matbench_log_kvrh",
        "source_pkl": os.path.relpath(RAW_PKL, ALIGNN_DIR),
        "n_total": len(records),
        "half_shuffle_seed": HALF_SHUFFLE_SEED,
        "n_half_1": len(halves["kvrh_half_1"]),
        "n_half_2": len(halves["kvrh_half_2"]),
    }
    with open(os.path.join(part_root, "kvrh_halves_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[halves] {manifest['n_half_1']} + {manifest['n_half_2']} (disjoint, seed={HALF_SHUFFLE_SEED})")


if __name__ == "__main__":
    main()
