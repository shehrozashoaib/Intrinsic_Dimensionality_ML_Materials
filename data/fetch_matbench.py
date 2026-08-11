#!/usr/bin/env python3
"""Fetch the Matbench tasks used in this project and save each as a gzipped JSON
list of {material_id, structure (pymatgen as_dict), target}. Mirrors the Zenodo
format so all datasets load the same way.

Tasks: matbench_dielectric, matbench_log_kvrh, matbench_phonons, matbench_mp_is_metal.
Requires: matbench==0.6 (in the CGCNN/ALIGNN environment).
"""
import argparse, gzip, json, os

TASKS = {
    "matbench_dielectric": "n",           # refractive index (target column name varies by version)
    "matbench_log_kvrh": "log10(K_VRH)",
    "matbench_phonons": "last phdos peak",
    "matbench_mp_is_metal": "is_metal",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    from matbench.bench import MatbenchBenchmark
    mb = MatbenchBenchmark(autoload=False, subset=list(TASKS))
    for task in mb.tasks:
        task.load()
        name = task.dataset_name
        recs = []
        # task.df has columns: structure (pymatgen Structure), <target>
        df = task.df
        target_col = [c for c in df.columns if c != "structure"][0]
        for idx, row in df.iterrows():
            recs.append({
                "material_id": str(idx),
                "structure": row["structure"].as_dict(),
                target_col: (bool(row[target_col]) if target_col == "is_metal"
                             else float(row[target_col])),
            })
        out = os.path.join(args.out_dir, f"{name}.json.gz")
        with gzip.open(out, "wt") as f:
            json.dump(recs, f)
        print(f"  {name}: {len(recs)} records -> {out}")


if __name__ == "__main__":
    main()
