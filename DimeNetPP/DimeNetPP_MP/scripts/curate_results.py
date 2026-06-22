#!/usr/bin/env python3
"""Curate the raw DimeNet++ sweep outputs into the GitHub `results/` layout.

Mirrors ALIGNN_MP/results/<exp>/ exactly:
    results/<exp>/
      README.md
      summary_mae_runtime.csv         (one row per run)
      predictions/                    (renamed per-run test prediction CSVs, all runs)
      logs/                           (one training log per dimension)

Reads each run's metadata.json under the raw results_dimenetpp_<task>_fastfood/ dirs.
Run from DimeNetPP_MP:  python scripts/curate_results.py
"""
import csv
import json
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parent

DATASETS = {
    "eform": {
        "results_root": "results_dimenetpp_eform_fastfood",
        "exp_dir": "results/eform_fastfood_formation_energy_350epochs",
        "task": "mp_formation_energy_per_atom",
        "prefix": "eform_fastfood",
        "pretty": "formation energy",
    },
    "bandgap": {
        "results_root": "results_dimenetpp_bandgap_fastfood",
        "exp_dir": "results/bandgap_fastfood_band_gap_350epochs",
        "task": "mp_band_gap",
        "prefix": "bandgap_fastfood",
        "pretty": "band gap",
    },
}

SUMMARY_FIELDS = [
    "task", "method", "dim_percent", "id_dim", "model_seed", "split_seed", "epochs",
    "batch_size", "best_val_mae", "test_mae", "duration_sec", "duration_hours",
    "start_time", "end_time", "prediction_csv",
]


def curate(ds):
    root = BASE / ds["results_root"]
    exp = BASE / ds["exp_dir"]
    pred_dir = exp / "predictions"
    log_dir = exp / "logs"
    pred_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    best_log_per_dim = {}  # dim% -> (model_seed, train.log path, run name)
    for meta in sorted(root.glob("*/metadata.json")):
        d = json.loads(meta.read_text())
        if d.get("status") != "success":
            print(f"  skip (status={d.get('status')}): {meta.parent.name}")
            continue
        run_dir = meta.parent
        dimp = int(d["id_dim_percent"])
        ms, ss, ep = d["model_seed"], d["split_seed"], d["epochs"]
        name = f"{ds['prefix']}_dim{dimp:03d}_modelseed{ms}_splitseed{ss}_epochs{ep}"

        # copy the per-run test prediction CSV (all runs)
        src = d.get("predictions_csv", "")
        if not (src and Path(src).exists()):
            cands = list(run_dir.glob("dimenetpp_test_predictions_*.csv"))
            src = str(cands[0]) if cands else ""
        if src and Path(src).exists():
            shutil.copyfile(src, pred_dir / f"{name}_predictions.csv")
        else:
            print(f"  WARNING: no prediction csv for {name}")

        # remember one training log per dimension (lowest model seed)
        tl = run_dir / "train.log"
        if tl.exists() and (dimp not in best_log_per_dim or ms < best_log_per_dim[dimp][0]):
            best_log_per_dim[dimp] = (ms, tl, name)

        dur = float(d.get("duration_sec") or 0)
        rows.append({
            "task": ds["task"], "method": "fastfood", "dim_percent": dimp,
            "id_dim": d.get("id_dim"), "model_seed": ms, "split_seed": ss,
            "epochs": ep, "batch_size": d.get("batch_size"),
            "best_val_mae": d.get("val_mae"), "test_mae": d.get("test_mae"),
            "duration_sec": int(dur), "duration_hours": round(dur / 3600, 4),
            "start_time": d.get("start_time"), "end_time": d.get("end_time"),
            "prediction_csv": f"predictions/{name}_predictions.csv",
        })

    # one training log per dimension
    for dimp, (ms, tl, name) in sorted(best_log_per_dim.items()):
        shutil.copyfile(tl, log_dir / f"{name}_train.log")

    rows.sort(key=lambda r: (r["dim_percent"], r["model_seed"]))
    with (exp / "summary_mae_runtime.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        w.writeheader()
        w.writerows(rows)

    dims = sorted({r["dim_percent"] for r in rows})
    (exp / "README.md").write_text(
        f"# DimeNet++ MP {ds['pretty']} Fastfood results\n\n"
        f"350-epoch DimeNet++ {ds['pretty']} Fastfood random-subspace sweep on the Materials "
        f"Project dataset (wrapper-v3, `dimenet_run_eform_v3.py`).\n\n"
        f"- `summary_mae_runtime.csv` — validation/test MAE (eV) and runtime for every "
        f"dimension/seed run ({len(rows)} runs).\n"
        f"- `predictions/` — per-crystal test-set predictions for each run, named by dimension, "
        f"model seed, split seed, and epoch count.\n"
        f"- `logs/` — one representative training log per intrinsic-dimension fraction "
        f"(lowest model seed), {len(dims)} dims: {', '.join(f'{d}%' for d in dims)}.\n\n"
        f"Intrinsic-dimension fractions swept: {', '.join(f'{d}%' for d in dims)}. "
        f"Seeds: model 123/456 (+789 for 45% and 60%), split 1123/1456/1789.\n"
    )
    print(f"{ds['results_root']}: {len(rows)} runs, {len(best_log_per_dim)} dim logs -> {exp}")


if __name__ == "__main__":
    for ds in DATASETS.values():
        curate(ds)
