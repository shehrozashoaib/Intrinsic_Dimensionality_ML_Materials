#!/usr/bin/env python3
"""Curate the raw DimeNet++ matbench log_kvrh sweeps into the GitHub `results/` layout.

Handles both the clipped (clipnorm=1.0) and the no-clip (clipnorm=0) ablation trees, one
curated exp per layer count (num_blocks):
    results/log_kvrh_fastfood_<N>layer_<E>epochs[_noclip]/
      README.md
      summary_mae_runtime.csv         (one row per run; includes num_blocks)
      predictions/                    (renamed per-run test prediction CSVs, all runs)
      logs/                           (one training log per dimension, lowest model seed)

Reads each run's metadata.json and groups by num_blocks.
Run from DimeNetPP_Matbench:  python scripts/curate_kvrh_results.py
"""
import csv
import json
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parent

# (raw results root, curated exp-name suffix, gradient-clipping description)
CONFIGS = [
    ("results_dimenetpp_log_kvrh_fastfood", "", "gradient clipping `clipnorm=1.0`"),
    ("results_dimenetpp_log_kvrh_noclip_fastfood", "_noclip", "NO gradient clipping (`clipnorm=0`)"),
]

SUMMARY_FIELDS = [
    "task", "method", "num_blocks", "dim_percent", "id_dim", "model_seed", "split_seed",
    "epochs", "batch_size", "best_val_mae", "test_mae", "duration_sec", "duration_hours",
    "start_time", "end_time", "prediction_csv",
]


def curate_root(raw_root, suffix, clip_desc):
    root = BASE / raw_root
    if not root.is_dir():
        print(f"  (skip, no dir: {raw_root})")
        return
    by_nb = {}
    for meta in sorted(root.glob("*/metadata.json")):
        d = json.loads(meta.read_text())
        if d.get("status") != "success":
            print(f"  skip (status={d.get('status')}): {meta.parent.name}")
            continue
        by_nb.setdefault(int(d["num_blocks"]), []).append((meta.parent, d))

    for nb, runs in sorted(by_nb.items()):
        ep = runs[0][1]["epochs"]
        exp = BASE / "results" / f"log_kvrh_fastfood_{nb}layer_{ep}epochs{suffix}"
        pred_dir = exp / "predictions"
        log_dir = exp / "logs"
        pred_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)

        rows = []
        best_log_per_dim = {}
        for run_dir, d in runs:
            dimp = int(d["id_dim_percent"])
            ms, ss = d["model_seed"], d["split_seed"]
            name = f"log_kvrh_fastfood_nb{nb}{suffix}_dim{dimp:03d}_modelseed{ms}_splitseed{ss}_epochs{ep}"

            src = d.get("predictions_csv", "")
            if not (src and Path(src).exists()):
                cands = list(run_dir.glob("dimenetpp_test_predictions_*.csv"))
                src = str(cands[0]) if cands else ""
            if src and Path(src).exists():
                shutil.copyfile(src, pred_dir / f"{name}_predictions.csv")
            else:
                print(f"  WARNING: no prediction csv for {name}")

            tl = run_dir / "train.log"
            if tl.exists() and (dimp not in best_log_per_dim or ms < best_log_per_dim[dimp][0]):
                best_log_per_dim[dimp] = (ms, tl, name)

            dur = float(d.get("duration_sec") or 0)
            rows.append({
                "task": "matbench_log_kvrh", "method": "fastfood", "num_blocks": nb,
                "dim_percent": dimp, "id_dim": d.get("id_dim"), "model_seed": ms,
                "split_seed": ss, "epochs": ep, "batch_size": d.get("batch_size"),
                "best_val_mae": d.get("val_mae"), "test_mae": d.get("test_mae"),
                "duration_sec": int(dur), "duration_hours": round(dur / 3600, 4),
                "start_time": d.get("start_time"), "end_time": d.get("end_time"),
                "prediction_csv": f"predictions/{name}_predictions.csv",
            })

        for dimp, (ms, tl, name) in sorted(best_log_per_dim.items()):
            shutil.copyfile(tl, log_dir / f"{name}_train.log")

        rows.sort(key=lambda r: (r["dim_percent"], r["model_seed"]))
        with (exp / "summary_mae_runtime.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
            w.writeheader()
            w.writerows(rows)

        dims = sorted({r["dim_percent"] for r in rows}, reverse=True)
        clip_tag = " (no gradient clipping — ablation)" if suffix else ""
        (exp / "README.md").write_text(
            f"# DimeNet++ matbench log_kvrh Fastfood results — {nb} layers ({nb} interaction blocks){clip_tag}\n\n"
            f"{ep}-epoch DimeNet++ log_kvrh Fastfood random-subspace sweep on the full matbench "
            f"log_kvrh dataset, with `num_blocks={nb}` (wrapper-v3, `dimenet_run_kvrh_v3.py`, "
            f"{clip_desc}).\n\n"
            f"- `summary_mae_runtime.csv` — validation/test MAE and runtime for every dimension/seed "
            f"run ({len(rows)} runs = {len(dims)} dims x 3 seeds).\n"
            f"- `predictions/` — per-crystal test-set predictions for each run, named by num_blocks, "
            f"dimension, model seed, split seed, and epoch count.\n"
            f"- `logs/` — one representative training log per intrinsic-dimension fraction "
            f"(lowest model seed), {len(dims)} dims: {', '.join(f'{x}%' for x in dims)}.\n\n"
            f"Intrinsic-dimension fractions swept: {', '.join(f'{x}%' for x in dims)}. "
            f"Seeds: model/split = 123/1123, 456/1456, 789/1789.\n"
            + ("\n**Ablation note:** this is the NO-CLIPPING counterpart of "
               f"`log_kvrh_fastfood_{nb}layer_*epochs/`. Without `clipnorm`, the deeper model's "
               "random-init loss spike is not tamed and the sweep is badly under-trained "
               "(markedly worse test MAE at every dimension; the effect is severe at 4 layers).\n"
               if suffix else "")
        )
        print(f"[{raw_root}] nb={nb}: {len(rows)} runs, {len(best_log_per_dim)} dim logs -> {exp.name}")


def main():
    for raw_root, suffix, clip_desc in CONFIGS:
        curate_root(raw_root, suffix, clip_desc)


if __name__ == "__main__":
    main()
