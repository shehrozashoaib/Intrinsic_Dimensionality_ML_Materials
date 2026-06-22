"""Aggregate a DimeNet++ sweep's per-run metadata.json into one summary CSV.

Completed runs are read from metadata.json (final val_mae/test_mae, duration, status, times).
In-flight runs (train.log present, no metadata yet) are reported with their current epoch.
Re-run any time to refresh. Pure stdlib.
"""
import argparse
import csv
import glob
import json
import os
import re


def parse_dirname(name):
    m = re.search(r"dim(\d+)_modelseed(\d+)_splitseed(\d+)_epochs(\d+)", name)
    if not m:
        return {}
    return {
        "id_dim_percent": int(m.group(1)),
        "model_seed": int(m.group(2)),
        "split_seed": int(m.group(3)),
        "epochs": int(m.group(4)),
    }


def latest_epoch(logpath):
    ep = total = None
    try:
        with open(logpath, errors="replace") as f:
            txt = f.read()
    except OSError:
        return None, None
    for m in re.finditer(r"Epoch (\d+)/(\d+)", txt):
        ep, total = int(m.group(1)), int(m.group(2))
    return ep, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--task", default="")
    ap.add_argument("--epochs", type=int, default=350,
                    help="Only include runs with this epoch count (excludes smoke/bench dirs). 0 = all.")
    args = ap.parse_args()

    rows = []
    for d in sorted(glob.glob(os.path.join(args.results_root, "*epochs*"))):
        if not os.path.isdir(d):
            continue
        name = os.path.basename(d)
        info = parse_dirname(name)
        if args.epochs and info.get("epochs") != args.epochs:
            continue
        row = {
            "task": args.task,
            "id_dim_percent": info.get("id_dim_percent"),
            "id_dim": (info["id_dim_percent"] / 100) if "id_dim_percent" in info else "",
            "model_seed": info.get("model_seed"),
            "split_seed": info.get("split_seed"),
            "epochs": info.get("epochs"),
            "val_mae": "", "test_mae": "", "duration_sec": "", "duration_min": "",
            "status": "", "current_epoch": "", "start_time": "", "end_time": "", "run": name,
        }
        meta_path = os.path.join(d, "metadata.json")
        if os.path.exists(meta_path):
            try:
                m = json.load(open(meta_path))
                row.update(
                    id_dim_percent=m.get("id_dim_percent", row["id_dim_percent"]),
                    model_seed=m.get("model_seed", row["model_seed"]),
                    split_seed=m.get("split_seed", row["split_seed"]),
                    epochs=m.get("epochs", row["epochs"]),
                    val_mae=m.get("val_mae", ""), test_mae=m.get("test_mae", ""),
                    duration_sec=m.get("duration_sec", ""),
                    status=m.get("status", ""),
                    start_time=m.get("start_time", ""), end_time=m.get("end_time", ""),
                )
                row["id_dim"] = (row["id_dim_percent"] / 100) if row["id_dim_percent"] is not None else ""
                try:
                    row["duration_min"] = round(float(m.get("duration_sec")) / 60, 1)
                except (TypeError, ValueError):
                    pass
            except (json.JSONDecodeError, OSError):
                row["status"] = "metadata_parse_error"
        else:
            ep, total = latest_epoch(os.path.join(d, "train.log"))
            row["status"] = "running" if ep else "starting"
            row["current_epoch"] = f"{ep}/{total}" if ep else ""
        rows.append(row)

    rows.sort(key=lambda r: (-(r["id_dim_percent"] or 0), r["model_seed"] or 0, r["split_seed"] or 0))
    cols = ["task", "id_dim", "id_dim_percent", "model_seed", "split_seed", "epochs",
            "val_mae", "test_mae", "duration_sec", "duration_min", "status",
            "current_epoch", "start_time", "end_time", "run"]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    done = sum(1 for r in rows if r["status"] == "success")
    failed = sum(1 for r in rows if r["status"] == "failed")
    print(f"Wrote {args.out}")
    print(f"{len(rows)} run dirs | {done} success | {failed} failed | "
          f"{len(rows)-done-failed} in-flight")
    for r in rows:
        mae = f"test_mae={r['test_mae']:.4f}" if isinstance(r["test_mae"], float) else f"test_mae={r['test_mae']}"
        dur = f"{r['duration_min']}min" if r["duration_min"] != "" else (r["current_epoch"] or "")
        print(f"  dim={r['id_dim_percent']}% seed={r['model_seed']}/{r['split_seed']} "
              f"[{r['status']}] {mae} {dur}")


if __name__ == "__main__":
    main()
