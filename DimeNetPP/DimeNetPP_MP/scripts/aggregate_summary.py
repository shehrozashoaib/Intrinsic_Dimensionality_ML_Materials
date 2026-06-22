"""Aggregate per-run metadata.json files in a results dir into one summary CSV.

Array-mode sweeps write each run's metadata.json but not the shared summary (to avoid
concurrent-append races). Run this to (re)build the summary at any time:

  python scripts/aggregate_summary.py results_dimenetpp_eform_fastfood    results_dimenetpp_eform_fastfood/dimenetpp_eform_fastfood_summary.csv
  python scripts/aggregate_summary.py results_dimenetpp_bandgap_fastfood  results_dimenetpp_bandgap_fastfood/dimenetpp_bandgap_fastfood_summary.csv
"""
import csv
import glob
import json
import os
import sys

FIELDS = [
    "task", "target_key", "wrapper", "id_dim", "id_dim_percent", "model_seed", "split_seed",
    "epochs", "batch_size", "val_mae", "test_mae", "duration_sec", "status", "exit_code",
    "start_time", "end_time", "output_dir", "predictions_csv", "history_json",
]


def _f(x):
    try:
        return f"{float(x):.4f}"
    except (TypeError, ValueError):
        return str(x)


def main():
    results_dir = sys.argv[1]
    out_csv = sys.argv[2]

    rows = []
    for mp in glob.glob(os.path.join(results_dir, "*", "metadata.json")):
        try:
            with open(mp) as f:
                rows.append(json.load(f))
        except Exception as e:
            print(f"skip {mp}: {e}")

    def key(d):
        return (-int(d.get("id_dim_percent", 0)), int(d.get("model_seed", 0)), int(d.get("split_seed", 0)))

    rows.sort(key=key)

    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        for d in rows:
            w.writerow(d)

    n_ok = sum(1 for d in rows if d.get("status") == "success")
    print(f"Wrote {out_csv}: {len(rows)} runs ({n_ok} success)")
    print(f"{'dim%':>5} {'mseed':>6} {'sseed':>6} {'val_mae':>9} {'test_mae':>9} {'status':>8} {'hrs':>5}")
    for d in rows:
        dur = d.get("duration_sec") or 0
        print(f"{str(d.get('id_dim_percent','')):>5} {str(d.get('model_seed','')):>6} "
              f"{str(d.get('split_seed','')):>6} {_f(d.get('val_mae')):>9} {_f(d.get('test_mae')):>9} "
              f"{str(d.get('status','')):>8} {dur/3600:>5.2f}")


if __name__ == "__main__":
    main()
