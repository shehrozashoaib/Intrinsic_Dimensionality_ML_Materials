#!/usr/bin/env python3
"""Collect every results_alignn_eform_fastfood/*/metadata.json into one CSV.

Use after the job array (Path B) finishes, since array tasks write their own
metadata.json but do not append to the shared summary (to avoid write races).

    python scripts/aggregate_eform_summary.py
"""
import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULT_ROOT = HERE.parent / "results_alignn_eform_fastfood"
SUMMARY_CSV = RESULT_ROOT / "alignn_eform_fastfood_summary.csv"

FIELDS = [
    "task", "target_key", "wrapper", "id_dim", "id_dim_percent", "data_seed",
    "model_seed", "split_seed", "epochs", "batch_size", "root_dir", "output_dir",
    "predictions_csv", "history_json", "best_val_mae", "test_mae", "duration_sec",
    "status", "exit_code", "start_time", "end_time",
]

rows = []
for meta in sorted(RESULT_ROOT.glob("*/metadata.json")):
    try:
        rows.append(json.loads(meta.read_text()))
    except Exception as e:
        print(f"skip {meta}: {e}")

rows.sort(key=lambda r: (r.get("id_dim_percent", 0), r.get("model_seed", 0)))

with SUMMARY_CSV.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=FIELDS)
    writer.writeheader()
    for r in rows:
        writer.writerow({k: r.get(k, "") for k in FIELDS})

print(f"Wrote {len(rows)} runs to {SUMMARY_CSV}")
ok = sum(1 for r in rows if r.get("status") == "success")
print(f"  success={ok}  failed={len(rows) - ok}")
