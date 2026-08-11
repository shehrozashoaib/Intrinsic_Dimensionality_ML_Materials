#!/usr/bin/env python3
"""Summarize DimeNet++ convparam validation-loss trends from train.log files."""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path


EPOCH_RE = re.compile(r"^Epoch\s+(\d+)/(\d+)")
METRIC_RE = re.compile(
    r"loss:\s*([-+0-9.eE]+)\s*-\s*val_loss:\s*([-+0-9.eE]+)"
)
RUN_RE = re.compile(
    r"ies(?P<ies>\d+)_dim(?P<dim>\d+)_modelseed(?P<model>\d+)_splitseed(?P<split>\d+)"
)


def parse_train_log(path: Path) -> tuple[int | None, int | None, list[tuple[int, float, float]]]:
    current_epoch = None
    total_epochs = None
    rows: list[tuple[int, float, float]] = []
    for line in path.read_text(errors="replace").splitlines():
        epoch_match = EPOCH_RE.match(line)
        if epoch_match:
            current_epoch = int(epoch_match.group(1))
            total_epochs = int(epoch_match.group(2))
            continue
        metric_match = METRIC_RE.search(line)
        if metric_match and current_epoch is not None:
            loss = float(metric_match.group(1))
            val_loss = float(metric_match.group(2))
            if math.isfinite(loss) and math.isfinite(val_loss):
                rows.append((current_epoch, loss, val_loss))
    return current_epoch, total_epochs, rows


def run_label(path: Path) -> str:
    match = RUN_RE.search(path.parent.name)
    if not match:
        return path.parent.name
    return (
        f"ies{match.group('ies')} dim{match.group('dim')} "
        f"seed{match.group('model')}:{match.group('split')}"
    )


def summarize(root: Path, tail: int, late_window: int) -> list[dict[str, object]]:
    summaries = []
    for log_path in sorted(root.glob("*/train.log")):
        last_epoch, total_epochs, rows = parse_train_log(log_path)
        if not rows:
            continue
        best_epoch, _, best_val = min(rows, key=lambda item: item[2])
        final_epoch, _, final_val = rows[-1]
        tail_rows = rows[-tail:]
        tail_best_epoch, _, tail_best_val = min(tail_rows, key=lambda item: item[2])
        tail_first = tail_rows[0][2]
        tail_delta = final_val - tail_first
        late_best = bool(total_epochs and best_epoch >= max(1, total_epochs - late_window + 1))
        summaries.append(
            {
                "label": run_label(log_path),
                "done": bool(total_epochs and final_epoch >= total_epochs),
                "final_epoch": final_epoch,
                "total_epochs": total_epochs,
                "best_epoch": best_epoch,
                "best_val": best_val,
                "final_val": final_val,
                "tail_best_epoch": tail_best_epoch,
                "tail_best_val": tail_best_val,
                "tail_delta": tail_delta,
                "late_best": late_best,
            }
        )
    return summaries


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("results_dimenetpp_log_kvrh_convparam_onecycle_fastfood"),
    )
    parser.add_argument("--tail", type=int, default=20)
    parser.add_argument("--late-window", type=int, default=20)
    args = parser.parse_args()

    rows = summarize(args.root, args.tail, args.late_window)
    if not rows:
        print(f"No train.log files with parsed val_loss rows found under {args.root}")
        return 0

    completed = [row for row in rows if row["done"]]
    active = [row for row in rows if not row["done"]]
    late = [row for row in completed if row["late_best"]]
    improving_tail = [row for row in completed if float(row["tail_delta"]) < 0]

    print(
        f"runs parsed={len(rows)} completed={len(completed)} active={len(active)} "
        f"completed_late_best={len(late)} completed_tail_down={len(improving_tail)}"
    )
    print(
        "label,status,epoch,best_epoch,best_val,final_val,"
        f"tail{args.tail}_best_epoch,tail{args.tail}_delta,late_best"
    )
    for row in sorted(rows, key=lambda item: (str(item["label"]), int(item["final_epoch"]))):
        status = "done" if row["done"] else "active"
        total = row["total_epochs"] or "?"
        print(
            f"{row['label']},{status},{row['final_epoch']}/{total},"
            f"{row['best_epoch']},{float(row['best_val']):.6g},"
            f"{float(row['final_val']):.6g},{row['tail_best_epoch']},"
            f"{float(row['tail_delta']):+.6g},{row['late_best']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
