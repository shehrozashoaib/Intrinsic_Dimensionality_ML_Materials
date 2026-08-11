#!/usr/bin/env python3
"""Janitor: delete oversized DimeNet++ training artifacts as runs finish.

WHY THIS EXISTS
---------------
The dense wrapper stores a real D x d projection matrix. With --restore_best, Keras'
ModelCheckpoint serialised EVERY weight -- including that frozen P -- which is 28 GB per save
at dim-100% (D = 83,685). Nineteen of those filled the disk and killed the first wrapper-variant
sweep at epoch ~172/180. The runner now keeps only the trainable weights in memory, so this
should not recur; this janitor is the safety net, and it also sweeps up the orthonormal Q cache
and any stray weight blobs from older code paths.

WHAT IT DELETES  (only inside the roots listed in ROOTS)
    *.weights.h5, *.h5, *.pt, *.pth, *.pth.tar, weights_*.npy, Q_rows*.npy
    ... and only when the file is larger than --min-mb.

WHAT IT NEVER TOUCHES
    metadata.json, *summary*.csv, train.log, history*.json, prediction/test-results csv,
    z_*.npy (the trainable vector -- small, and the only thing needed to rebuild a model).

SAFETY
    * --dry-run (default) prints and deletes nothing. Pass --apply to actually remove.
    * By default a run directory is only cleaned once it looks FINISHED: it has a metadata.json,
      or its train.log has not been modified for --idle-min minutes. --include-running overrides.
    * Never follows symlinks; never recurses outside ROOTS.

USAGE
    python cleanup_large_artifacts.py                      # one dry-run pass
    python cleanup_large_artifacts.py --apply              # one real pass
    python cleanup_large_artifacts.py --apply --loop 300   # run alongside training, every 5 min
"""
import argparse, fnmatch, os, sys, time

BASE = os.path.dirname(os.path.abspath(__file__))
ROOTS = [
    os.path.join(BASE, "DimeNetPP_Matbench", "results_dimenetpp_kvrh_wrapper_variants"),
    os.path.join(BASE, "DimeNetPP_Matbench", "orthonormal_q_cache_v3"),
    os.path.join(BASE, "DimeNetPP_MP", "results_dimenetpp_is_metal_onecycle_fastfood"),
    os.path.join(BASE, "DimeNetPP_Matbench", "results_dimenetpp_onecycle_fixedtest"),
    os.path.join(BASE, "DimeNetPP_MP", "results_dimenetpp_onecycle_fixedtest"),
]
KILL = ["*.weights.h5", "*.h5", "*.pt", "*.pth", "*.pth.tar", "weights_*.npy", "Q_rows*.npy"]
KEEP = ["metadata.json", "*summary*.csv", "train.log", "history*.json", "z_*.npy",
        "*prediction*.csv", "*test_results*.csv", "*.out", "*.err"]


def protected(name):
    return any(fnmatch.fnmatch(name, k) for k in KEEP)


def targeted(name):
    return any(fnmatch.fnmatch(name, k) for k in KILL)


def run_finished(run_dir, idle_min):
    """A run is 'finished' if metadata.json exists, or train.log has gone quiet."""
    if os.path.exists(os.path.join(run_dir, "metadata.json")):
        return True
    log = os.path.join(run_dir, "train.log")
    if os.path.exists(log):
        return (time.time() - os.path.getmtime(log)) > idle_min * 60
    return False   # no log and no metadata -> probably just started; leave it alone


def sweep(min_mb, apply_, idle_min, include_running, quiet=False):
    freed = 0
    hits = 0
    for root in ROOTS:
        if not os.path.isdir(root):
            continue
        for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
            # the Q cache is a flat dir of standalone .npy files, not per-run dirs
            is_qcache = os.path.basename(root).startswith("orthonormal_q_cache")
            if not is_qcache and not include_running:
                # only clean inside a run dir that has finished
                if any(f in ("metadata.json", "train.log") for f in filenames):
                    if not run_finished(dirpath, idle_min):
                        continue
            for fn in filenames:
                if protected(fn) or not targeted(fn):
                    continue
                fp = os.path.join(dirpath, fn)
                try:
                    if os.path.islink(fp):
                        continue
                    sz = os.path.getsize(fp)
                except OSError:
                    continue
                if sz < min_mb * 1e6:
                    continue
                hits += 1
                freed += sz
                if not quiet:
                    print(f"  {'DELETE' if apply_ else 'would delete'}  {sz/1e9:7.2f} GB  {fp}")
                if apply_:
                    try:
                        os.remove(fp)
                    except OSError as e:
                        print(f"  !! failed to remove {fp}: {e}", file=sys.stderr)
    return hits, freed


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true", help="actually delete (default: dry run)")
    ap.add_argument("--min-mb", type=float, default=100.0, help="only files larger than this")
    ap.add_argument("--idle-min", type=float, default=20.0,
                    help="treat a run as finished if train.log is this many minutes stale")
    ap.add_argument("--include-running", action="store_true",
                    help="also clean run dirs that still look active (NOT recommended)")
    ap.add_argument("--loop", type=float, default=0.0,
                    help="repeat every N seconds (0 = single pass)")
    a = ap.parse_args()

    while True:
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        hits, freed = sweep(a.min_mb, a.apply, a.idle_min, a.include_running)
        verb = "freed" if a.apply else "reclaimable"
        print(f"[{stamp}] {hits} file(s), {freed/1e9:.2f} GB {verb}"
              + ("" if a.apply else "   (dry run -- pass --apply to delete)"), flush=True)
        if a.loop <= 0:
            break
        time.sleep(a.loop)


if __name__ == "__main__":
    main()
