import argparse
import json
import pickle
import random
from pathlib import Path

from matbench.bench import MatbenchBenchmark

DEFAULT_TASKS = ("matbench_phonons", "matbench_dielectric", "matbench_log_kvrh")


def _as_list(values):
    if hasattr(values, "tolist"):
        return values.tolist()
    return list(values)


def _split_indices(n_items, seed, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-8:
        raise ValueError("train/val/test ratios must sum to 1.0")
    indices = list(range(n_items))
    random.Random(seed).shuffle(indices)
    n_train = int(train_ratio * n_items)
    n_val = int(val_ratio * n_items)
    n_test = n_items - n_train - n_val
    return {
        "train": indices[:n_train],
        "val": indices[n_train:n_train + n_val],
        "test": indices[n_train + n_val:n_train + n_val + n_test],
    }


def _select_tasks(benchmark, names):
    wanted = set(names)
    tasks = [task for task in benchmark.tasks if task.dataset_name in wanted]
    found = {task.dataset_name for task in tasks}
    missing = sorted(wanted - found)
    if missing:
        raise ValueError(f"Matbench task(s) not found: {missing}")
    return tasks


def main():
    parser = argparse.ArgumentParser(description="Create deterministic 80/10/10 CGCNN Matbench raw splits.")
    parser.add_argument("--output-dir", default="cgcnn/cached_matbench/raw", help="directory for raw split .pkl files")
    parser.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS), help="Matbench task names to export")
    parser.add_argument("--fold", default=0, type=int, help="Matbench fold used to recover full data with targets")
    parser.add_argument("--seed", default=123, type=int, help="shuffle seed")
    parser.add_argument("--train-ratio", default=0.8, type=float)
    parser.add_argument("--val-ratio", default=0.1, type=float)
    parser.add_argument("--test-ratio", default=0.1, type=float)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mb = MatbenchBenchmark(autoload=False)
    for task in _select_tasks(mb, args.tasks):
        if args.fold not in task.folds:
            raise ValueError(f"Fold {args.fold} is not available for {task.dataset_name}; options={task.folds}")

        print(f"[raw] loading {task.dataset_name} fold={args.fold}")
        task.load()
        train_inputs, train_outputs = task.get_train_and_val_data(args.fold)
        test_inputs, test_outputs = task.get_test_data(args.fold, include_target=True)

        structures = _as_list(train_inputs) + _as_list(test_inputs)
        targets = _as_list(train_outputs) + _as_list(test_outputs)
        if len(structures) != len(targets):
            raise ValueError(f"Mismatched input/target lengths for {task.dataset_name}")

        split_indices = _split_indices(
            len(structures), args.seed, args.train_ratio, args.val_ratio, args.test_ratio
        )
        payload = {
            "task": task.dataset_name,
            "fold_source": args.fold,
            "seed": args.seed,
            "ratios": {
                "train": args.train_ratio,
                "val": args.val_ratio,
                "test": args.test_ratio,
            },
            "splits": {},
        }
        for split, indices in split_indices.items():
            payload["splits"][split] = {
                "structures": [structures[i] for i in indices],
                "targets": [targets[i] for i in indices],
            }

        pkl_path = out_dir / f"{task.dataset_name}_seed{args.seed}_raw.pkl"
        with pkl_path.open("wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

        meta = {
            "task": task.dataset_name,
            "source_fold": args.fold,
            "seed": args.seed,
            "n_total": len(structures),
            "n_train": len(split_indices["train"]),
            "n_val": len(split_indices["val"]),
            "n_test": len(split_indices["test"]),
            "raw_path": str(pkl_path),
        }
        with (out_dir / f"{task.dataset_name}_seed{args.seed}_raw.json").open("w") as f:
            json.dump(meta, f, indent=2)
        print(f"[raw] wrote {pkl_path} ({meta['n_train']}/{meta['n_val']}/{meta['n_test']})")


if __name__ == "__main__":
    main()
