import argparse
import json
import pickle
import sys
from pathlib import Path

import torch

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from cgcnn.data import CIFData


def _materialize_split(structures, targets, root_dir, max_num_nbr, radius, dmin, step):
    dataset = CIFData(
        structures,
        targets,
        root_dir=root_dir,
        max_num_nbr=max_num_nbr,
        radius=radius,
        dmin=dmin,
        step=step,
        random_seed=None,
    )
    return [dataset[i] for i in range(len(dataset))]


def main():
    parser = argparse.ArgumentParser(description="Materialize CGCNN graph tensors from raw Matbench splits.")
    parser.add_argument("--raw-dir", default="cgcnn/cached_matbench/raw", help="directory containing *_raw.pkl files")
    parser.add_argument("--output-dir", default="cgcnn/cached_matbench/tensors", help="directory for tensor caches")
    parser.add_argument("--root-dir", default="cgcnn/root_dir", help="directory containing atom_init.json")
    parser.add_argument("--tasks", nargs="*", default=None, help="optional task-name filter")
    parser.add_argument("--seed", default=123, type=int, help="seed used in raw split filename")
    parser.add_argument("--max-num-nbr", default=12, type=int)
    parser.add_argument("--radius", default=8.0, type=float)
    parser.add_argument("--dmin", default=0.0, type=float)
    parser.add_argument("--step", default=0.2, type=float)
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    wanted = set(args.tasks) if args.tasks else None

    raw_files = sorted(raw_dir.glob(f"*_seed{args.seed}_raw.pkl"))
    if not raw_files:
        raise FileNotFoundError(f"No raw split files found in {raw_dir} for seed {args.seed}")

    index = {}
    for raw_file in raw_files:
        with raw_file.open("rb") as f:
            payload = pickle.load(f)
        task = payload["task"]
        if wanted and task not in wanted:
            continue

        task_dir = out_dir / task / f"seed{payload['seed']}"
        task_dir.mkdir(parents=True, exist_ok=True)
        print(f"[tensor] materializing {task} -> {task_dir}")

        split_paths = {}
        split_counts = {}
        for split in ("train", "val", "test"):
            split_payload = payload["splits"][split]
            samples = _materialize_split(
                split_payload["structures"],
                split_payload["targets"],
                root_dir=args.root_dir,
                max_num_nbr=args.max_num_nbr,
                radius=args.radius,
                dmin=args.dmin,
                step=args.step,
            )
            metadata = {
                "task": task,
                "split": split,
                "seed": payload["seed"],
                "source_raw": str(raw_file),
                "root_dir": args.root_dir,
                "max_num_nbr": args.max_num_nbr,
                "radius": args.radius,
                "dmin": args.dmin,
                "step": args.step,
                "n_samples": len(samples),
            }
            split_path = task_dir / f"{split}.pt"
            torch.save({"samples": samples, "metadata": metadata}, split_path)
            split_paths[split] = str(split_path)
            split_counts[split] = len(samples)
            print(f"[tensor] wrote {split_path} ({len(samples)} samples)")

        task_meta = {
            "task": task,
            "seed": payload["seed"],
            "raw_path": str(raw_file),
            "splits": split_paths,
            "counts": split_counts,
            "max_num_nbr": args.max_num_nbr,
            "radius": args.radius,
            "dmin": args.dmin,
            "step": args.step,
        }
        with (task_dir / "metadata.json").open("w") as f:
            json.dump(task_meta, f, indent=2)
        index[task] = str(task_dir)

    if not index:
        raise ValueError(f"No raw files matched tasks={sorted(wanted)}")
    with (out_dir / f"index_seed{args.seed}.json").open("w") as f:
        json.dump(index, f, indent=2)
    print(f"[tensor] wrote index {out_dir / f'index_seed{args.seed}.json'}")


if __name__ == "__main__":
    main()
