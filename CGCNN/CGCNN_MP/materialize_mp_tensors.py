"""Build CGCNN graph-tensor caches for the MP eform / band-gap datasets.

Mirrors the Matbench materializer but reads the same MP sources used for DimeNet++/ALIGNN:
  --source eform_idprop : ALIGNN id_prop.json (JARVIS atoms),    target formation_energy_per_atom
  --source mp21_jsonl   : mp21.jsonl (pymatgen Structure dicts), target band_gap

Graphs are built ONCE for all structures (optionally with a spawn process pool), then reshuffled
into 80/10/10 train/val/test for each split seed and saved as <out>/<task>/seed<seed>/{train,val,test}.pt
in TensorCIFData format ({"samples": [...], "metadata": {...}}).
"""
import argparse
import json
import multiprocessing as mp
import random
import time
from pathlib import Path

import numpy as np
import torch

THIS_DIR = Path(__file__).resolve().parent
ATOM_INIT = THIS_DIR / "root_dir" / "atom_init.json"


def load_records(source, input_path, limit=0):
    """Return (struct_dicts (pymatgen as_dict), targets list) in file order."""
    sdicts, targets = [], []
    if source == "mp21_jsonl":
        with open(input_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if r.get("band_gap") is None or r.get("structure") is None:
                    continue
                sdicts.append(r["structure"])
                targets.append(float(r["band_gap"]))
                if limit and len(sdicts) >= limit:
                    break
    elif source == "eform_idprop":
        from jarvis.core.atoms import Atoms
        with open(input_path) as f:
            rows = json.load(f)
        for r in rows:
            if r.get("formation_energy_per_atom") is None or r.get("atoms") is None:
                continue
            sdicts.append(Atoms.from_dict(r["atoms"]).pymatgen_converter().as_dict())
            targets.append(float(r["formation_energy_per_atom"]))
            if limit and len(sdicts) >= limit:
                break
    else:
        raise ValueError(source)
    return sdicts, targets


def build_one(payload):
    """Build a CGCNN graph sample for one structure. Returns ((atom_fea, nbr_fea, nbr_fea_idx), target)."""
    sdict, target, max_num_nbr, radius, dmin, step = payload
    from pymatgen.core.structure import Structure
    from cgcnn.data import AtomCustomJSONInitializer, GaussianDistance
    global _ARI, _GDF
    try:
        _ARI
    except NameError:
        _ARI = AtomCustomJSONInitializer(str(ATOM_INIT))
        _GDF = GaussianDistance(dmin=dmin, dmax=radius, step=step)
    crystal = Structure.from_dict(sdict)
    atom_fea = np.vstack([_ARI.get_atom_fea(crystal[i].specie.number) for i in range(len(crystal))])
    all_nbrs = crystal.get_all_neighbors(radius, include_index=True)
    all_nbrs = [sorted(n, key=lambda x: x[1]) for n in all_nbrs]
    nbr_fea_idx, nbr_fea = [], []
    for nbr in all_nbrs:
        if len(nbr) < max_num_nbr:
            nbr_fea_idx.append(list(map(lambda x: x[2], nbr)) + [0] * (max_num_nbr - len(nbr)))
            nbr_fea.append(list(map(lambda x: x[1], nbr)) + [radius + 1.0] * (max_num_nbr - len(nbr)))
        else:
            nbr_fea_idx.append(list(map(lambda x: x[2], nbr[:max_num_nbr])))
            nbr_fea.append(list(map(lambda x: x[1], nbr[:max_num_nbr])))
    nbr_fea = _GDF.expand(np.array(nbr_fea))
    # Return plain numpy (pickled by value through the pool) instead of torch
    # tensors -- torch tensors returned from a spawn Pool go through fd-based
    # shared memory, which exhausts mmap/fd limits for ~150k structures.
    sample = (
        np.asarray(atom_fea, dtype=np.float32),
        np.asarray(nbr_fea, dtype=np.float32),
        np.asarray(nbr_fea_idx, dtype=np.int64),
    )
    return sample, float(target)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True, choices=["eform_idprop", "mp21_jsonl"])
    p.add_argument("--input", required=True)
    p.add_argument("--task-name", required=True, help="e.g. mp_eform or mp_bandgap")
    p.add_argument("--seeds", default="1123,1456,1789")
    p.add_argument("--output-dir", default="cached_mp/tensors")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--max-num-nbr", type=int, default=12)
    p.add_argument("--radius", type=float, default=8.0)
    p.add_argument("--dmin", type=float, default=0.0)
    p.add_argument("--step", type=float, default=0.2)
    args = p.parse_args()

    t0 = time.time()
    sdicts, targets = load_records(args.source, args.input, args.limit)
    N = len(sdicts)
    print(f"Loaded {N} records in {time.time()-t0:.1f}s", flush=True)

    payloads = [(sdicts[i], targets[i], args.max_num_nbr, args.radius, args.dmin, args.step) for i in range(N)]
    tb = time.time()
    if args.workers > 1:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=args.workers) as pool:
            raw = pool.map(build_one, payloads, chunksize=64)
    else:
        raw = [build_one(pl) for pl in payloads]
    print(f"Built {len(raw)} graphs in {time.time()-tb:.1f}s ({N/(time.time()-tb):.1f}/s)", flush=True)

    # Convert numpy samples to torch tensors in the main process (no cross-process
    # tensor sharing). Matches TensorCIFData / collate_pool: ((atom_fea, nbr_fea,
    # nbr_fea_idx), target) with float32 features and a 1-elem float target.
    tc = time.time()
    samples = [
        (
            (torch.from_numpy(a), torch.from_numpy(b), torch.from_numpy(c)),
            torch.tensor([t], dtype=torch.float32),
        )
        for (a, b, c), t in raw
    ]
    del raw
    print(f"Converted to torch in {time.time()-tc:.1f}s", flush=True)

    out_root = Path(args.output_dir) / args.task_name
    n_train, n_val = int(0.8 * N), int(0.1 * N)
    for seed in [int(s) for s in args.seeds.split(",")]:
        perm = list(range(N))
        random.seed(seed)
        random.shuffle(perm)
        split_idx = {"train": perm[:n_train], "val": perm[n_train:n_train + n_val], "test": perm[n_train + n_val:]}
        seed_dir = out_root / f"seed{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        counts = {}
        for split, idxs in split_idx.items():
            payload = {
                "samples": [samples[i] for i in idxs],
                "metadata": {"task": args.task_name, "split": split, "seed": seed,
                             "source": args.source, "n_samples": len(idxs),
                             "max_num_nbr": args.max_num_nbr, "radius": args.radius},
            }
            torch.save(payload, seed_dir / f"{split}.pt")
            counts[split] = len(idxs)
        with (seed_dir / "metadata.json").open("w") as f:
            json.dump({"task": args.task_name, "seed": seed, "counts": counts, "n_records": N}, f, indent=2)
        print(f"[seed {seed}] wrote {seed_dir}  counts={counts}", flush=True)

    print(f"ALL DONE in {time.time()-t0:.1f}s ({(time.time()-t0)/60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
