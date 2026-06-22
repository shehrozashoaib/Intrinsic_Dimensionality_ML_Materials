import argparse
import gc
import os
import numpy as np
import torch


def main():
    p = argparse.ArgumentParser('Generate orthonormal Q with PyTorch')
    p.add_argument('--rows', type=int, required=True)
    p.add_argument('--cols', type=int, required=True)
    p.add_argument('--seed', type=int, default=123)
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--dtype', type=str, default='float32', choices=['float32'])
    p.add_argument('--out', type=str, required=True)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        torch.cuda.manual_seed_all(args.seed)
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device('cpu')

    dtype = torch.float32
    print(f'[TorchQR] device={device} rows={args.rows} cols={args.cols} seed={args.seed}', flush=True)
    A = torch.randn(args.rows, args.cols, device=device, dtype=dtype)
    Q, _ = torch.linalg.qr(A, mode='reduced')
    del A
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
    gc.collect()

    q_np = Q.detach().cpu().numpy().astype(np.float32, copy=False)
    del Q
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    gc.collect()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.save(args.out, q_np)
    del q_np
    gc.collect()
    print(f'[TorchQR] wrote {args.out}', flush=True)


if __name__ == '__main__':
    main()
