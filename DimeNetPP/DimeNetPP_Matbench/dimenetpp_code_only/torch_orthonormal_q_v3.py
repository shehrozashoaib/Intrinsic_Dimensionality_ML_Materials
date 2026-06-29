import argparse
import gc
import os
import time

import numpy as np
import torch


def main():
    p = argparse.ArgumentParser('Generate orthonormal Q with PyTorch for v3')
    p.add_argument('--rows', type=int, required=True)
    p.add_argument('--cols', type=int, required=True)
    p.add_argument('--seed', type=int, default=123)
    p.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'])
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--out', type=str, required=True)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    if args.device == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA requested for torch_orthonormal_q_v3.py but torch.cuda.is_available() is False')
        torch.cuda.set_device(args.gpu)
        torch.cuda.manual_seed_all(args.seed)
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device('cpu')

    print(f'[TorchQR-v3] device={device} rows={args.rows} cols={args.cols} seed={args.seed}', flush=True)
    t0 = time.time()
    A = torch.randn(args.rows, args.cols, device=device, dtype=torch.float32)
    print(f'[TorchQR-v3] random matrix built in {time.time() - t0:.2f}s', flush=True)

    t1 = time.time()
    Q, _ = torch.linalg.qr(A, mode='reduced')
    print(f'[TorchQR-v3] QR finished in {time.time() - t1:.2f}s', flush=True)
    del A
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
    gc.collect()

    t2 = time.time()
    q_np = Q.detach().cpu().numpy().astype(np.float32, copy=False)
    print(f'[TorchQR-v3] moved Q to CPU numpy in {time.time() - t2:.2f}s', flush=True)
    del Q
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    gc.collect()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    t3 = time.time()
    np.save(args.out, q_np)
    print(f'[TorchQR-v3] saved Q in {time.time() - t3:.2f}s', flush=True)
    del q_np
    gc.collect()
    print(f'[TorchQR-v3] wrote {args.out}', flush=True)


if __name__ == '__main__':
    main()
