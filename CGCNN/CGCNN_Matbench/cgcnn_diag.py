import torch, torch.nn as nn, time
from cgcnn.model import CrystalGraphConvNet
from cgcnn.subspace import RandomSubspaceWrapper

torch.manual_seed(0)
orig_fea, nbr_fea_len = 92, 41
m = CrystalGraphConvNet(orig_fea, nbr_fea_len, atom_fea_len=56, n_conv=4, h_fea_len=64, n_h=3)
P = sum(p.numel() for p in m.parameters())
print(f"PARAM COUNT (n_conv=4,atom=56,h=64,n_h=3): {P:,}", flush=True)


def make_batch(nc=64, na=8, M=12):
    N = nc * na
    atom_fea = torch.randn(N, orig_fea)
    nbr_fea = torch.randn(N, M, nbr_fea_len)
    nbr_idx = torch.zeros(N, M, dtype=torch.long)
    crystal_atom_idx = []
    for c in range(nc):
        idx = torch.arange(c * na, (c + 1) * na)
        crystal_atom_idx.append(idx)
        for a in idx:
            nbr_idx[a] = idx[torch.randint(0, na, (M,))]
    tgt = torch.stack([atom_fea[ci, 0].mean() for ci in crystal_atom_idx]).view(-1, 1)
    return (atom_fea, nbr_fea, nbr_idx, crystal_atom_idx), tgt


batch, tgt = make_batch()
tgt = (tgt - tgt.mean()) / (tgt.std() + 1e-9)


def run(opt_name, dim, steps=200, lr=1e-3):
    t0 = time.time()
    torch.manual_seed(0)
    base = CrystalGraphConvNet(orig_fea, nbr_fea_len, atom_fea_len=56, n_conv=4, h_fea_len=64, n_h=3)
    w = RandomSubspaceWrapper(base, d=dim, method="fastfood", seed=123)
    opt = torch.optim.SGD([w.z], lr, momentum=0.9) if opt_name == "SGD" else torch.optim.Adam([w.z], lr)
    lossf = nn.MSELoss()
    w.train()
    z0 = w.z.detach().clone()
    losses = []
    for s in range(steps):
        out = w(*batch)
        loss = lossf(out, tgt)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if s in (0, 49, 199):
            losses.append(round(loss.item(), 4))
    dz = (w.z.detach() - z0).norm().item()
    gnorm = w.z.grad.norm().item() if w.z.grad is not None else None
    print(f"  {opt_name:4s} dim={dim:<4} loss[0,50,200]={losses}  |dz|={dz:.4f}  "
          f"last_gradnorm={gnorm:.3e}  ({time.time()-t0:.1f}s)", flush=True)


print("\nOverfit test (train loss should DROP if learning works):", flush=True)
for opt_name in ("SGD", "Adam"):
    for dim in (1.0, 0.1):
        run(opt_name, dim)
print("DIAG DONE", flush=True)
