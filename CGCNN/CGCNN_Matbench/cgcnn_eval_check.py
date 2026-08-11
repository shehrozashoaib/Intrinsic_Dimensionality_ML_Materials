import torch, torch.nn as nn
from cgcnn.model import CrystalGraphConvNet
from cgcnn.subspace import RandomSubspaceWrapper
torch.manual_seed(0)
of,nf=92,41
def make_batch(nc=64,na=8,M=12):
    N=nc*na; af=torch.randn(N,of); bf=torch.randn(N,M,nf); ni=torch.zeros(N,M,dtype=torch.long); cai=[]
    for c in range(nc):
        idx=torch.arange(c*na,(c+1)*na); cai.append(idx)
        for a in idx: ni[a]=idx[torch.randint(0,na,(M,))]
    tgt=torch.stack([af[ci,0].mean() for ci in cai]).view(-1,1); return (af,bf,ni,cai),tgt
batch,tgt=make_batch(); tgt=(tgt-tgt.mean())/(tgt.std()+1e-9)
base=CrystalGraphConvNet(of,nf,atom_fea_len=56,n_conv=4,h_fea_len=64,n_h=3)
w=RandomSubspaceWrapper(base,d=1.0,method="fastfood",seed=123)
opt=torch.optim.Adam([w.z],1e-3); lf=nn.MSELoss(); w.train()
for s in range(200):
    out=w(*batch); loss=lf(out,tgt); opt.zero_grad(); loss.backward(); opt.step()
train_mse=lf(w(*batch),tgt).item()
w.eval()
with torch.no_grad(): eval_mse=lf(w(*batch),tgt).item()
print(f"after Adam 200 steps: TRAIN-mode MSE={train_mse:.4f}  EVAL-mode MSE={eval_mse:.4f}", flush=True)
print("EVAL OK (BN running stats reflect training)" if eval_mse < 0.2 else "EVAL BROKEN (BN running-stats issue under functional_call)", flush=True)
