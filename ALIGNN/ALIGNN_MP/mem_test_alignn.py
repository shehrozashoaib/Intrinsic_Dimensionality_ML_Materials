import torch
from alignn.models.alignn_atomwise import ALIGNNAtomWise, ALIGNNAtomWiseConfig
BASE=dict(name="alignn_atomwise",alignn_layers=1,gcn_layers=1,atom_input_features=92,
          edge_input_features=80,triplet_input_features=40,embedding_features=64,
          output_features=1,classification=False,calculate_gradient=False,
          atomwise_output_features=0,energy_mult_natoms=False)
print("GPU:",torch.cuda.get_device_name(0),
      f"total={torch.cuda.get_device_properties(0).total_memory/1e9:.0f} GB")
for h in (64,128,256):
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    m=ALIGNNAtomWise(ALIGNNAtomWiseConfig(hidden_features=h,**BASE)).cuda()
    n=sum(p.numel() for p in m.parameters())
    # simulate optimizer state (AdamW keeps 2 moments) -> the real training footprint of params
    opt=torch.optim.AdamW(m.parameters(),lr=1e-3)
    for p in m.parameters(): p.grad=torch.zeros_like(p)
    opt.step()
    peak=torch.cuda.max_memory_allocated()/1e6
    print(f"  hidden={h:3d} ({ {64:'1x',128:'2x',256:'4x'}[h] }): params={n:>9,}  "
          f"params+grads+AdamW state peak = {peak:.1f} MB")
    del m,opt
