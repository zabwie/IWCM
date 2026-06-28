#!/usr/bin/env python3
"""Grid world drift at H=25. Loads compositional_grid.pkl."""
import sys, pickle, torch, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
torch.set_num_threads(1)
import torch.nn as nn
from src.iwcm.slot_energy import SlotIWCMEnergy
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N, d_slot, H, d_a = 8, 19, 25, 11

with open("data/compositional_grid.pkl", "rb") as f:
    data = pickle.load(f)
tv = [(torch.from_numpy(z0).float(), torch.from_numpy(A).float(), torch.from_numpy(Z).float()) for z0,A,Z in data["train_valid"]]
ts = [(torch.from_numpy(z0).float(), torch.from_numpy(A).float(), torch.from_numpy(Z).float()) for z0,A,Z in data["test_valid"]]

class MLP(nn.Module):
    def __init__(s):
        super().__init__(); s.net = nn.Sequential(nn.Linear(N*d_slot+d_a,256),nn.ReLU(),nn.Linear(256,256),nn.ReLU(),nn.Linear(256,N*d_slot))
    def forward(s,z,a): B=z.shape[0]; return s.net(torch.cat([z.reshape(B,-1),a],-1)).reshape(B,N,d_slot)
    def rollout(s,z0,A): B,H=A.shape[:2];z=z0;zs=[];[zs.append(z:=s(z,A[:,t])) for t in range(H)]; return torch.stack(zs,1)

rl = MLP().to(DEVICE); opt = torch.optim.Adam(rl.parameters(),lr=1e-3)
pairs = [(torch.cat([z.unsqueeze(0),Z],0)[t],A[t],torch.cat([z.unsqueeze(0),Z],0)[t+1]) for z,A,Z in tv for t in range(H)]
for ep in range(50):
    idxs = torch.randperm(len(pairs)); ls = []
    for i in range(0,len(idxs),256):
        idx=idxs[i:i+256]; zs=torch.stack([pairs[j][0] for j in idx]).to(DEVICE)
        ac=torch.stack([pairs[j][1] for j in idx]).to(DEVICE); nx=torch.stack([pairs[j][2] for j in idx]).to(DEVICE)
        l=nn.MSELoss()(rl(zs,ac),nx); opt.zero_grad();l.backward();opt.step();ls.append(l.item())
    if (ep+1)%10==0: print(f"  ep {ep+1:3d}: loss={np.mean(ls):.6f}")

m = SlotIWCMEnergy(d_slot,d_a,hidden_dim=192,num_slots=N).to(DEVICE)
m.load_state_dict(torch.load("outputs/checkpoints/slot_iwcm_energy_v2_perobject.pt",map_location=DEVICE,weights_only=True)); m.eval()

def solve(z0,A,steps=50,lr=0.01,init_Z=None):
    B,Hf=A.shape[:2]; Z=(init_Z.clone().detach() if init_Z is not None else torch.randn(B,Hf,N,d_slot,device=DEVICE))
    Z.requires_grad_(True); v=torch.zeros_like(Z)
    for _ in range(steps):
        e=m(z0,A,Z).mean(); g=torch.autograd.grad(e,Z,create_graph=False)[0]; v=0.9*v+g
        Z=Z.detach()-lr*v; Z.requires_grad_(True); v=v.detach()
    return Z

S=[5,10,15,20,25]; R={t:[] for t in S}
for z0,A,Zt in ts:
    zb,Ab,Zb=z0.unsqueeze(0).to(DEVICE),A.unsqueeze(0).to(DEVICE),Zt.unsqueeze(0).to(DEVICE)
    with torch.no_grad(): Zr=rl.rollout(zb,Ab)
    Zc=solve(zb,Ab); Zw=solve(zb,Ab,init_Z=Zr)
    for t in S:
        with torch.no_grad(): R[t].append((m(zb,Ab[:,:t],Zb[:,:t]).item(),m(zb,Ab[:,:t],Zr[:,:t]).item(),m(zb,Ab[:,:t],Zc[:,:t]).item(),m(zb,Ab[:,:t],Zw[:,:t]).item()))
for t in S:
    v=np.mean([x[0] for x in R[t]]);r=np.mean([x[1] for x in R[t]]);c=np.mean([x[2] for x in R[t]]);w=np.mean([x[3] for x in R[t]])
    print(f"{t:4d} | {v:9.3f} {r:9.3f} {c:9.3f} {w:9.3f} | {r-v:+9.3f} {c-v:+9.3f} {w-v:+9.3f}")
v=R[25]; print(f"\nH=25: rollout Δ={np.mean([x[1]-x[0] for x in v]):+.3f}, cold Δ={np.mean([x[2]-x[0] for x in v]):+.3f}, warm Δ={np.mean([x[3]-x[0] for x in v]):+.3f}")
