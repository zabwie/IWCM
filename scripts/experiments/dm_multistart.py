#!/usr/bin/env python3
"""Multi-start solver: cheap probes + commitment from random init."""
from _common import *
w,e,da = env(); tv,tc,ts = data(w,e)
m = train_iwcm(tv,tc,d_a=da)
rl = Rollout(8,19,da).to(DEV); train_rl(rl,tv,100)
def multi(z0,A,N=20,pk=3,fk=100,lr=0.01):
    B,Hf = A.shape[:2]; bZ=None; bE=float('inf')
    for _ in range(N):
        Z = torch.randn(B,Hf,8,19,device=DEV,requires_grad=True); v = torch.zeros_like(Z)
        for _ in range(pk):
            g = torch.autograd.grad(m(z0,A,Z).mean(),Z,create_graph=False)[0]
            v = 0.9*v+g; Z = Z.detach()-lr*v; Z.requires_grad_(True); v = v.detach()
        e=m(z0,A,Z).mean().item()
        if e<bE: bE=e; bZ=Z.clone().detach()
    Z=bZ.requires_grad_(True); v=torch.zeros_like(Z)
    for _ in range(fk):
        g = torch.autograd.grad(m(z0,A,Z).mean(),Z,create_graph=False)[0]
        v = 0.9*v+g; Z = Z.detach()-lr*v; Z.requires_grad_(True); v = v.detach()
    return Z
S = list(range(10,101,10)); R = {t:[] for t in S}
for z0,A,Zt in ts:
    zb,Ab,Zb = z0.unsqueeze(0).to(DEV),A.unsqueeze(0).to(DEV),Zt.unsqueeze(0).to(DEV)
    Zc = torch.randn(1,100,8,19,device=DEV,requires_grad=True); v=torch.zeros_like(Zc)
    for _ in range(100):
        g = torch.autograd.grad(m(zb,Ab,Zc).mean(),Zc,create_graph=False)[0]
        v = 0.9*v+g; Zc = Zc.detach()-0.01*v; Zc.requires_grad_(True); v = v.detach()
    Zm = multi(zb,Ab)
    with torch.no_grad(): Zr = rl.rollout(zb,Ab)
    Zw = solve(m,zb,Ab,init_Z=Zr,steps=100,lr=0.005)
    for t in S:
        with torch.no_grad(): R[t].append((m(zb,Ab[:,:t],Zb[:,:t]).item(),m(zb,Ab[:,:t],Zc[:,:t]).item(),m(zb,Ab[:,:t],Zm[:,:t]).item(),m(zb,Ab[:,:t],Zw[:,:t]).item()))
for t in S:
    v=np.mean([x[0] for x in R[t]]);c=np.mean([x[1] for x in R[t]]);m_=np.mean([x[2] for x in R[t]]);w=np.mean([x[3] for x in R[t]])
    print(f"{t:4d} | {v:9.3f} {c:9.3f} {m_:9.3f} {w:9.3f} | {c-v:+9.3f} {m_-v:+9.3f} {w-v:+9.3f}")
v=R[100]; print(f"\nH=100: cold Δ={np.mean([x[1]-x[0] for x in v]):+.3f}, multi Δ={np.mean([x[2]-x[0] for x in v]):+.3f}, warm Δ={np.mean([x[3]-x[0] for x in v]):+.3f}")
