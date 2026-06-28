#!/usr/bin/env python3
"""Compare init: z0-replication vs random vs warm-start."""
from _common import *
w,e,da = env(); tv,tc,ts = data(w,e)
m = train_iwcm(tv,tc,d_a=da)
rl = Rollout(8,19,da).to(DEV); train_rl(rl,tv,100)
S = list(range(10,101,10)); R = {t:[] for t in S}
for z0,A,Zt in ts:
    zb,Ab,Zb = z0.unsqueeze(0).to(DEV),A.unsqueeze(0).to(DEV),Zt.unsqueeze(0).to(DEV)
    Zc = solve(m,zb,Ab,init_Z=torch.randn(1,100,8,19,device=DEV))
    Zz = z0.unsqueeze(0).unsqueeze(1).expand(-1,100,-1,-1).clone().to(DEV); Zz.requires_grad_(True)
    Zz = solve(m,zb,Ab,init_Z=Zz)
    with torch.no_grad(): Zr = rl.rollout(zb,Ab)
    Zw = solve(m,zb,Ab,init_Z=Zr,steps=100,lr=0.005)
    for t in S:
        with torch.no_grad(): R[t].append((m(zb,Ab[:,:t],Zb[:,:t]).item(),m(zb,Ab[:,:t],Zc[:,:t]).item(),m(zb,Ab[:,:t],Zz[:,:t]).item(),m(zb,Ab[:,:t],Zw[:,:t]).item()))
for t in S:
    v=np.mean([x[0] for x in R[t]]); c=np.mean([x[1] for x in R[t]]); z=np.mean([x[2] for x in R[t]]); w=np.mean([x[3] for x in R[t]])
    print(f"{t:4d} | {v:9.3f} {c:9.3f} {z:9.3f} {w:9.3f} | {c-v:+9.3f} {z-v:+9.3f} {w-v:+9.3f}")
v=R[100]; print(f"H=100: cold Δ={np.mean([x[1]-x[0] for x in v]):+.3f}, z0rep Δ={np.mean([x[2]-x[0] for x in v]):+.3f}, warm Δ={np.mean([x[3]-x[0] for x in v]):+.3f}")
