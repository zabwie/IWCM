#!/usr/bin/env python3
"""Drift: rollout vs IWCM warm-start on DM Control at H=100."""
from _common import *
w,e,da = env(); tv,tc,ts = data(w,e)
m = train_iwcm(tv,tc,d_a=da)
rl = Rollout(8,19,da).to(DEV); train_rl(rl,tv,100)
S = list(range(10,101,10)); R = {t:[] for t in S}
for z0,A,Zt in ts[:80]:
    zb,Ab,Zb = z0.unsqueeze(0).to(DEV),A.unsqueeze(0).to(DEV),Zt.unsqueeze(0).to(DEV)
    with torch.no_grad(): Zr = rl.rollout(zb,Ab)
    Zw = solve(m,zb,Ab,init_Z=Zr,steps=100,lr=0.005)
    for t in S:
        with torch.no_grad(): R[t].append((m(zb,Ab[:,:t],Zb[:,:t]).item(),m(zb,Ab[:,:t],Zr[:,:t]).item(),m(zb,Ab[:,:t],Zw[:,:t]).item()))
for t in S:
    v=np.mean([x[0] for x in R[t]]); r=np.mean([x[1] for x in R[t]]); w=np.mean([x[2] for x in R[t]])
    print(f"{t:4d} | {v:9.3f} {r:9.3f} {w:9.3f} | {r-v:+9.3f} {w-v:+9.3f}")
v=R[100]; dr=np.mean([x[1]-x[0] for x in v]); dw=np.mean([x[2]-x[0] for x in v])
print(f"\nH=100: rollout Δ={dr:+.3f}, warm Δ={dw:+.3f}")
