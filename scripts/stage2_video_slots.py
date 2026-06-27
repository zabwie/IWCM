#!/usr/bin/env python3
"""Stage 2 v2: VideoSlotIWCM — calibrated comparison with AUROC, energy dist, balanced acc."""
import sys, time, numpy as np, torch, torch.nn.functional as F
from pathlib import Path
from collections import defaultdict
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.seed import set_seed
from src.encoder.oracle_slot_encoder import (encode_oracle_trajectory, build_door_key_map, ORACLE_SLOT_DIM, MAX_OBJECTS)
from src.encoder.video_encoder import VideoEncoder
from src.encoder.decoder import VideoDecoder
from src.iwcm.fused_energy import FusedIWCMEnergy
from src.env.grid_world import GridWorld
from src.env.renderer import GridWorldRenderer
from src.env.scenarios import Scenario

HORIZON, GRID_SIZE, FRAME_SIZE, CELL_PX = 25, 8, 64, 8
SLOT_DIM = 64
NUM_TRAIN, NUM_TEST, NUM_VAL = 200, 50, 30
EPOCHS, LR = 200, 3e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VTYPES = ["delete", "swap", "teleport", "duplicate", "transform"]

def gen_traj(scenario, rng, corrupt=None):
    gw = GridWorld(grid_size=GRID_SIZE, objects_config=scenario.to_env_config(), seed=int(rng.randint(0,2**31)))
    gw.reset(); states=[dict(gw.get_state())]; acts=[]
    for _ in range(HORIZON+10):
        va=gw.get_valid_actions()
        if not va: break
        a=int(rng.choice(va)); s,_,done,_=gw.step(a); states.append(dict(s)); acts.append(a)
        if done: break
    if len(states)<HORIZON+1: return None
    actions=np.array(acts[:HORIZON],dtype=np.int64)
    meta={"law_type":"valid","violation_type":"none"}
    if corrupt:
        tc=rng.randint(HORIZON//4,3*HORIZON//4); oids=list(states[0].get("objects",{}).keys())
        if not oids: return None
        tgt=oids[rng.randint(0,len(oids))]
        if corrupt=="delete":
            for t in range(tc,HORIZON+1): states[t].setdefault("objects",{}).pop(tgt,None)
        elif corrupt=="swap" and len(oids)>=2:
            o2=oids[rng.randint(0,len(oids))]
            while o2==tgt: o2=oids[rng.randint(0,len(oids))]
            for t in range(tc,HORIZON+1):
                ob=states[t].setdefault("objects",{})
                if tgt in ob and o2 in ob: ob[tgt],ob[o2]=dict(ob[o2]),dict(ob[tgt])
        elif corrupt=="teleport":
            nr,nc=rng.randint(0,GRID_SIZE-1),rng.randint(0,GRID_SIZE-1)
            for t in range(tc,HORIZON+1):
                if tgt in states[t].get("objects",{}): states[t]["objects"][tgt]["pos"]=(nr,nc)
        elif corrupt=="duplicate":
            ni=tgt+"_dup"
            for t in range(tc,HORIZON+1):
                if tgt in states[t].get("objects",{}): states[t]["objects"][ni]=dict(states[t]["objects"][tgt])
        elif corrupt=="transform":
            for t in range(tc,HORIZON+1):
                if tgt in states[t].get("objects",{}): states[t]["objects"][tgt]["type"]="box"
        meta={"law_type":"conservation" if corrupt in["delete","duplicate","transform"] else "identity","violation_type":corrupt}
    return states[:HORIZON+1],actions,meta

def render_traj(states,renderer):
    return torch.stack([torch.from_numpy(renderer.render_frame(s)).float().permute(2,0,1)/255.0 for s in states])

def oracle_encode(states,actions,scenario):
    dkm=build_door_key_map(scenario.to_env_config()); goal=scenario.to_env_config().get("goal",None)
    return encode_oracle_trajectory(states,actions,HORIZON,GRID_SIZE,goal,dkm)

def gen_dataset(nv,nc,rng):
    scenarios=[Scenario.from_preset(n,GRID_SIZE) for n in["key_door_simple","multi_object"]]
    renderer=GridWorldRenderer(grid_size=GRID_SIZE,cell_px=CELL_PX); vd,cd=[],[]
    for _ in range(nv):
        sc=scenarios[rng.randint(0,len(scenarios))]; r=gen_traj(sc,rng)
        if r is None: continue
        st,ac,m=r; orc=oracle_encode(st,ac,sc)
        if orc is None: continue
        vd.append((render_traj(st,renderer),orc,ac,m))
    for _ in range(nc):
        sc=scenarios[rng.randint(0,len(scenarios))]; vt=VTYPES[rng.randint(0,len(VTYPES))]
        r=gen_traj(sc,rng,vt)
        if r is None: continue
        st,ac,m=r; orc=oracle_encode(st,ac,sc)
        if orc is None: continue
        cd.append((render_traj(st,renderer),orc,ac,m))
    return {"valid":vd,"corrupt":cd}

def compute_metrics(model,data,use_oracle=True,encoder=None,iwcm_v=None):
    energies,labels=[],[]
    for frames,oracle,actions,meta in data["valid"]:
        if use_oracle:
            z0,A,Z=oracle
            e=model(torch.from_numpy(z0).float().unsqueeze(0).to(DEVICE),torch.from_numpy(A).float().unsqueeze(0).to(DEVICE),torch.from_numpy(Z).float().unsqueeze(0).to(DEVICE)).item()
        else:
            f=frames.unsqueeze(0).to(DEVICE); slots=encoder(f[:,:HORIZON])
            e=iwcm_v(slots[:,0],torch.from_numpy(actions[:HORIZON]).float().unsqueeze(0).to(DEVICE),slots).item()
        energies.append(e); labels.append(0)
    for frames,oracle,actions,meta in data["corrupt"]:
        if use_oracle:
            z0,A,Z=oracle
            e=model(torch.from_numpy(z0).float().unsqueeze(0).to(DEVICE),torch.from_numpy(A).float().unsqueeze(0).to(DEVICE),torch.from_numpy(Z).float().unsqueeze(0).to(DEVICE)).item()
        else:
            f=frames.unsqueeze(0).to(DEVICE); slots=encoder(f[:,:HORIZON])
            e=iwcm_v(slots[:,0],torch.from_numpy(actions[:HORIZON]).float().unsqueeze(0).to(DEVICE),slots).item()
        energies.append(e); labels.append(1)
    energies=np.array(energies); labels=np.array(labels)
    from sklearn.metrics import roc_auc_score; auroc=roc_auc_score(labels,energies)
    return {"energies":energies,"labels":labels,"auroc":auroc}

def calibrate(val_e,val_l,test_e,test_l):
    bt,bb=0,0
    for t in np.linspace(val_e.min(),val_e.max(),200):
        p=val_e>t; va=((p==0)&(val_l==0)).sum()/max((val_l==0).sum(),1)
        ir=((p==1)&(val_l==1)).sum()/max((val_l==1).sum(),1); b=(va+ir)/2
        if b>bb: bb,bt=b,t
    pt=test_e>bt; va=((pt==0)&(test_l==0)).sum()/max((test_l==0).sum(),1)
    ir=((pt==1)&(test_l==1)).sum()/max((test_l==1).sum(),1)
    return {"threshold":bt,"valid_acc":va,"invalid_rej":ir,"balanced_acc":(va+ir)/2}

def train_oracle(train_data):
    m=FusedIWCMEnergy(d_slot=ORACLE_SLOT_DIM,d_action=11,hidden=128,num_slots=MAX_OBJECTS).to(DEVICE)
    opt=torch.optim.Adam(m.parameters(),lr=LR)
    tv=[(torch.from_numpy(z0).float(),torch.from_numpy(A).float(),torch.from_numpy(Z).float()) for _,(z0,A,Z),_,_ in train_data["valid"]]
    tc=[(torch.from_numpy(z0).float(),torch.from_numpy(A).float(),torch.from_numpy(Z).float(),meta) for _,(z0,A,Z),_,meta in train_data["corrupt"]]
    n=min(len(tv),len(tc))
    for _ in range(EPOCHS):
        vi=np.random.choice(len(tv),n,replace=False); ci=np.random.choice(len(tc),n,replace=False)
        vz0=torch.stack([tv[i][0] for i in vi]).to(DEVICE); cz0=torch.stack([tc[i][0] for i in ci]).to(DEVICE)
        vA=torch.stack([tv[i][1] for i in vi]).to(DEVICE); cA=torch.stack([tc[i][1] for i in ci]).to(DEVICE)
        vZ=torch.stack([tv[i][2] for i in vi]).to(DEVICE); cZ=torch.stack([tc[i][2] for i in ci]).to(DEVICE)
        opt.zero_grad(); ev=m(vz0,vA,vZ); ec=m(cz0,cA,cZ)
        loss=(F.relu(ev+1.0).mean()+F.relu(1.0-ec).mean()+0.001*(ev.pow(2).mean()+ec.pow(2).mean()))
        loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step()
    m.eval(); return m

def train_video(train_data, encoder=None):
    enc = encoder if encoder is not None else VideoEncoder(frame_size=FRAME_SIZE,in_channels=3,num_slots=MAX_OBJECTS,slot_dim=SLOT_DIM).to(DEVICE)
    iw=FusedIWCMEnergy(d_slot=SLOT_DIM,d_action=11,hidden=128,num_slots=MAX_OBJECTS).to(DEVICE)
    opt=torch.optim.Adam(list(enc.parameters())+list(iw.parameters()),lr=LR)
    n=min(len(train_data["valid"]),len(train_data["corrupt"]))
    for _ in range(EPOCHS):
        vi=np.random.choice(len(train_data["valid"]),min(n,len(train_data["valid"])),replace=False)
        ci=np.random.choice(len(train_data["corrupt"]),min(n,len(train_data["corrupt"])),replace=False)
        vf=torch.stack([train_data["valid"][i][0] for i in vi]).to(DEVICE)
        vA=torch.stack([torch.from_numpy(train_data["valid"][i][2][:HORIZON]).float() for i in vi]).to(DEVICE)
        vs=enc(vf[:,:HORIZON]); vz=vs[:,0]
        cf=torch.stack([train_data["corrupt"][i][0] for i in ci]).to(DEVICE)
        cA=torch.stack([torch.from_numpy(train_data["corrupt"][i][2][:HORIZON]).float() for i in ci]).to(DEVICE)
        cs=enc(cf[:,:HORIZON]); cz=cs[:,0]
        opt.zero_grad()
        ev=iw(vz,vA,vs); ec=iw(cz,cA,cs)
        loss_iwcm=(F.relu(ev+1.0).mean()+F.relu(1.0-ec).mean()+0.001*(ev.pow(2).mean()+ec.pow(2).mean()))
        loss_smooth=0.01*(vs[:,1:]-vs[:,:-1]).pow(2).mean()
        vsn=F.normalize(vs.mean(dim=1),dim=-1)
        sim=torch.bmm(vsn,vsn.transpose(1,2)); eye=torch.eye(MAX_OBJECTS,device=DEVICE).unsqueeze(0)
        loss_div=0.01*F.relu(sim-eye).mean()
        (loss_iwcm+loss_smooth+loss_div).backward()
        torch.nn.utils.clip_grad_norm_(list(enc.parameters())+list(iw.parameters()),1.0); opt.step()
    enc.eval(); iw.eval(); return enc,iw

def pretrain_encoder(train_data, epochs=50):
    """Pretrain encoder+decoder on frame reconstruction before IWCM training."""
    enc=VideoEncoder(frame_size=FRAME_SIZE,in_channels=3,num_slots=MAX_OBJECTS,slot_dim=SLOT_DIM).to(DEVICE)
    dec=VideoDecoder(slot_dim=SLOT_DIM,frame_size=FRAME_SIZE,out_channels=3).to(DEVICE)
    opt=torch.optim.Adam(list(enc.parameters())+list(dec.parameters()),lr=LR)
    n=min(len(train_data["valid"]),32)
    for ep in range(epochs):
        vi=np.random.choice(len(train_data["valid"]),n,replace=False)
        vf=torch.stack([train_data["valid"][i][0] for i in vi]).to(DEVICE)
        # Reconstruct each frame independently: flatten H into batch
        B,H,C,W_img,H_img=vf.shape
        frames_flat=vf.reshape(B*H,C,W_img,H_img)
        slots_flat=enc.encode_frame(frames_flat)  # (B*H, N, d)
        recon_flat=dec.decode_frame(slots_flat)    # (B*H, C, W, H)
        loss=F.mse_loss(recon_flat,frames_flat)
        opt.zero_grad(); loss.backward(); opt.step()
        if (ep+1)%20==0: print(f"  recon ep {ep+1}/{epochs}: loss={loss.item():.4f}")
    enc.eval(); dec.eval()
    return enc,dec

def main():
    set_seed(42); rng=np.random.RandomState(42)
    print("STAGE 2 v2 — Calibrated VideoSlotIWCM"); print("="*55)
    t0=time.time()
    tr=gen_dataset(NUM_TRAIN,NUM_TRAIN,rng); va=gen_dataset(NUM_VAL,NUM_VAL,np.random.RandomState(99))
    te=gen_dataset(NUM_TEST,NUM_TEST,np.random.RandomState(123))
    print(f"Data: train v={len(tr['valid'])} c={len(tr['corrupt'])} val v={len(va['valid'])} c={len(va['corrupt'])} test v={len(te['valid'])} c={len(te['corrupt'])} ({time.time()-t0:.1f}s)")
    print("Train Oracle..."); t0=time.time(); mo=train_oracle(tr); print(f"  {time.time()-t0:.1f}s")
    print("Pretrain Encoder+Decoder (reconstruction)..."); t0=time.time()
    enc,dec=pretrain_encoder(tr, epochs=50); print(f"  {time.time()-t0:.1f}s")
    print("Train Video IWCM on pretrained slots..."); t0=time.time()
    enc,iwv=train_video(tr, encoder=enc); print(f"  {time.time()-t0:.1f}s")
    mo_val=compute_metrics(mo,va,use_oracle=True); mv_val=compute_metrics(None,va,use_oracle=False,encoder=enc,iwcm_v=iwv)
    mo_test=compute_metrics(mo,te,use_oracle=True); mv_test=compute_metrics(None,te,use_oracle=False,encoder=enc,iwcm_v=iwv)
    # Energy stats
    for name,met in[("Oracle",mo_test),("Learned",mv_test)]:
        ev=met["energies"][met["labels"]==0]; ei=met["energies"][met["labels"]==1]
        print(f"\n{name}: E_valid mean={ev.mean():.2f} std={ev.std():.2f}  E_invalid mean={ei.mean():.2f} std={ei.std():.2f}")
        print(f"  overlap={(ev>ei.min()).mean():.2f}")
    co=calibrate(mo_val["energies"],mo_val["labels"],mo_test["energies"],mo_test["labels"])
    cv=calibrate(mv_val["energies"],mv_val["labels"],mv_test["energies"],mv_test["labels"])
    print("\n"+"="*55); print(f"{'Metric':<20} {'Oracle':>12} {'Learned':>12} {'Target':>8}")
    print("-"*55)
    for n,o,v,t in[("AUROC",mo_test["auroc"],mv_test["auroc"],0.85),("Valid Acc",co["valid_acc"],cv["valid_acc"],0.70),("Invalid Rej",co["invalid_rej"],cv["invalid_rej"],0.70),("Balanced Acc",co["balanced_acc"],cv["balanced_acc"],0.75)]:
        print(f"{n:<20} {o:>12.3f} {v:>12.3f} {t:>8.2f}")
    print(f"\nVerdict: ",end="")
    if cv["balanced_acc"]>=0.75: print("Stage 2 VALIDATED")
    elif mv_test["auroc"]>=0.80: print("AUROC OK — calibration gap. Add recon loss.")
    elif mv_test["auroc"]>=0.65: print("Weak signal (AUROC={:.2f}). Encoder needs reconstruction training.".format(mv_test["auroc"]))
    else: print("No signal. Encoder not learning meaningful slots.")

if __name__=="__main__": main()
