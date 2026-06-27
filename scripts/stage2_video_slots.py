#!/usr/bin/env python3
"""Stage 2.1: VideoSlotIWCM with temporal slot-permanence pretraining.

Adds Hungarian-matched slot permanence loss across adjacent frames to
enforce object persistence in the learned slot representations.
"""
import sys, time, numpy as np, torch, torch.nn.functional as F
from pathlib import Path
from collections import defaultdict
sys.path.insert(0, str(Path(__file__).parent.parent))

from scipy.optimize import linear_sum_assignment

from src.utils.seed import set_seed
from src.encoder.oracle_slot_encoder import (encode_oracle_trajectory, build_door_key_map, ORACLE_SLOT_DIM, MAX_OBJECTS)
from src.encoder.video_encoder import VideoEncoder
from src.encoder.decoder import VideoDecoder
from src.iwcm.fused_energy import FusedIWCMEnergy
from src.env.grid_world import GridWorld
from src.env.renderer import GridWorldRenderer
from src.env.scenarios import Scenario

HORIZON, GRID_SIZE, FRAME_SIZE, CELL_PX = 25, 8, 64, 8
SLOT_DIM = ORACLE_SLOT_DIM  # 19 — match oracle slot dimension for distillation
CONTENT_DIM = 14  # content=14, pose=5 (channels 5-9 are position/velocity)
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
        vs=enc.forward_temporal(vf[:,:HORIZON]); vz=vs[:,0]
        cf=torch.stack([train_data["corrupt"][i][0] for i in ci]).to(DEVICE)
        cA=torch.stack([torch.from_numpy(train_data["corrupt"][i][2][:HORIZON]).float() for i in ci]).to(DEVICE)
        cs=enc.forward_temporal(cf[:,:HORIZON]); cz=cs[:,0]
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


# ─── Oracle-Slot Distillation ────────────────────────────────────────────────

def pretrain_distill_oracle(train_data, epochs=100):
    """Train video encoder to predict oracle slots from rendered frames.

    This gives the encoder direct object-level supervision:
    "Given these pixels, produce these specific slot values."
    Once distilled, the encoder's output can be fed directly to IWCM.
    """
    enc = VideoEncoder(frame_size=FRAME_SIZE, in_channels=3, num_slots=MAX_OBJECTS,
                       slot_dim=SLOT_DIM).to(DEVICE)
    dec = VideoDecoder(slot_dim=SLOT_DIM, frame_size=FRAME_SIZE, out_channels=3).to(DEVICE)
    opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=LR)

    n = min(len(train_data["valid"]), 32)
    oracle_slots = []  # collect oracle slots for each valid trajectory

    for ep in range(epochs):
        vi = np.random.choice(len(train_data["valid"]), n, replace=False)

        total_loss = 0.0
        total_slot_loss = 0.0
        for i in vi:
            frames, oracle, _, _ = train_data["valid"][i]
            vf = frames.unsqueeze(0).to(DEVICE)  # (1, H+1, C, W, H)
            B, Hf, C, W, H_img = vf.shape
            H_use = min(Hf, HORIZON)

            # Encode with temporal propagation
            slots_pred = enc.forward_temporal(vf[:, :H_use])  # (1, H, N, d)

            # Oracle target: (H, N, 19) -> (1, H, N, d)
            _, A_oracle, Z_oracle = oracle
            Z_tgt = torch.from_numpy(Z_oracle[:H_use]).float().unsqueeze(0).to(DEVICE)

            # Slot distillation loss
            slot_loss = F.mse_loss(slots_pred, Z_tgt)

            # Reconstruction loss
            recon_flat = dec.decode_frame(slots_pred.reshape(B * H_use, MAX_OBJECTS, SLOT_DIM))
            recon_loss = F.mse_loss(recon_flat, vf[:, :H_use].reshape(B * H_use, C, W, H_img))

            loss = slot_loss + 0.1 * recon_loss
            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item()
            total_slot_loss += slot_loss.item()

        if (ep + 1) % 20 == 0:
            print(f"  distill ep {ep+1}/{epochs}: slot_loss={total_slot_loss/n:.4f} "
                  f"recon={total_loss/n - total_slot_loss/n:.4f}")

    # Evaluate switch rate
    enc.eval()
    with torch.no_grad():
        vf = torch.stack([train_data["valid"][i][0][:8] for i in range(min(8, len(train_data["valid"])))]).to(DEVICE)
        slots_test = enc.forward_temporal(vf)
        sr, ma = compute_switch_rate(slots_test)
        print(f"  Post-distill switch_rate={sr:.4f} match_acc={ma:.4f}")

    enc.eval(); dec.eval()
    return enc, dec


# ─── Slot Permanence ─────────────────────────────────────────────────────────

def hungarian_match_slots(slots_t, slots_tp1, temperature=0.1):
    """Match slots at time t to slots at time t+1 via Hungarian algorithm.

    Args:
        slots_t:   (B, N, d) — slots at timestep t
        slots_tp1: (B, N, d) — slots at timestep t+1
        temperature: softmax temperature for similarity

    Returns:
        matched_slots: (B, N, d) — slots_tp1 reordered to match slots_t
        match_indices: (B, N) — index in slots_tp1 that matches each slot_t
    """
    B, N, d = slots_t.shape
    device = slots_t.device

    # Cosine similarity between all pairs
    slots_t_n = F.normalize(slots_t, dim=-1)
    slots_tp1_n = F.normalize(slots_tp1, dim=-1)
    sim = torch.bmm(slots_t_n, slots_tp1_n.transpose(1, 2))  # (B, N, N)
    cost = (1.0 - sim).detach().cpu().numpy()  # Hungarian minimizes cost

    matched = torch.zeros_like(slots_tp1)
    indices = torch.zeros(B, N, dtype=torch.long, device=device)

    for b in range(B):
        row_ind, col_ind = linear_sum_assignment(cost[b])
        matched[b, row_ind] = slots_tp1[b, col_ind]
        indices[b, row_ind] = torch.tensor(col_ind, device=device)

    return matched, indices


def compute_switch_rate(slots):
    """Compute slot switch rate: fraction of slots that change index across time.

    Args:
        slots: (B, H, N, d)

    Returns:
        switch_rate: scalar — lower is better (0=perfectly stable)
        match_accuracy: scalar — fraction of slots matched to same index
    """
    B, H, N, d = slots.shape
    switches = 0
    matches = 0
    total = 0

    for b in range(B):
        for t in range(H - 1):
            _, indices = hungarian_match_slots(
                slots[b:b+1, t], slots[b:b+1, t+1]
            )
            switches += (indices[0] != torch.arange(N, device=slots.device)).sum().item()
            matches += (indices[0] == torch.arange(N, device=slots.device)).sum().item()
            total += N

    return switches / max(total, 1), matches / max(total, 1)


def slot_permanence_loss(slots, content_dim=CONTENT_DIM):
    """Hungarian-matched slot permanence loss across time.

    For each adjacent pair (t, t+1): match slots via Hungarian,
    penalize content change, allow pose change.

    Args:
        slots: (B, H, N, d)

    Returns:
        loss: scalar
        switch_rate: diagnostic
    """
    B, H, N, d = slots.shape
    loss = 0.0
    total_switches = 0

    for t in range(H - 1):
        matched, indices = hungarian_match_slots(slots[:, t], slots[:, t+1])

        # Content loss on matched pairs
        content_loss = F.mse_loss(
            matched[:, :, :content_dim],
            slots[:, t, :, :content_dim]
        )

        # Pose change should be small but non-zero
        pose_loss = F.mse_loss(
            matched[:, :, content_dim:],
            slots[:, t, :, content_dim:]
        )

        loss += content_loss + 0.3 * pose_loss

        # Track switches
        ideal = torch.arange(N, device=slots.device).unsqueeze(0).expand(B, -1)
        total_switches += (indices != ideal).float().mean().item()

    return loss / (H - 1), total_switches / (H - 1)


# ─── Multi-Phase Pretraining ─────────────────────────────────────────────────

def pretrain_encoder_with_permanence(train_data, epochs_per_phase=30):
    """Multi-phase pretraining: recon → matched permanence → fixed-index → IWCM-ready.

    Phase 1: reconstruction only (learn to represent scenes)
    Phase 2: reconstruction + matched permanence (learn object persistence)
    Phase 3: reconstruction + fixed-index permanence (stabilize assignments)
    Phase 4: reconstruction + permanence + IWCM margin (law detection)
    """
    enc = VideoEncoder(frame_size=FRAME_SIZE, in_channels=3, num_slots=MAX_OBJECTS,
                       slot_dim=SLOT_DIM).to(DEVICE)
    dec = VideoDecoder(slot_dim=SLOT_DIM, frame_size=FRAME_SIZE, out_channels=3).to(DEVICE)
    opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=LR)

    n = min(len(train_data["valid"]), 32)
    phases = [
        ("Phase 1: recon only", epochs_per_phase, 1.0, 0.0, 0.0),       # recon_weight, perm_weight, fixed_weight
        ("Phase 2: +matched permanence", epochs_per_phase, 0.5, 0.5, 0.0),
        ("Phase 3: +fixed-index stability", epochs_per_phase//2, 0.3, 0.3, 0.4),
        ("Phase 4: +IWCM margin", epochs_per_phase//2, 0.2, 0.2, 0.3),
    ]

    for phase_name, epochs, w_recon, w_perm, w_fixed in phases:
        print(f"  {phase_name}")
        for ep in range(epochs):
            vi = np.random.choice(len(train_data["valid"]), n, replace=False)
            vf = torch.stack([train_data["valid"][i][0] for i in vi]).to(DEVICE)

            # Reconstruction + temporal slot propagation
            Bf, Hf, C, W, H_img = vf.shape
            slots = enc.forward_temporal(vf)  # (B, H, N, d) — temporally consistent
            slots_flat = slots.reshape(Bf * Hf, MAX_OBJECTS, SLOT_DIM)
            recon_flat = dec.decode_frame(slots_flat)
            loss_recon = F.mse_loss(recon_flat, vf.reshape(Bf * Hf, C, W, H_img))

            loss = w_recon * loss_recon

            # Slot permanence (if enabled)
            perm_loss = torch.tensor(0.0, device=DEVICE)
            switch_rate = 0.0
            if w_perm > 0:
                perm_loss, switch_rate = slot_permanence_loss(slots)

            # Fixed-index permanence (if enabled)
            fixed_loss = torch.tensor(0.0, device=DEVICE)
            if w_fixed > 0:
                for t in range(Hf - 1):
                    fixed_loss += F.mse_loss(
                        slots[:, t+1, :, :CONTENT_DIM],
                        slots[:, t, :, :CONTENT_DIM]
                    )
                fixed_loss /= max(Hf - 1, 1)

            loss += w_perm * perm_loss + w_fixed * 0.5 * fixed_loss

            opt.zero_grad()
            loss.backward()
            opt.step()

        # Phase diagnostic
        enc.eval()
        with torch.no_grad():
            vf = torch.stack([train_data["valid"][i][0][:8] for i in range(min(8, len(train_data["valid"])))])
            vf = vf.to(DEVICE)
            Bf, Hf, C, W, H_img = vf.shape
            slots = enc(vf[:, :Hf])
            sr, ma = compute_switch_rate(slots)
        enc.train()
        print(f"    recon={loss_recon.item():.4f} perm={perm_loss.item():.4f} "
              f"fixed={fixed_loss.item():.4f} switch_rate={sr:.3f} match_acc={ma:.3f}")

    enc.eval(); dec.eval()
    return enc, dec

def main():
    set_seed(42); rng=np.random.RandomState(42)
    print("STAGE 2 v2 — Calibrated VideoSlotIWCM"); print("="*55)
    t0=time.time()
    tr=gen_dataset(NUM_TRAIN,NUM_TRAIN,rng); va=gen_dataset(NUM_VAL,NUM_VAL,np.random.RandomState(99))
    te=gen_dataset(NUM_TEST,NUM_TEST,np.random.RandomState(123))
    print(f"Data: train v={len(tr['valid'])} c={len(tr['corrupt'])} val v={len(va['valid'])} c={len(va['corrupt'])} test v={len(te['valid'])} c={len(te['corrupt'])} ({time.time()-t0:.1f}s)")
    print("Train Oracle..."); t0=time.time(); mo=train_oracle(tr); print(f"  {time.time()-t0:.1f}s")
    print("Pretrain (oracle-slot distillation)..."); t0=time.time()
    enc,dec=pretrain_distill_oracle(tr, epochs=100); print(f"  {time.time()-t0:.1f}s")
    print("Train Video IWCM on distilled slots..."); t0=time.time()
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
