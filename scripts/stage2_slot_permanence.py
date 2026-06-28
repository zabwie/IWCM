#!/usr/bin/env python3
"""Stage 2.2: SlotPermanenceEncoder — stable temporal slot identity for video IWCM.

Compares four approaches on grid-world rendered videos:
  1. Recon-only baseline (parallel slot attention, no temporal tracking)
  2. Temporal propagation (init from previous frame, no transition model)
  3. SlotPermanenceEncoder (transition + spatial anchor + content loss)
  4. Oracle slots (symbolic ground truth, upper bound)

Training: multi-phase (recon → permanence → IWCM margin)
Evaluation: switch rate, AUROC, balanced accuracy vs oracle.
"""

import sys, time, numpy as np, torch, torch.nn.functional as F
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.seed import set_seed
from src.encoder.oracle_slot_encoder import (
    encode_oracle_trajectory, build_door_key_map,
    ORACLE_SLOT_DIM, MAX_OBJECTS,
)
from src.encoder import (
    VideoEncoder, SlotPermanenceEncoder,
    content_smoothness_loss, slot_diversity_loss,
    transition_consistency_loss, compute_slot_switch_rate,
)
from src.encoder.slot_structure import SlotStructureHead, structure_loss, extract_oracle_targets
from src.encoder.decoder import VideoDecoder
from src.iwcm.fused_energy import FusedIWCMEnergy
from src.env.grid_world import GridWorld
from src.env.renderer import GridWorldRenderer
from src.env.scenarios import Scenario

# ─── Configuration ──────────────────────────────────────────────────────────
HORIZON, GRID_SIZE, FRAME_SIZE, CELL_PX = 25, 8, 64, 8
SLOT_DIM = ORACLE_SLOT_DIM  # 19 — matches oracle for direct distillation
CONTENT_DIM = 14  # first 14 channels: type+pos+velocity+flags
NUM_TRAIN, NUM_TEST, NUM_VAL = 200, 50, 30
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VTYPES = ["delete", "swap", "teleport", "duplicate", "transform"]

# Training hyperparameters
PHASE1_EPOCHS = 50     # reconstruction
PHASE2_EPOCHS = 100    # + oracle distillation + permanence
PHASE25_EPOCHS = 100   # IWCM-only training (frozen encoder)
PHASE3_EPOCHS = 150    # alternating encoder/IWCM
LR = 3e-4
LR_IWCM = 1e-3
DISTILL_WEIGHT = 1.0   # primary loss: MSE vs oracle slots
CONTENT_LOSS_WEIGHT = 0.3
DIVERSITY_LOSS_WEIGHT = 0.05
TRANSITION_LOSS_WEIGHT = 0.2
IWCM_MARGIN_WEIGHT = 1.0
RECON_WEIGHT = 0.3
ALTERNATE_STEPS = 5


# ─── Data Generation ────────────────────────────────────────────────────────

def gen_traj(scenario, rng, corrupt=None):
    gw = GridWorld(grid_size=GRID_SIZE, objects_config=scenario.to_env_config(),
                   seed=int(rng.randint(0, 2 ** 31)))
    gw.reset()
    states = [dict(gw.get_state())]
    acts = []
    for _ in range(HORIZON + 10):
        va = gw.get_valid_actions()
        if not va:
            break
        a = int(rng.choice(va))
        s, _, done, _ = gw.step(a)
        states.append(dict(s))
        acts.append(a)
        if done:
            break
    if len(states) < HORIZON + 1:
        return None
    actions = np.array(acts[:HORIZON], dtype=np.int64)
    meta = {"law_type": "valid", "violation_type": "none"}
    if corrupt:
        tc = rng.randint(HORIZON // 4, 3 * HORIZON // 4)
        oids = list(states[0].get("objects", {}).keys())
        if not oids:
            return None
        tgt = oids[rng.randint(0, len(oids))]
        if corrupt == "delete":
            for t in range(tc, HORIZON + 1):
                states[t].setdefault("objects", {}).pop(tgt, None)
        elif corrupt == "swap" and len(oids) >= 2:
            o2 = oids[rng.randint(0, len(oids))]
            while o2 == tgt:
                o2 = oids[rng.randint(0, len(oids))]
            for t in range(tc, HORIZON + 1):
                ob = states[t].setdefault("objects", {})
                if tgt in ob and o2 in ob:
                    ob[tgt], ob[o2] = dict(ob[o2]), dict(ob[tgt])
        elif corrupt == "teleport":
            nr, nc = rng.randint(0, GRID_SIZE - 1), rng.randint(0, GRID_SIZE - 1)
            for t in range(tc, HORIZON + 1):
                if tgt in states[t].get("objects", {}):
                    states[t]["objects"][tgt]["pos"] = (nr, nc)
        elif corrupt == "duplicate":
            ni = tgt + "_dup"
            for t in range(tc, HORIZON + 1):
                if tgt in states[t].get("objects", {}):
                    states[t]["objects"][ni] = dict(states[t]["objects"][tgt])
        elif corrupt == "transform":
            for t in range(tc, HORIZON + 1):
                if tgt in states[t].get("objects", {}):
                    states[t]["objects"][tgt]["type"] = "box"
        meta = {
            "law_type": "conservation" if corrupt in ["delete", "duplicate", "transform"] else "identity",
            "violation_type": corrupt,
        }
    return states[:HORIZON + 1], actions, meta


def render_traj(states, renderer):
    return torch.stack([
        torch.from_numpy(renderer.render_frame(s)).float().permute(2, 0, 1) / 255.0
        for s in states
    ])


def oracle_encode(states, actions, scenario):
    dkm = build_door_key_map(scenario.to_env_config())
    goal = scenario.to_env_config().get("goal", None)
    return encode_oracle_trajectory(states, actions, HORIZON, GRID_SIZE, goal, dkm)


def gen_dataset(nv, nc, rng):
    scenarios = [Scenario.from_preset(n, GRID_SIZE) for n in ["key_door_simple", "multi_object"]]
    renderer = GridWorldRenderer(grid_size=GRID_SIZE, cell_px=CELL_PX)
    vd, cd = [], []
    for _ in range(nv):
        sc = scenarios[rng.randint(0, len(scenarios))]
        r = gen_traj(sc, rng)
        if r is None:
            continue
        st, ac, m = r
        orc = oracle_encode(st, ac, sc)
        if orc is None:
            continue
        vd.append((render_traj(st, renderer), orc, ac, m))
    for _ in range(nc):
        sc = scenarios[rng.randint(0, len(scenarios))]
        vt = VTYPES[rng.randint(0, len(VTYPES))]
        r = gen_traj(sc, rng, vt)
        if r is None:
            continue
        st, ac, m = r
        orc = oracle_encode(st, ac, sc)
        if orc is None:
            continue
        cd.append((render_traj(st, renderer), orc, ac, m))
    return {"valid": vd, "corrupt": cd}


# ─── Oracle Baseline Training ───────────────────────────────────────────────

def train_oracle(train_data, epochs=300):
    m = FusedIWCMEnergy(d_slot=ORACLE_SLOT_DIM, d_action=11, hidden=128,
                        num_slots=MAX_OBJECTS).to(DEVICE)
    opt = torch.optim.Adam(m.parameters(), lr=LR)
    tv = [(torch.from_numpy(z0).float(), torch.from_numpy(A).float(),
           torch.from_numpy(Z).float())
          for _, (z0, A, Z), _, _ in train_data["valid"]]
    tc = [(torch.from_numpy(z0).float(), torch.from_numpy(A).float(),
           torch.from_numpy(Z).float(), meta)
          for _, (z0, A, Z), _, meta in train_data["corrupt"]]
    n = min(len(tv), len(tc))
    for ep in range(epochs):
        vi = np.random.choice(len(tv), n, replace=False)
        ci = np.random.choice(len(tc), n, replace=False)
        vz0 = torch.stack([tv[i][0] for i in vi]).to(DEVICE)
        cz0 = torch.stack([tc[i][0] for i in ci]).to(DEVICE)
        vA = torch.stack([tv[i][1] for i in vi]).to(DEVICE)
        cA = torch.stack([tc[i][1] for i in ci]).to(DEVICE)
        vZ = torch.stack([tv[i][2] for i in vi]).to(DEVICE)
        cZ = torch.stack([tc[i][2] for i in ci]).to(DEVICE)
        opt.zero_grad()
        ev = m(vz0, vA, vZ)
        ec = m(cz0, cA, cZ)
        loss = F.relu(ev + 1.0).mean() + F.relu(1.0 - ec).mean() + \
               0.001 * (ev.pow(2).mean() + ec.pow(2).mean())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
    m.eval()
    return m


# ─── Video Encoder Training ─────────────────────────────────────────────────

def train_slot_permanence(train_data):
    """Multi-phase: recon → distill+permanence → IWCM-solo → alternating."""
    enc = SlotPermanenceEncoder(
        frame_size=FRAME_SIZE, in_channels=3, num_slots=MAX_OBJECTS,
        slot_dim=SLOT_DIM, content_dim=CONTENT_DIM,
        slot_iters=3, d_action=11, anchor_beta=10.0, anchor_beta_anneal=2.0,
    ).to(DEVICE)
    dec = VideoDecoder(slot_dim=SLOT_DIM, frame_size=FRAME_SIZE, out_channels=3).to(DEVICE)
    iw = FusedIWCMEnergy(d_slot=SLOT_DIM, d_action=11, hidden=128,
                         num_slots=MAX_OBJECTS).to(DEVICE)

    opt_enc = torch.optim.Adam(
        list(enc.parameters()) + list(dec.parameters()), lr=LR)
    opt_iwcm = torch.optim.Adam(iw.parameters(), lr=LR_IWCM)

    n = min(len(train_data["valid"]), len(train_data["corrupt"]), 32)
    nc = min(n, len(train_data["corrupt"]))

    def get_oracle_Z(indices, data_key="valid"):
        oracles = [train_data[data_key][i][1] for i in indices]
        return extract_oracle_targets(oracles, HORIZON, DEVICE)

    # ═══ Phase 1: Reconstruction ═══
    print(f"\n  Phase 1 ({PHASE1_EPOCHS} epochs): Recon")
    for ep in range(PHASE1_EPOCHS):
        vi = np.random.choice(len(train_data["valid"]), n, replace=False)
        vf = torch.stack([train_data["valid"][i][0][:HORIZON] for i in vi]).to(DEVICE)
        B, H, C, W, H_img = vf.shape
        actions_t = torch.stack([
            torch.from_numpy(train_data["valid"][i][2][:HORIZON]).float()
            for i in vi
        ]).to(DEVICE)
        slots = enc(vf, actions_t)
        slots_flat = slots.reshape(B * H, MAX_OBJECTS, SLOT_DIM)
        recon = dec.decode_frame(slots_flat)
        loss = F.mse_loss(recon, vf.reshape(B * H, C, W, H_img))
        opt_enc.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(enc.parameters(), 1.0)
        opt_enc.step()
        enc.step_anchor_beta()
        if (ep + 1) % 20 == 0:
            print(f"    ep {ep+1:3d}: recon={loss.item():.4f}")

    # ═══ Phase 2: Oracle distillation + permanence (valid + corrupt) ═══
    print(f"\n  Phase 2 ({PHASE2_EPOCHS} epochs): Distill + Permanence")
    for ep in range(PHASE2_EPOCHS):
        vi = np.random.choice(len(train_data["valid"]), n, replace=False)
        ci = np.random.choice(len(train_data["corrupt"]), nc, replace=False)

        # Valid batch
        vf = torch.stack([train_data["valid"][i][0][:HORIZON] for i in vi]).to(DEVICE)
        Bv, Hv, C, W, H_img = vf.shape
        vA = torch.stack([
            torch.from_numpy(train_data["valid"][i][2][:HORIZON]).float()
            for i in vi
        ]).to(DEVICE)
        oracle_V = get_oracle_Z(vi, "valid")
        vs = enc(vf, vA)

        # Corrupt batch — ALSO distill!
        cf = torch.stack([train_data["corrupt"][i][0][:HORIZON] for i in ci]).to(DEVICE)
        Bc, Hc = cf.shape[0], cf.shape[1]
        cA = torch.stack([
            torch.from_numpy(train_data["corrupt"][i][2][:HORIZON]).float()
            for i in ci
        ]).to(DEVICE)
        oracle_C = get_oracle_Z(ci, "corrupt")
        cs = enc(cf, cA)

        # Oracle distillation on BOTH valid and corrupt
        loss_distill = DISTILL_WEIGHT * (
            F.mse_loss(vs, oracle_V) + F.mse_loss(cs, oracle_C))

        # Reconstruction (valid only — for visual slot quality)
        vs_flat = vs.reshape(Bv * Hv, MAX_OBJECTS, SLOT_DIM)
        recon = dec.decode_frame(vs_flat)
        loss_recon = RECON_WEIGHT * F.mse_loss(recon, vf.reshape(Bv * Hv, C, W, H_img))

        # Permanence (valid only — corrupt trajectories should NOT be smoothed)
        loss_content = CONTENT_LOSS_WEIGHT * content_smoothness_loss(vs, CONTENT_DIM)
        loss_diversity = DIVERSITY_LOSS_WEIGHT * slot_diversity_loss(vs, CONTENT_DIM, margin=0.3)
        loss_trans = TRANSITION_LOSS_WEIGHT * transition_consistency_loss(vs, enc.transition, vA)

        loss = loss_distill + loss_recon + loss_content + loss_diversity + loss_trans
        opt_enc.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(enc.parameters()) + list(dec.parameters()), 1.0)
        opt_enc.step()
        enc.step_anchor_beta()

        if (ep + 1) % 40 == 0:
            with torch.no_grad():
                sr = compute_slot_switch_rate(vs, CONTENT_DIM)
            print(f"    ep {ep+1:3d}: distill={loss_distill.item():.4f} "
                  f"recon={loss_recon.item():.4f} sr={sr:.3f}")

    # ═══ Phase 2.5: IWCM-only (encoder frozen) ═══
    print(f"\n  Phase 2.5 ({PHASE25_EPOCHS} epochs): IWCM-only (encoder frozen)")
    enc.eval()
    for p in enc.parameters():
        p.requires_grad = False
    for ep in range(PHASE25_EPOCHS):
        vi = np.random.choice(len(train_data["valid"]), n, replace=False)
        ci = np.random.choice(len(train_data["corrupt"]), nc, replace=False)
        with torch.no_grad():
            vf = torch.stack([train_data["valid"][i][0][:HORIZON] for i in vi]).to(DEVICE)
            vA = torch.stack([torch.from_numpy(train_data["valid"][i][2][:HORIZON]).float() for i in vi]).to(DEVICE)
            vs = enc(vf, vA)
            cf = torch.stack([train_data["corrupt"][i][0][:HORIZON] for i in ci]).to(DEVICE)
            cA = torch.stack([torch.from_numpy(train_data["corrupt"][i][2][:HORIZON]).float() for i in ci]).to(DEVICE)
            cs = enc(cf, cA)
        ev = iw(vs[:, 0], vA, vs); ec = iw(cs[:, 0], cA, cs)
        loss = F.relu(ev + 1.0).mean() + F.relu(1.0 - ec).mean() + 0.001 * (ev.pow(2).mean() + ec.pow(2).mean())
        opt_iwcm.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(iw.parameters(), 1.0)
        opt_iwcm.step()
        if (ep + 1) % 40 == 0:
            print(f"    ep {ep+1:3d}: loss={loss.item():.4f} ev={ev.mean().item():.2f} ec={ec.mean().item():.2f}")

    enc.train()
    for p in enc.parameters():
        p.requires_grad = True

    # ═══ Phase 3: Alternating ═══
    print(f"\n  Phase 3 ({PHASE3_EPOCHS} epochs): Alternating")
    for ep in range(PHASE3_EPOCHS):
        vi = np.random.choice(len(train_data["valid"]), n, replace=False)
        ci = np.random.choice(len(train_data["corrupt"]), nc, replace=False)
        vf = torch.stack([train_data["valid"][i][0][:HORIZON] for i in vi]).to(DEVICE)
        vA = torch.stack([torch.from_numpy(train_data["valid"][i][2][:HORIZON]).float() for i in vi]).to(DEVICE)
        cf = torch.stack([train_data["corrupt"][i][0][:HORIZON] for i in ci]).to(DEVICE)
        cA = torch.stack([torch.from_numpy(train_data["corrupt"][i][2][:HORIZON]).float() for i in ci]).to(DEVICE)
        oracle_Z = get_oracle_Z(vi)

        if (ep // ALTERNATE_STEPS) % 2 == 0:
            vs = enc(vf, vA).detach(); cs = enc(cf, cA).detach()
            ev = iw(vs[:, 0], vA, vs); ec = iw(cs[:, 0], cA, cs)
            loss = F.relu(ev + 1.0).mean() + F.relu(1.0 - ec).mean() + 0.001 * (ev.pow(2).mean() + ec.pow(2).mean())
            opt_iwcm.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(iw.parameters(), 1.0)
            opt_iwcm.step()
        else:
            vs = enc(vf, vA); cs = enc(cf, cA)
            Bv, Hv = vs.shape[0], vs.shape[1]
            loss_distill = DISTILL_WEIGHT * F.mse_loss(vs, oracle_Z)
            vs_flat = vs.reshape(Bv * Hv, MAX_OBJECTS, SLOT_DIM)
            recon = dec.decode_frame(vs_flat)
            loss_recon = RECON_WEIGHT * F.mse_loss(recon, vf.reshape(Bv * Hv, C, W, H_img))
            ev = iw(vs[:, 0], vA, vs); ec = iw(cs[:, 0], cA, cs)
            loss_iwcm = F.relu(ev + 1.0).mean() + F.relu(1.0 - ec).mean()
            loss_content = CONTENT_LOSS_WEIGHT * content_smoothness_loss(vs, CONTENT_DIM)
            loss_diversity = DIVERSITY_LOSS_WEIGHT * slot_diversity_loss(vs, CONTENT_DIM, margin=0.3)
            loss = loss_distill + loss_recon + loss_iwcm + loss_content + loss_diversity
            opt_enc.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(list(enc.parameters()) + list(dec.parameters()), 1.0)
            opt_enc.step()
            enc.step_anchor_beta()

        if (ep + 1) % 50 == 0:
            with torch.no_grad():
                etv = iw(vs[:, 0], vA, vs).mean().item()
                etc = iw(cs[:, 0], cA, cs).mean().item()
                sr = compute_slot_switch_rate(vs, CONTENT_DIM)
            print(f"    ep {ep+1:3d}: ev={etv:.2f} ec={etc:.2f} sr={sr:.3f}")

    enc.eval(); dec.eval(); iw.eval()
    return enc, dec, iw


# ─── Baselines ──────────────────────────────────────────────────────────────

def train_baseline_recon(train_data, epochs=PHASE1_EPOCHS):
    """Baseline: standard VideoEncoder with reconstruction only."""
    enc = VideoEncoder(frame_size=FRAME_SIZE, in_channels=3, num_slots=MAX_OBJECTS,
                       slot_dim=SLOT_DIM).to(DEVICE)
    dec = VideoDecoder(slot_dim=SLOT_DIM, frame_size=FRAME_SIZE, out_channels=3).to(DEVICE)
    opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=LR)
    n = min(len(train_data["valid"]), 32)
    for ep in range(epochs):
        vi = np.random.choice(len(train_data["valid"]), n, replace=False)
        vf = torch.stack([train_data["valid"][i][0] for i in vi]).to(DEVICE)
        B, H, C, W, H_img = vf.shape
        frames_flat = vf.reshape(B * H, C, W, H_img)
        slots_flat = enc.encode_frame(frames_flat)
        recon_flat = dec.decode_frame(slots_flat)
        loss = F.mse_loss(recon_flat, frames_flat)
        opt.zero_grad()
        loss.backward()
        opt.step()
    enc.eval()
    return enc


def train_baseline_temporal(train_data, encoder=None, epochs=PHASE1_EPOCHS):
    """Baseline: VideoEncoder with temporal slot propagation."""
    enc = encoder if encoder is not None else VideoEncoder(
        frame_size=FRAME_SIZE, in_channels=3, num_slots=MAX_OBJECTS,
        slot_dim=SLOT_DIM).to(DEVICE)
    iw = FusedIWCMEnergy(d_slot=SLOT_DIM, d_action=11, hidden=128,
                         num_slots=MAX_OBJECTS).to(DEVICE)
    opt = torch.optim.Adam(list(enc.parameters()) + list(iw.parameters()), lr=LR)
    n = min(len(train_data["valid"]), len(train_data["corrupt"]), 32)
    for ep in range(epochs):
        vi = np.random.choice(len(train_data["valid"]), min(n, len(train_data["valid"])), replace=False)
        ci = np.random.choice(len(train_data["corrupt"]), min(n, len(train_data["corrupt"])), replace=False)
        vf = torch.stack([train_data["valid"][i][0][:HORIZON] for i in vi]).to(DEVICE)
        vA = torch.stack([
            torch.from_numpy(train_data["valid"][i][2][:HORIZON]).float()
            for i in vi
        ]).to(DEVICE)
        vs = enc.forward_temporal(vf)
        vz = vs[:, 0]
        cf = torch.stack([train_data["corrupt"][i][0][:HORIZON] for i in ci]).to(DEVICE)
        cA = torch.stack([
            torch.from_numpy(train_data["corrupt"][i][2][:HORIZON]).float()
            for i in ci
        ]).to(DEVICE)
        cs = enc.forward_temporal(cf)
        cz = cs[:, 0]
        opt.zero_grad()
        ev = iw(vz, vA, vs)
        ec = iw(cz, cA, cs)
        loss = F.relu(ev + 1.0).mean() + F.relu(1.0 - ec).mean() + \
               0.001 * (ev.pow(2).mean() + ec.pow(2).mean())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(enc.parameters()) + list(iw.parameters()), 1.0)
        opt.step()
    enc.eval()
    iw.eval()
    return enc, iw


# ─── Evaluation ─────────────────────────────────────────────────────────────

def compute_energy_metrics(model_or_enc, data, iwcm=None, use_oracle=True):
    energies, labels = [], []
    for frames, oracle, actions, meta in data["valid"]:
        if use_oracle:
            z0, A, Z = oracle
            e = model_or_enc(
                torch.from_numpy(z0).float().unsqueeze(0).to(DEVICE),
                torch.from_numpy(A).float().unsqueeze(0).to(DEVICE),
                torch.from_numpy(Z).float().unsqueeze(0).to(DEVICE),
            ).item()
        else:
            f = frames.unsqueeze(0).to(DEVICE)
            a_t = torch.from_numpy(actions[:HORIZON]).float().unsqueeze(0).to(DEVICE)
            slots = model_or_enc(f, a_t)
            e = iwcm(slots[:, 0], a_t, slots).item()
        energies.append(e)
        labels.append(0)
    for frames, oracle, actions, meta in data["corrupt"]:
        if use_oracle:
            z0, A, Z = oracle
            e = model_or_enc(
                torch.from_numpy(z0).float().unsqueeze(0).to(DEVICE),
                torch.from_numpy(A).float().unsqueeze(0).to(DEVICE),
                torch.from_numpy(Z).float().unsqueeze(0).to(DEVICE),
            ).item()
        else:
            f = frames.unsqueeze(0).to(DEVICE)
            a_t = torch.from_numpy(actions[:HORIZON]).float().unsqueeze(0).to(DEVICE)
            slots = model_or_enc(f, a_t)
            e = iwcm(slots[:, 0], a_t, slots).item()
        energies.append(e)
        labels.append(1)
    energies = np.array(energies)
    labels = np.array(labels)
    from sklearn.metrics import roc_auc_score
    auroc = roc_auc_score(labels, energies)
    return {"energies": energies, "labels": labels, "auroc": auroc}


def calibrate(val_e, val_l, test_e, test_l):
    best_t, best_b = 0, 0
    for t in np.linspace(val_e.min(), val_e.max(), 200):
        p = val_e > t
        va = ((p == 0) & (val_l == 0)).sum() / max((val_l == 0).sum(), 1)
        ir = ((p == 1) & (val_l == 1)).sum() / max((val_l == 1).sum(), 1)
        b = (va + ir) / 2
        if b > best_b:
            best_b, best_t = b, t
    pt = test_e > best_t
    va = ((pt == 0) & (test_l == 0)).sum() / max((test_l == 0).sum(), 1)
    ir = ((pt == 1) & (test_l == 1)).sum() / max((test_l == 1).sum(), 1)
    return {"threshold": best_t, "valid_acc": va, "invalid_rej": ir,
            "balanced_acc": (va + ir) / 2}


def compute_encoder_switch_rate(encoder, test_data, n_samples=8):
    """Measure slot switch rate on test trajectories."""
    with torch.no_grad():
        indices = list(range(min(n_samples, len(test_data["valid"]))))
        vf = torch.stack([test_data["valid"][i][0][:HORIZON] for i in indices]).to(DEVICE)
        actions_t = torch.stack([
            torch.from_numpy(test_data["valid"][i][2][:HORIZON]).float()
            for i in indices
        ]).to(DEVICE)
        slots = encoder(vf, actions_t)
        sr = compute_slot_switch_rate(slots, CONTENT_DIM)
    return sr


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    set_seed(42)
    rng = np.random.RandomState(42)
    print("STAGE 2.2 — SlotPermanenceEncoder")
    print("=" * 65)

    # Generate data
    t0 = time.time()
    train_data = gen_dataset(NUM_TRAIN, NUM_TRAIN, rng)
    val_data = gen_dataset(NUM_VAL, NUM_VAL, np.random.RandomState(99))
    test_data = gen_dataset(NUM_TEST, NUM_TEST, np.random.RandomState(123))
    print(f"Data: train v={len(train_data['valid'])} c={len(train_data['corrupt'])} "
          f"val v={len(val_data['valid'])} c={len(val_data['corrupt'])} "
          f"test v={len(test_data['valid'])} c={len(test_data['corrupt'])} "
          f"({time.time() - t0:.1f}s)")

    # ── Oracle baseline ──
    print("\n[1/5] Training Oracle IWCM...")
    t0 = time.time()
    oracle_iwcm = train_oracle(train_data)
    print(f"  Done ({time.time() - t0:.1f}s)")

    # ── Baseline: Recon only ──
    print("\n[2/5] Training Recon-Only Baseline...")
    t0 = time.time()
    enc_recon = train_baseline_recon(train_data)
    iw_recon = FusedIWCMEnergy(
        d_slot=SLOT_DIM, d_action=11, hidden=128, num_slots=MAX_OBJECTS,
    ).to(DEVICE)
    sr_recon = compute_encoder_switch_rate(
        lambda v, a: enc_recon.forward_temporal(v), test_data)
    print(f"  Switch rate: {sr_recon:.3f} ({time.time() - t0:.1f}s)")

    # ── Baseline: Temporal propagation ──
    print("\n[3/5] Training Temporal-Propagation Baseline...")
    t0 = time.time()
    enc_temp, iw_temp = train_baseline_temporal(train_data)
    sr_temp = compute_encoder_switch_rate(
        lambda v, a: enc_temp.forward_temporal(v), test_data)
    print(f"  Switch rate: {sr_temp:.3f} ({time.time() - t0:.1f}s)")

    # ── SlotPermanenceEncoder ──
    print("\n[4/5] Training SlotPermanenceEncoder...")
    t0 = time.time()
    enc_perm, dec_perm, iw_perm = train_slot_permanence(train_data)
    sr_perm = compute_encoder_switch_rate(enc_perm, test_data)
    print(f"  Switch rate: {sr_perm:.3f} ({time.time() - t0:.1f}s)")

    # ── Evaluate all ──
    print("\n[5/5] Evaluating...")
    # Oracle
    mo_val = compute_energy_metrics(oracle_iwcm, val_data, use_oracle=True)
    mo_test = compute_energy_metrics(oracle_iwcm, test_data, use_oracle=True)

    # Recon only
    mr_val = compute_energy_metrics(
        lambda v, a: enc_recon.forward_temporal(v), val_data,
        iwcm=iw_recon, use_oracle=False)
    mr_test = compute_energy_metrics(
        lambda v, a: enc_recon.forward_temporal(v), test_data,
        iwcm=iw_recon, use_oracle=False)

    # Temporal
    mt_val = compute_energy_metrics(
        lambda v, a: enc_temp.forward_temporal(v), val_data,
        iwcm=iw_temp, use_oracle=False)
    mt_test = compute_energy_metrics(
        lambda v, a: enc_temp.forward_temporal(v), test_data,
        iwcm=iw_temp, use_oracle=False)

    # Permanence
    mp_val = compute_energy_metrics(enc_perm, val_data, iwcm=iw_perm, use_oracle=False)
    mp_test = compute_energy_metrics(enc_perm, test_data, iwcm=iw_perm, use_oracle=False)

    # Calibrate
    co = calibrate(mo_val["energies"], mo_val["labels"],
                   mo_test["energies"], mo_test["labels"])
    cr = calibrate(mr_val["energies"], mr_val["labels"],
                   mr_test["energies"], mr_test["labels"])
    ct = calibrate(mt_val["energies"], mt_val["labels"],
                   mt_test["energies"], mt_test["labels"])
    cp = calibrate(mp_val["energies"], mp_val["labels"],
                   mp_test["energies"], mp_test["labels"])

    # ── Results ──
    print("\n" + "=" * 75)
    print(f"{'Metric':<22} {'Oracle':>10} {'Recon':>10} {'Temporal':>10} {'Permanence':>12}")
    print("-" * 75)
    rows = [
        ("Switch Rate", f"{0.0:.3f}", f"{sr_recon:.3f}", f"{sr_temp:.3f}", f"{sr_perm:.3f}"),
        ("AUROC", f"{mo_test['auroc']:.3f}", f"{mr_test['auroc']:.3f}",
         f"{mt_test['auroc']:.3f}", f"{mp_test['auroc']:.3f}"),
        ("Valid Accuracy", f"{co['valid_acc']:.3f}", f"{cr['valid_acc']:.3f}",
         f"{ct['valid_acc']:.3f}", f"{cp['valid_acc']:.3f}"),
        ("Invalid Rejection", f"{co['invalid_rej']:.3f}", f"{cr['invalid_rej']:.3f}",
         f"{ct['invalid_rej']:.3f}", f"{cp['invalid_rej']:.3f}"),
        ("Balanced Accuracy", f"{co['balanced_acc']:.3f}", f"{cr['balanced_acc']:.3f}",
         f"{ct['balanced_acc']:.3f}", f"{cp['balanced_acc']:.3f}"),
    ]
    for name, o, r, t, p in rows:
        print(f"{name:<22} {o:>10} {r:>10} {t:>10} {p:>12}")

    # Energy distribution
    print(f"\n{'Model':<22} {'E_valid':>10} {'E_invalid':>10} {'Overlap':>10}")
    print("-" * 55)
    for name, met in [("Oracle", mo_test), ("Recon", mr_test),
                       ("Temporal", mt_test), ("Permanence", mp_test)]:
        ev = met["energies"][met["labels"] == 0]
        ei = met["energies"][met["labels"] == 1]
        overlap = (ev > ei.min()).mean() if len(ei) > 0 else 1.0
        print(f"{name:<22} {ev.mean():>10.2f} {ei.mean():>10.2f} {overlap:>10.2f}")

    # Verdict
    print("\nVerdict: ", end="")
    if cp["balanced_acc"] >= 0.75:
        print("SLOT PERMANENCE VALIDATED — AUROC %.3f meets threshold" % mp_test["auroc"])
    elif mp_test["auroc"] >= 0.75:
        print("PROMISING — AUROC %.3f approaching target. Increase epochs or data." % mp_test["auroc"])
    elif mp_test["auroc"] >= 0.70:
        print("MODERATE — AUROC %.3f. Switch rate still %.3f. Consider stronger content loss." % (
            mp_test["auroc"], sr_perm))
    else:
        print("WEAK — AUROC %.3f. Switch rate %.3f. Slot identity not yet stable." % (
            mp_test["auroc"], sr_perm))


if __name__ == "__main__":
    main()
