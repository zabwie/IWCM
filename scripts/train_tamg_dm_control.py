"""Train TAMGSlotEncoder + FusedIWCMEnergy on DM Control rendered frames.
Self-supervised: velocity MSE, contrastive identity, clustering type, reconstruction.
IWCM margin loss on corrupted-vs-valid slot trajectories.

Ponytail: loop over 3 domains, no config framework, inline everything.
Script: ~110 lines for a full 3-domain training + evaluation pipeline.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, torch.nn.functional as F
import numpy as np
from collections import defaultdict
from sklearn.metrics import roc_auc_score
from src.env.dm_control_wrapper import DMControlWrapper
from src.env.dm_control_encoder import DMControlOracleEncoder
from src.tamg.slot_encoder import TAMGSlotEncoder, corrupt_tamg_slots
from src.iwcm.fused_energy import FusedIWCMEnergy

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
H, N_TRAIN, N_TEST, EPOCHS, LR = 25, 400, 80, 150, 3e-3
CORRUPTIONS_PER_TRAJ = 3
DOMAINS = [('cartpole', 'swingup', 1), ('cheetah', 'run', 6), ('walker', 'walk', 6)]

def generate_data(n_traj, horizon, domain, task, seed=42):
    rng = np.random.RandomState(seed)
    w = DMControlWrapper(domain, task, seed=int(rng.randint(0, 2**31)))
    enc_oracle = DMControlOracleEncoder(domain)
    data = []
    for _ in range(n_traj * 2):  # generous overshoot
        if len(data) >= n_traj: break
        traj = w.generate_trajectory(horizon)
        if traj is None: continue
        physics_states, _, actions = traj
        frames = np.stack([w._env.physics.render(camera_id=0, height=64, width=64)
                           for _ in range(horizon + 1)], axis=0)
        enc = enc_oracle.encode_trajectory(physics_states, actions, horizon)
        if enc is None: continue
        z0_o, A_o, Z_o = enc
        data.append((frames, z0_o, A_o, Z_o))
    return data[:n_traj]

def prep_frames(data):
    frames_list, act_list = [], []
    for frames_np, _, A_np, _ in data:
        f = torch.from_numpy(frames_np).float().permute(0, 3, 1, 2) / 255.0
        frames_list.append(f.unsqueeze(0))
        act_list.append(torch.from_numpy(np.array(A_np)).float().unsqueeze(0))
    return torch.cat(frames_list, dim=0), torch.cat(act_list, dim=0)

def get_energies(enc, energy_fn, data, data_frames, data_acts, H):
    valid_E, corr_E = [], []
    rng_eval = np.random.RandomState(99)
    for i in range(len(data)):
        frames_b = data_frames[i:i+1].to(DEV)
        A_b = data_acts[i:i+1].to(DEV)
        with torch.no_grad():
            slot_list = []
            for t in range(H):
                Ft = frames_b[:, t]; Ft1 = frames_b[:, t + 1]
                slots_t, _ = enc(Ft, Ft1); slot_list.append(slots_t)
            Z = torch.stack(slot_list, dim=1); z0 = Z[:, 0]
            valid_E.append(energy_fn(z0, A_b, Z).item())
            Z_np = Z.cpu().numpy()
            for _ in range(3):
                Zc_np = corrupt_tamg_slots(Z_np.copy(), rng_eval)
                corr_E.append(energy_fn(z0, A_b, torch.from_numpy(Zc_np).float().to(DEV)).item())
    return valid_E, corr_E

for domain, task, da in DOMAINS:
    print(f"\n{'='*60}")
    print(f"Training TAMG on {domain}/{task}")
    print(f"{'='*60}")

    all_data = generate_data(N_TRAIN + N_TEST, H, domain, task)
    train_data, test_data = all_data[:N_TRAIN], all_data[N_TRAIN:]
    train_frames, train_acts = prep_frames(train_data)
    test_frames, test_acts = prep_frames(test_data)

    enc = TAMGSlotEncoder(num_slots=8, d_feat=64, img_size=64).to(DEV)
    energy_fn = FusedIWCMEnergy(d_slot=19, d_action=da, hidden=128, num_slots=8).to(DEV)
    opt = torch.optim.Adam(list(enc.parameters()) + list(energy_fn.parameters()), lr=LR)
    rng = np.random.RandomState(42)

    for ep in range(1, EPOCHS + 1):
        idxs = np.random.permutation(N_TRAIN)
        losses = defaultdict(list)
        for start in range(0, N_TRAIN, 32):
            idx = idxs[start:start + 32]
            frames_b = train_frames[idx].to(DEV)
            A_b = train_acts[idx].to(DEV)
            slot_list = []
            for t in range(H):
                Ft, Ft1 = frames_b[:, t], frames_b[:, t + 1]
                slots_t, slosses = enc(Ft, Ft1)
                slot_list.append(slots_t)
            Z = torch.stack(slot_list, dim=1)
            z0 = Z[:, 0]

            loss_vel = torch.stack([slosses['loss_vel'] for _ in range(H)]).mean()
            loss_id = torch.stack([slosses['loss_id'] for _ in range(H)]).mean()
            loss_type = torch.stack([slosses['loss_type'] for _ in range(H)]).mean()
            loss_recon = torch.stack([slosses['loss_recon'] for _ in range(H)]).mean()

            Z_np = Z.detach().cpu().numpy()
            Zc_list = []
            for _ in range(CORRUPTIONS_PER_TRAJ):
                Zc_list.append(torch.from_numpy(corrupt_tamg_slots(Z_np, rng)).float().to(DEV))
            Zc = torch.cat(Zc_list, dim=0)

            # ponytail: detach Z from energy to avoid competing gradients with self-supervised losses
            Ev = energy_fn(z0, A_b, Z.detach())
            z0_rep = z0.unsqueeze(0).expand(CORRUPTIONS_PER_TRAJ, -1, -1, -1).reshape(-1, 8, 19)
            A_rep = A_b.unsqueeze(0).expand(CORRUPTIONS_PER_TRAJ, -1, -1, -1).reshape(-1, H, da)
            Ec = energy_fn(z0_rep, A_rep, Zc)
            loss_margin = F.relu(Ev + 0.5).mean() + F.relu(Ev.mean() + 1.0 - Ec).mean()
            loss_total = loss_margin + 0.1 * loss_vel + 0.1 * loss_id + 0.05 * loss_type + 0.1 * loss_recon
            opt.zero_grad(); loss_total.backward(); opt.step()
            for k, v in [('margin', loss_margin), ('vel', loss_vel), ('id', loss_id),
                         ('type', loss_type), ('recon', loss_recon)]:
                losses[k].append(v.item())

        if ep % 25 == 0 or ep == EPOCHS:
            print(f"  ep {ep:3d}: margin={np.mean(losses['margin']):.4f} vel={np.mean(losses['vel']):.4f} "
                  f"id={np.mean(losses['id']):.4f} type={np.mean(losses['type']):.4f} recon={np.mean(losses['recon']):.4f}")

    enc.eval(); energy_fn.eval()
    valid_E, corr_E = get_energies(enc, energy_fn, test_data, test_frames, test_acts, H)
    auroc = roc_auc_score([0]*len(valid_E)+[1]*len(corr_E), valid_E+corr_E)
    print(f"  >>> {domain}/{task:8s} AUROC: {auroc:.3f}  (valid E={np.mean(valid_E):.2f}, corr E={np.mean(corr_E):.2f})")
