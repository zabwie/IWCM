#!/usr/bin/env python3
"""Train IWCM with oracle object-slot encoding — test representation bottleneck."""
import sys, torch, pickle, numpy as np, torch.nn.functional as F
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.seed import set_seed
from src.iwcm.model import IWCM
from src.metrics.evaluation import metric_cross_surface_law_generalization, metric_valid_invalid_classification
from src.encoder.oracle_slot_encoder import ORACLE_SLOT_DIM, MAX_OBJECTS

set_seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"
GS, H = 8, 25
D_STATE = MAX_OBJECTS * ORACLE_SLOT_DIM  # 120
print(f"Device: {device}, d_state={D_STATE}")

# Load oracle-slot data
with open("data/oracle_slot_trajs.pkl", "rb") as f:
    data = pickle.load(f)

def to_torch(entries):
    return [(torch.from_numpy(z0).reshape(-1).float(), torch.from_numpy(A).float(),
             torch.from_numpy(Z).reshape(H, -1).float())
            for z0, A, Z in entries]

valid = to_torch(data["valid"])
corr = to_torch(data["corruptions"])
print(f"Valid: {len(valid)}, Corruptions: {len(corr)}")

# Model (smaller since state is 120-dim, not 256)
model = IWCM(d_state=D_STATE, d_action=11, hidden_dim=192)
opt = torch.optim.Adam(model.parameters(), lr=5e-5)
model.to(device)
print(f"Model: {model.count_parameters_str()} params")

NUM_EPOCHS = 200
BATCH = 32
MARGIN = 1.5
REG = 0.0005

for epoch in range(NUM_EPOCHS):
    vi = np.random.choice(len(valid), min(BATCH//2, len(valid)), replace=False)
    ci = np.random.choice(len(corr), min(BATCH, len(corr)), replace=False)

    v_z0 = torch.stack([valid[i][0] for i in vi]).to(device)
    v_A = torch.stack([valid[i][1] for i in vi]).to(device)
    v_Z = torch.stack([valid[i][2] for i in vi]).to(device)
    c_z0 = torch.stack([corr[i][0] for i in ci]).to(device)
    c_A = torch.stack([corr[i][1] for i in ci]).to(device)
    c_Z = torch.stack([corr[i][2] for i in ci]).to(device)

    opt.zero_grad()
    ev = model.energy(v_z0, v_A, v_Z)
    ec = model.energy(c_z0, c_A, c_Z)
    loss = F.relu(ev + MARGIN).mean() + F.relu(MARGIN - ec).mean() + REG * (ev.pow(2).mean() + ec.pow(2).mean())
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()

    if (epoch+1) % 40 == 0:
        print(f"Epoch {epoch+1}: loss={loss.item():.4f} E_valid={ev.mean().item():.2f} E_invalid={ec.mean().item():.2f}")

# Evaluate cross-surface
model.eval()
print("\nCross-surface evaluation (oracle slots)...")

# Generate oracle-slot eval data
from src.env.grid_world import GridWorld
from src.env.scenarios import Scenario, PREDEFINED_SCENARIOS
from src.encoder.oracle_slot_encoder import encode_oracle_trajectory, build_door_key_map
from src.env.symbolic_state import SymbolicState, SymbolicTrajectory, symbolic_to_state_dict
from src.ac3.mutations.grammar import SymbolicMutationGrammar
from copy import deepcopy

rng = np.random.RandomState(99)
grammar = SymbolicMutationGrammar()

# Oracle-slot encoded eval valid trajectories
eval_valid = []
for sn in ['key_door_simple','multi_object']:
    sc = Scenario.from_preset(sn, GS)
    dkm = build_door_key_map(sc.to_env_config())
    for _ in range(50):
        env = GridWorld(grid_size=GS, objects_config=sc.to_env_config(), seed=int(rng.randint(0,2**31)))
        env.reset()
        states, actions = [], []
        state = env.get_state(); states.append(deepcopy(state))
        for t in range(H*3):
            va = env.get_valid_actions()
            if not va: break
            a = int(rng.choice(va))
            state, _, done, _ = env.step(a)
            states.append(deepcopy(state)); actions.append(a)
            if done: break
        if len(states) >= H+1 and len(actions) >= H:
            goal = states[0].get('goal', None)
            enc = encode_oracle_trajectory(states, actions, H, GS, goal, dkm)
            if enc:
                eval_valid.append(to_torch([enc])[0])
print(f"Eval valid: {len(eval_valid)}")

# Check accept scores
v_scores = []
for z0, A, Z in eval_valid[:50]:
    s = model.score_accept(z0.to(device).unsqueeze(0), A.to(device).unsqueeze(0), Z.to(device).unsqueeze(0)).item()
    v_scores.append(s)
print(f"Valid accept: mean={np.mean(v_scores):.4f} >0.5:{(np.array(v_scores)>0.5).mean():.3f}")

# Oracle-slot eval invalid trajectories (from cross-surface scenarios)
for law_name in ['conservation','identity','locality','temporal']:
    inv_scores = []
    for sn in ['multi_object','key_door_simple']:
        sc = Scenario.from_preset(sn, GS)
        dkm = build_door_key_map(sc.to_env_config())
        for _ in range(30):
            env = GridWorld(grid_size=GS, objects_config=sc.to_env_config(), seed=int(rng.randint(0,2**31)))
            env.reset()
            states, actions = [], []
            state = env.get_state(); states.append(deepcopy(state))
            for t in range(H*3):
                va = env.get_valid_actions()
                if not va: break
                a = int(rng.choice(va))
                state, _, done, _ = env.step(a)
                states.append(deepcopy(state)); actions.append(a)
                if done: break
            if len(states) < H+1: continue
            goal = states[0].get('goal', None)
            sym = []
            for s in states[:H+1]:
                op, ot = {}, {}
                for oid, obj in s.get('objects', {}).items():
                    op[oid] = tuple(obj.get('pos', (0,0)))
                    ot[oid] = obj.get('type', 'unknown')
                sym.append(SymbolicState(agent_pos=tuple(s.get('agent_pos',(0,0))), grid_size=GS, step=0,
                          object_positions=op, object_types=ot,
                          door_states=s.get('door_states',{}), inventory=s.get('inventory',[])))
            st = SymbolicTrajectory(states=sym, actions=list(actions[:H]), horizon=H)
            for mut in grammar.mutations:
                corr = mut.apply(st, rng)
                cd = [symbolic_to_state_dict(ss) for ss in corr.states]
                if len(cd) >= H+1:
                    enc = encode_oracle_trajectory(cd, list(corr.actions[:H]), H, GS, goal, dkm)
                    if enc:
                        z0c, Ac, Zc = to_torch([enc])[0]
                        s = model.score_accept(z0c.to(device).unsqueeze(0), Ac.to(device).unsqueeze(0), Zc.to(device).unsqueeze(0)).item()
                        inv_scores.append(s)
                        if len(inv_scores) >= 30: break
                if len(inv_scores) >= 30: break
            if len(inv_scores) >= 30: break
        if len(inv_scores) >= 30: break
    print(f'{law_name}: mean={np.mean(inv_scores):.4f} <0.5:{(np.array(inv_scores)<0.5).mean():.3f}')

model.save("outputs/checkpoints/iwcm_oracle_slots.pt")
print("Saved to outputs/checkpoints/iwcm_oracle_slots.pt")
