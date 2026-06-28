#!/usr/bin/env python3
"""Train IWCM with slot-based object-centric encoding for cross-surface generalization."""
import sys, torch, pickle, numpy as np, torch.nn.functional as F
from pathlib import Path
from copy import deepcopy
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.seed import set_seed
from src.iwcm.model import IWCM
from src.metrics.evaluation import metric_cross_surface_law_generalization, metric_valid_invalid_classification
from src.encoder.slot_encoder import encode_slots, encode_trajectory_slots, encode_action_slot
from src.env.scenarios import Scenario, PREDEFINED_SCENARIOS
from src.env.grid_world import GridWorld
from src.env.symbolic_state import SymbolicState, SymbolicTrajectory, symbolic_to_state_dict
from src.ac3.oracle import SymbolicOracle
from src.ac3.mutations.grammar import SymbolicMutationGrammar

set_seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"
SLOT_DIM, MAX_OBJ, H, GS = 16, 8, 25, 8
D_STATE = MAX_OBJ * SLOT_DIM  # 128
print(f"Device: {device}, d_state={D_STATE}")

# Load valid slot trajectories
with open("data/slot_trajs.pkl", "rb") as f:
    valid_trajs = pickle.load(f)
valid = [(torch.from_numpy(z0).reshape(-1), torch.from_numpy(A),
          torch.from_numpy(Z).reshape(H, -1)) for z0, A, Z in valid_trajs]
print(f"Valid slot trajectories: {len(valid)}")

# Generate slot-encoded corruption pairs (all surface forms)
grammar = SymbolicMutationGrammar()
oracle = SymbolicOracle()
rng = np.random.RandomState(42)

all_corr = []
for scenario_name in list(PREDEFINED_SCENARIOS.keys())[:5]:
    scenario = Scenario.from_preset(scenario_name, GS)
    for _ in range(80):
        env = GridWorld(grid_size=GS, objects_config=scenario.to_env_config(), seed=int(rng.randint(0, 2**31)))
        env.reset()
        states, actions = [], []
        state = env.get_state()
        states.append(deepcopy(state))
        for t in range(H * 3):
            valid_acts = env.get_valid_actions()
            if not valid_acts: break
            a = int(rng.choice(valid_acts))
            state, _, done, _ = env.step(a)
            states.append(deepcopy(state)); actions.append(a)
            if done: break
        if len(states) < H + 1: continue

        goal = states[0].get('goal', None)
        sym_states = []
        for s in states[:H+1]:
            obj_pos, obj_types = {}, {}
            for oid, obj in s.get("objects", {}).items():
                obj_pos[oid] = tuple(obj.get("pos", (0,0)))
                obj_types[oid] = obj.get("type", "unknown")
            sym_states.append(SymbolicState(
                agent_pos=tuple(s.get("agent_pos", (0,0))), grid_size=GS, step=0,
                object_positions=obj_pos, object_types=obj_types,
                door_states=s.get("door_states", {}), inventory=s.get("inventory", []),
            ))
        st = SymbolicTrajectory(states=sym_states, actions=list(actions[:H]), horizon=H)

        # Apply each mutation type to create corruptions
        for mutation in grammar.mutations:
            corrupted = mutation.apply(st, rng)
            corrupted_dicts = [symbolic_to_state_dict(ss) for ss in corrupted.states]
            if len(corrupted_dicts) >= H + 1:
                enc = encode_trajectory_slots(corrupted_dicts, list(corrupted.actions[:H]), H, GS, goal)
                if enc:
                    z0_c, A_c, Z_c = enc
                    all_corr.append((z0_c, A_c, Z_c))

print(f"Corruption pairs: {len(all_corr)}")

corr = [(torch.from_numpy(z0).reshape(-1), torch.from_numpy(A),
         torch.from_numpy(Z).reshape(H, -1)) for z0, A, Z in all_corr]

# Model
model = IWCM(d_state=D_STATE, d_action=11, hidden_dim=256)
opt = torch.optim.Adam(model.parameters(), lr=5e-5)
model.to(device)
print(f"Model: {model.count_parameters_str()} params")

NUM_EPOCHS = 200
BATCH = 32
MARGIN = 1.5
REG = 0.0005
best_ho = 0.0

for epoch in range(NUM_EPOCHS):
    # Sample batch
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

print(f"\nBest held-out: {best_ho:.3f}")
model.save("outputs/checkpoints/iwcm_slots.pt")
