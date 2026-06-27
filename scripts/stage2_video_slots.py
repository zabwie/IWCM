#!/usr/bin/env python3
"""Stage 2: VideoSlotIWCM — learned slots vs oracle slots on rendered video.

Generates GridWorld trajectories, renders video frames, encodes with both
oracle slots and learned (CNN+SlotAttention) slots, trains two IWCM models,
and compares accuracy on the 7 violation metrics.

Key test: do learned slots retain oracle-slot accuracy on law violation detection?
"""
import sys, time, pickle, numpy as np, torch
from pathlib import Path
from collections import defaultdict
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.seed import set_seed
from src.encoder.oracle_slot_encoder import (
    encode_oracle_slots, encode_oracle_trajectory,
    build_door_key_map, ORACLE_SLOT_DIM, MAX_OBJECTS,
)
from src.encoder.video_encoder import VideoEncoder
from src.iwcm.fused_energy import FusedIWCMEnergy
from src.iwcm.micro_energy import MicroIWCM
from src.env.grid_world import GridWorld
from src.env.renderer import GridWorldRenderer
from src.env.scenarios import Scenario

# ─── Config ──────────────────────────────────────────────────────────────────
HORIZON = 25
GRID_SIZE = 8
FRAME_SIZE = 64
NUM_TRAIN_VALID = 60
NUM_TRAIN_CORRUPT = 120
NUM_TEST_VALID = 40
NUM_TEST_CORRUPT = 80
EPOCHS = 150
BATCH_VALID = 8
BATCH_CORRUPT = 16
LR = 3e-4
SLOT_DIM = 64  # learned slot dimension (must match VideoEncoder)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

VIOLATION_TYPES = ["delete", "swap", "teleport", "duplicate", "transform"]

# ─── Data Generation ─────────────────────────────────────────────────────────

def generate_trajectory(scenario, rng, horizon, corrupt_type=None):
    """Generate one trajectory with optional corruption. Returns (states, actions, meta)."""
    gw = GridWorld(grid_size=GRID_SIZE, objects_config=scenario.to_env_config(),
                   seed=int(rng.randint(0, 2**31)))
    gw.reset()
    states, actions = [], []
    state = gw.get_state()
    states.append(dict(state))
    for _ in range(horizon + 5):  # extra steps for safety
        acts = gw.get_valid_actions()
        if not acts: break
        a = int(rng.choice(acts))
        s, _, done, _ = gw.step(a)
        states.append(dict(s))
        actions.append(a)
        if done: break

    if len(states) < horizon + 1:
        return None

    meta = {"law_type": "valid", "violation_type": "none"}
    if corrupt_type:
        # Apply corruption: modify states at a random timestep range
        t_corrupt = rng.randint(horizon // 4, 3 * horizon // 4)
        obj_ids = list(states[0].get("objects", {}).keys())
        if not obj_ids:
            return None  # can't corrupt without objects
        target_obj = obj_ids[rng.randint(0, len(obj_ids))]

        if corrupt_type == "delete":
            for t in range(t_corrupt, horizon + 1):
                states[t].setdefault("objects", {}).pop(target_obj, None)

        elif corrupt_type == "swap" and len(obj_ids) >= 2:
            other = obj_ids[rng.randint(0, len(obj_ids))]
            while other == target_obj:
                other = obj_ids[rng.randint(0, len(obj_ids))]
            for t in range(t_corrupt, horizon + 1):
                objs = states[t].setdefault("objects", {})
                if target_obj in objs and other in objs:
                    objs[target_obj], objs[other] = dict(objs[other]), dict(objs[target_obj])

        elif corrupt_type == "teleport" and target_obj in states[t_corrupt].get("objects", {}):
            obj = states[t_corrupt]["objects"][target_obj]
            new_r = rng.randint(0, GRID_SIZE - 1)
            new_c = rng.randint(0, GRID_SIZE - 1)
            for t in range(t_corrupt, horizon + 1):
                if target_obj in states[t].get("objects", {}):
                    states[t]["objects"][target_obj]["pos"] = (new_r, new_c)

        elif corrupt_type == "duplicate":
            new_id = target_obj + "_dup"
            for t in range(t_corrupt, horizon + 1):
                if target_obj in states[t].get("objects", {}):
                    states[t]["objects"][new_id] = dict(states[t]["objects"][target_obj])

        elif corrupt_type == "transform":
            for t in range(t_corrupt, horizon + 1):
                if target_obj in states[t].get("objects", {}):
                    states[t]["objects"][target_obj]["type"] = "box"

        meta = {"law_type": "conservation" if corrupt_type in ["delete", "duplicate", "transform"]
                else "identity", "violation_type": corrupt_type}

    return states[:horizon + 1], np.array(actions[:horizon], dtype=np.int64), meta


def render_trajectory(states, renderer):
    """Render trajectory states to video frames. Returns (H+1, C, H_img, W_img) uint8 tensor."""
    frames = []
    for state in states:
        img = renderer.render_frame(state)
        frames.append(torch.from_numpy(img).float().permute(2, 0, 1) / 255.0)  # (H,W,C)→(C,H,W)
    return torch.stack(frames)  # (H+1, C, H, W)


def encode_oracle(states, actions, scenario):
    """Encode trajectory with oracle slot encoder."""
    dkm = build_door_key_map(scenario.to_env_config())
    goal = scenario.to_env_config().get("goal", None)
    return encode_oracle_trajectory(states, actions, HORIZON, GRID_SIZE, goal, dkm)


def generate_dataset(num_valid, num_corrupt, rng):
    """Generate balanced dataset of valid + corrupted trajectories."""
    scenarios = [Scenario.from_preset(name, GRID_SIZE)
                 for name in ["key_door_simple", "multi_object"]]
    renderer = GridWorldRenderer(grid_size=GRID_SIZE, cell_px=FRAME_SIZE // GRID_SIZE)

    data = {"valid": [], "corrupt": []}  # each: list of (frames, oracle_slots, actions, meta)

    # Valid trajectories
    for _ in range(num_valid):
        scenario = scenarios[rng.randint(0, len(scenarios))]
        result = generate_trajectory(scenario, rng, HORIZON)
        if result is None: continue
        states, actions, meta = result
        oracle = encode_oracle(states, actions, scenario)
        if oracle is None: continue
        frames = render_trajectory(states, renderer)
        data["valid"].append((frames, oracle, actions, meta))

    # Corrupted trajectories
    for _ in range(num_corrupt):
        scenario = scenarios[rng.randint(0, len(scenarios))]
        vt = VIOLATION_TYPES[rng.randint(0, len(VIOLATION_TYPES))]
        result = generate_trajectory(scenario, rng, HORIZON, corrupt_type=vt)
        if result is None: continue
        states, actions, meta = result
        oracle = encode_oracle(states, actions, scenario)
        if oracle is None: continue
        frames = render_trajectory(states, renderer)
        data["corrupt"].append((frames, oracle, actions, meta))

    return data


# ─── Training ────────────────────────────────────────────────────────────────

def train_oracle_iwcm(train_data, test_data):
    """Train IWCM on oracle slots. Returns (model, eval_results)."""
    model = FusedIWCMEnergy(d_slot=ORACLE_SLOT_DIM, d_action=11, hidden=128, num_slots=MAX_OBJECTS).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    tv = [(torch.from_numpy(z0).float(), torch.from_numpy(A).float(), torch.from_numpy(Z).float())
          for _, (z0, A, Z), _, _ in train_data["valid"]]
    tc = [(torch.from_numpy(z0).float(), torch.from_numpy(A).float(), torch.from_numpy(Z).float(), m)
          for _, (z0, A, Z), _, m in train_data["corrupt"]]

    for ep in range(EPOCHS):
        vi = np.random.choice(len(tv), min(BATCH_VALID, len(tv)), replace=False)
        ci = np.random.choice(len(tc), min(BATCH_CORRUPT, len(tc)), replace=False)
        vz0 = torch.stack([tv[i][0] for i in vi]).to(DEVICE)
        cz0 = torch.stack([tc[i][0] for i in ci]).to(DEVICE)
        vA = torch.stack([tv[i][1] for i in vi]).to(DEVICE)
        cA = torch.stack([tc[i][1] for i in ci]).to(DEVICE)
        vZ = torch.stack([tv[i][2] for i in vi]).to(DEVICE)
        cZ = torch.stack([tc[i][2] for i in ci]).to(DEVICE)
        opt.zero_grad()
        ev = model(vz0, vA, vZ); ec = model(cz0, cA, cZ)
        loss = (torch.nn.functional.relu(ev + 1.0).mean() +
                torch.nn.functional.relu(1.0 - ec).mean() +
                0.001 * (ev.pow(2).mean() + ec.pow(2).mean()))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    model.eval()
    return model, evaluate_model(model, test_data, use_oracle=True)


def train_video_iwcm(train_data, test_data):
    """Train VideoEncoder + IWCM jointly on rendered frames."""
    encoder = VideoEncoder(frame_size=FRAME_SIZE, in_channels=3, num_slots=MAX_OBJECTS,
                           slot_dim=SLOT_DIM).to(DEVICE)
    iwcm = FusedIWCMEnergy(d_slot=SLOT_DIM, d_action=11, hidden=128, num_slots=MAX_OBJECTS).to(DEVICE)

    opt = torch.optim.Adam(list(encoder.parameters()) + list(iwcm.parameters()), lr=LR)

    for ep in range(EPOCHS):
        vi = np.random.choice(len(train_data["valid"]), min(BATCH_VALID, len(train_data["valid"])), replace=False)
        ci = np.random.choice(len(train_data["corrupt"]), min(BATCH_CORRUPT, len(train_data["corrupt"])), replace=False)

        # Valid batch: frames (B, H+1, C, H_img, W_img) → encode to (B, H, N, d)
        v_frames = torch.stack([train_data["valid"][i][0] for i in vi]).to(DEVICE)
        v_A = torch.stack([torch.from_numpy(train_data["valid"][i][2]).float() for i in vi]).to(DEVICE)
        v_frames_in = v_frames[:, :HORIZON]  # (B, H, C, H_img, W_img) — first H frames as input
        v_slots = encoder(v_frames_in)  # (B, H, N, d)
        v_z0 = v_slots[:, 0]
        v_A_in = v_A[:, :HORIZON]  # H actions

        # Corrupt batch
        c_frames = torch.stack([train_data["corrupt"][i][0] for i in ci]).to(DEVICE)
        c_A = torch.stack([torch.from_numpy(train_data["corrupt"][i][2]).float() for i in ci]).to(DEVICE)
        c_frames_in = c_frames[:, :HORIZON]
        c_slots = encoder(c_frames_in)
        c_z0 = c_slots[:, 0]
        c_A_in = c_A[:, :HORIZON]

        opt.zero_grad()
        ev = iwcm(v_z0, v_A_in, v_slots); ec = iwcm(c_z0, c_A_in, c_slots)
        recon_loss = torch.tensor(0.0, device=DEVICE)

        loss = (torch.nn.functional.relu(ev + 1.0).mean() +
                torch.nn.functional.relu(1.0 - ec).mean() +
                0.001 * (ev.pow(2).mean() + ec.pow(2).mean()) + recon_loss)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(iwcm.parameters()), 1.0)
        opt.step()

    encoder.eval()
    iwcm.eval()
    return (encoder, iwcm), evaluate_model_video(encoder, iwcm, test_data)


# ─── Evaluation ──────────────────────────────────────────────────────────────

def evaluate_model(model, test_data, use_oracle=True):
    """Evaluate on oracle-slot data. Returns dict of per-violation metrics."""
    per_vtype = defaultdict(list)
    va, ir = [], []

    for frames, oracle, actions, meta in test_data["valid"]:
        z0, A, Z = oracle
        s = model.score_acceptance(
            torch.from_numpy(z0).float().unsqueeze(0).to(DEVICE),
            torch.from_numpy(A).float().unsqueeze(0).to(DEVICE),
            torch.from_numpy(Z).float().unsqueeze(0).to(DEVICE),
        ).item()
        va.append(s > 0.5)

    for frames, oracle, actions, meta in test_data["corrupt"]:
        z0, A, Z = oracle
        s = model.score_acceptance(
            torch.from_numpy(z0).float().unsqueeze(0).to(DEVICE),
            torch.from_numpy(A).float().unsqueeze(0).to(DEVICE),
            torch.from_numpy(Z).float().unsqueeze(0).to(DEVICE),
        ).item()
        ir.append(s < 0.5)
        per_vtype[meta.get("violation_type", "?")].append(s < 0.5)

    result = {"valid_acc": np.mean(va) if va else 0, "invalid_rej": np.mean(ir) if ir else 0}
    for vt in VIOLATION_TYPES:
        result[vt] = np.mean(per_vtype.get(vt, [0])) if per_vtype.get(vt) else 0
    return result


def evaluate_model_video(encoder, iwcm, test_data):
    """Evaluate video encoder + IWCM on rendered frames."""
    per_vtype = defaultdict(list)
    va, ir = [], []

    for frames, oracle, actions, meta in test_data["valid"]:
        f = frames.unsqueeze(0).to(DEVICE)  # (1, H+1, C, H_img, W_img)
        slots = encoder(f[:, :HORIZON])  # (1, H, N, d)
        z0 = slots[:, 0]
        A = torch.from_numpy(actions[:HORIZON]).float().unsqueeze(0).to(DEVICE)
        s = iwcm.score_acceptance(z0, A, slots).item()
        va.append(s > 0.5)

    for frames, oracle, actions, meta in test_data["corrupt"]:
        f = frames.unsqueeze(0).to(DEVICE)
        slots = encoder(f[:, :HORIZON])
        z0 = slots[:, 0]
        A = torch.from_numpy(actions[:HORIZON]).float().unsqueeze(0).to(DEVICE)
        s = iwcm.score_acceptance(z0, A, slots).item()
        ir.append(s < 0.5)
        per_vtype[meta.get("violation_type", "?")].append(s < 0.5)

    result = {"valid_acc": np.mean(va) if va else 0, "invalid_rej": np.mean(ir) if ir else 0}
    for vt in VIOLATION_TYPES:
        result[vt] = np.mean(per_vtype.get(vt, [0])) if per_vtype.get(vt) else 0
    return result


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    set_seed(42)
    rng = np.random.RandomState(42)

    print("=" * 72)
    print("STAGE 2: Oracle-Slot IWCM vs Video-Slot IWCM")
    print(f"H={HORIZON}, grid={GRID_SIZE}x{GRID_SIZE}, frames={FRAME_SIZE}x{FRAME_SIZE}")
    print(f"Train: {NUM_TRAIN_VALID} valid + {NUM_TRAIN_CORRUPT} corrupt")
    print(f"Test:  {NUM_TEST_VALID} valid + {NUM_TEST_CORRUPT} corrupt")
    print("=" * 72)

    # Generate data
    t0 = time.time()
    print("\nGenerating trajectories and rendering frames...")
    train_data = generate_dataset(NUM_TRAIN_VALID, NUM_TRAIN_CORRUPT, rng)
    test_data = generate_dataset(NUM_TEST_VALID, NUM_TEST_CORRUPT, rng)
    print(f"  Train: {len(train_data['valid'])} valid, {len(train_data['corrupt'])} corrupt")
    print(f"  Test:  {len(test_data['valid'])} valid, {len(test_data['corrupt'])} corrupt")
    print(f"  Time:  {time.time() - t0:.1f}s")

    # Train oracle-slot IWCM
    print("\nTraining Oracle-Slot IWCM...")
    t0 = time.time()
    model_o, results_o = train_oracle_iwcm(train_data, test_data)
    print(f"  Time: {time.time() - t0:.1f}s")
    print(f"  Results: valid={results_o['valid_acc']:.3f} rej={results_o['invalid_rej']:.3f}")
    for vt in VIOLATION_TYPES:
        print(f"    {vt:<12}: {results_o[vt]:.3f}")

    # Train video-slot IWCM
    print("\nTraining Video-Slot IWCM...")
    t0 = time.time()
    (encoder, iwcm_v), results_v = train_video_iwcm(train_data, test_data)
    print(f"  Time: {time.time() - t0:.1f}s")
    print(f"  Results: valid={results_v['valid_acc']:.3f} rej={results_v['invalid_rej']:.3f}")
    for vt in VIOLATION_TYPES:
        print(f"    {vt:<12}: {results_v[vt]:.3f}")

    # Comparison
    print("\n" + "=" * 72)
    print("COMPARISON: Oracle Slots vs Learned Slots")
    print("=" * 72)
    print(f"{'Metric':<16} {'Oracle':>10} {'Learned':>10} {'Ratio':>10} {'Status':>10}")
    print("-" * 60)
    for k in ["valid_acc", "invalid_rej"] + VIOLATION_TYPES:
        o = results_o.get(k, 0)
        v = results_v.get(k, 0)
        ratio = v / max(o, 0.001)
        status = "OK" if ratio > 0.7 else ("LOW" if ratio > 0.4 else "LOST")
        print(f"{k:<16} {o:>10.3f} {v:>10.3f} {ratio:>10.2f} {status:>10}")

    # Save
    torch.save({"oracle_iwcm": model_o.state_dict(), "encoder": encoder.state_dict(),
                "video_iwcm": iwcm_v.state_dict()}, "outputs/checkpoints/stage2_models.pt")
    print("\nModels saved to outputs/checkpoints/stage2_models.pt")


if __name__ == "__main__":
    main()
