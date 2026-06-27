"""Oracle object-slot encoder — extracts rich per-object features from symbolic state.

Produces (N_objects, d_slot) per-state representations where each slot contains:
  - object type embedding (key/door/box/occluder/agent)
  - position (x, y normalized)
  - velocity / displacement from previous step
  - held_by_agent flag
  - visible/occluded flag
  - door/key relation (which key opens which door)
  - distance to goal
  - object persistence (same object across time)

This is the diagnostic tool for testing whether object-centric representation
is the bottleneck for cross-surface law generalization.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional


# Slot dimension: type(5) + pos(2) + velocity(2) + held(1) + occluded(1) + door_key(2) + goal_dist(1) + persistent(1) = 15
ORACLE_SLOT_DIM = 15
MAX_OBJECTS = 8


def encode_oracle_slots(
    state: dict,
    prev_state: Optional[dict] = None,
    grid_size: int = 8,
    goal: dict = None,
    door_key_map: Optional[Dict[str, str]] = None,
) -> np.ndarray:
    """Encode a GridWorld state into rich per-object oracle slots.

    Args:
        state: GridWorld.get_state() dict.
        prev_state: Previous state for velocity computation (None for initial state).
        grid_size: Grid dimensions.
        goal: Optional goal specification.
        door_key_map: Optional mapping door_id → key_id.

    Returns:
        numpy array of shape (MAX_OBJECTS, ORACLE_SLOT_DIM) — zero-padded.
    """
    slots = np.zeros((MAX_OBJECTS, ORACLE_SLOT_DIM), dtype=np.float32)
    obj_idx = 0

    ar, ac = state["agent_pos"]

    # Agent slot (always first)
    slots[obj_idx, 0] = 1.0  # type: agent (channel 0)
    slots[obj_idx, 5] = ar / grid_size
    slots[obj_idx, 6] = ac / grid_size

    if prev_state:
        pr, pc = prev_state.get("agent_pos", (ar, ac))
        slots[obj_idx, 7] = (ar - pr) / grid_size  # velocity y
        slots[obj_idx, 8] = (ac - pc) / grid_size  # velocity x

    obj_idx += 1

    # Object slots
    type_channels = {"key": 1, "door": 2, "box": 3, "occluder": 4}

    for oid, obj in state.get("objects", {}).items():
        if obj_idx >= MAX_OBJECTS:
            break

        r, c = obj["pos"]
        obj_type = obj.get("type", "unknown")
        props = obj.get("properties", {})

        # Type one-hot
        tc = type_channels.get(obj_type, 0)
        if 0 < tc < 5:
            slots[obj_idx, tc] = 1.0

        # Position
        slots[obj_idx, 5] = r / grid_size
        slots[obj_idx, 6] = c / grid_size

        # Velocity
        if prev_state and oid in prev_state.get("objects", {}):
            pr, pc = prev_state["objects"][oid].get("pos", (r, c))
            slots[obj_idx, 7] = (r - pr) / grid_size
            slots[obj_idx, 8] = (c - pc) / grid_size

        # Held by agent
        if oid in state.get("inventory", []):
            slots[obj_idx, 9] = 1.0

        # Occluded (inferred from object type — occluders block)
        if obj_type == "occluder":
            slots[obj_idx, 10] = 1.0

        # Door/key relation
        if obj_type == "door" and door_key_map and oid in door_key_map:
            key_id = door_key_map[oid]
            door_states = state.get("door_states", {})
            slots[obj_idx, 11] = 1.0 if door_states.get(oid, False) else 0.0
            slots[obj_idx, 12] = 1.0 if key_id in state.get("inventory", []) else 0.0

        # Distance to goal
        if goal and goal.get("type") == "position":
            gr, gc = goal["pos"]
            slots[obj_idx, 13] = (abs(r - gr) + abs(c - gc)) / (grid_size * 2)

        # Persistence marker (same object across time — always 1.0 for real objects)
        slots[obj_idx, 14] = 1.0

        obj_idx += 1

    return slots


def build_door_key_map(objects_config: dict) -> Dict[str, str]:
    """Extract door_id → key_id mapping from objects configuration."""
    mapping = {}
    if objects_config and "objects" in objects_config:
        for obj in objects_config["objects"]:
            if obj.get("type") == "door" and "key_id" in obj:
                mapping[obj["id"]] = obj["key_id"]
    return mapping


def encode_oracle_trajectory(
    states: List[dict],
    actions: List[int],
    horizon: int = 25,
    grid_size: int = 8,
    goal: dict = None,
    door_key_map: Optional[Dict[str, str]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Encode a full trajectory into oracle-slot (z0, A, Z) format.

    Returns:
        z0: (MAX_OBJECTS, ORACLE_SLOT_DIM) — initial state with zero velocity
        A: (horizon, 11) — action sequence (one-hot)
        Z: (horizon, MAX_OBJECTS, ORACLE_SLOT_DIM) — state slot sequence
    """
    if len(states) < horizon + 1:
        return None

    z0 = encode_oracle_slots(states[0], None, grid_size, goal, door_key_map)
    A = np.zeros((horizon, 11), dtype=np.float32)
    for i, a in enumerate(actions[:horizon]):
        if 0 <= a < 11:
            A[i, a] = 1.0

    Z_list = []
    for t in range(horizon):
        prev = states[t]
        curr = states[t + 1]
        Z_list.append(encode_oracle_slots(curr, prev, grid_size, goal, door_key_map))
    Z = np.stack(Z_list, axis=0)

    return z0, A, Z
