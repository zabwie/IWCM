"""Symbolic slot encoder — extracts per-object feature vectors from grid state.

Replaces the flat 256-dim grid encoding with (N_objects, d_slot) per-state
representations. Each object gets features for position, type, state, and
goal relation — enabling the constraint heads to learn laws that abstract
across object types (e.g., "duplication = violation" regardless of whether
it's a key or a box).
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional


SLOT_DIM = 16
MAX_OBJECTS = 8


def encode_slots(state: dict, grid_size: int = 8, goal: dict = None) -> np.ndarray:
    """Encode a GridWorld state into per-object slot features.

    Args:
        state: GridWorld.get_state() dict.
        grid_size: Grid dimensions.
        goal: Optional goal specification.

    Returns:
        numpy array of shape (MAX_OBJECTS, SLOT_DIM) — zero-padded.
    """
    slots = np.zeros((MAX_OBJECTS, SLOT_DIM), dtype=np.float32)
    obj_idx = 0

    # Agent slot (always present)
    ar, ac = state["agent_pos"]
    slots[obj_idx, 0] = ar / grid_size
    slots[obj_idx, 1] = ac / grid_size
    slots[obj_idx, 2] = 1.0  # is_agent flag
    obj_idx += 1

    # Object slots
    for oid, obj in state.get("objects", {}).items():
        if obj_idx >= MAX_OBJECTS:
            break
        r, c = obj["pos"]
        obj_type = obj.get("type", "unknown")

        slots[obj_idx, 0] = r / grid_size
        slots[obj_idx, 1] = c / grid_size

        # Type one-hot in channels 3-6
        type_map = {"key": 3, "door": 4, "box": 5, "occluder": 6}
        if obj_type in type_map:
            slots[obj_idx, type_map[obj_type]] = 1.0

        # Door state
        if obj_type == "door":
            door_states = state.get("door_states", {})
            slots[obj_idx, 7] = 1.0 if door_states.get(oid, False) else 0.0

        # In inventory
        if oid in state.get("inventory", []):
            slots[obj_idx, 8] = 1.0

        # Distance to agent
        slots[obj_idx, 9] = (abs(r - ar) + abs(c - ac)) / (grid_size * 2)

        # Distance to goal
        if goal and goal.get("type") == "position":
            gr, gc = goal["pos"]
            slots[obj_idx, 10] = (abs(r - gr) + abs(c - gc)) / (grid_size * 2)

        obj_idx += 1

    # Fill remaining slots with existing objects (cycle if fewer than MAX)
    # This ensures all slots have some content for fixed-size processing
    if obj_idx < MAX_OBJECTS and obj_idx > 1:
        for i in range(obj_idx, MAX_OBJECTS):
            slots[i] = slots[1 + (i - 1) % (obj_idx - 1)]
            slots[i, :3] = 0.0  # clear position/agent flag for copies

    return slots


def encode_action_slot(action: int, num_actions: int = 11) -> np.ndarray:
    """One-hot encode action, broadcastable to slot dimensions."""
    return np.eye(num_actions, dtype=np.float32)[action]


def encode_trajectory_slots(
    states: List[dict], actions: List[int], horizon: int = 25,
    grid_size: int = 8, goal: dict = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Encode a full trajectory into slot-based (z0, A, Z) format.

    Returns:
        z0: (MAX_OBJECTS, SLOT_DIM) — initial state slots
        A: (horizon, num_actions) — action sequence
        Z: (horizon, MAX_OBJECTS, SLOT_DIM) — state slot sequence
    """
    if len(states) < horizon + 1:
        return None

    z0 = encode_slots(states[0], grid_size, goal)
    A = np.array([encode_action_slot(a) for a in actions[:horizon]])
    Z = np.array([encode_slots(s, grid_size, goal) for s in states[1:horizon + 1]])
    return z0, A, Z
