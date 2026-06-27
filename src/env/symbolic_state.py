"""Symbolic state oracle for AC3 constraint validation (Section 5).

Provides full access to grid world state as structured symbolic data.
Used by the constraint oracle in AC3 to determine whether trajectories
violate causal constraints. This is the ground-truth oracle that the
TAMG Validator Committee aims to approximate in a self-supervised way.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set, Any
from copy import deepcopy


@dataclass
class SymbolicState:
    """Full symbolic snapshot of a grid world state.

    Provides structured access to all state information needed for
    constraint violation detection.
    """
    # Agent
    agent_pos: Tuple[int, int]

    # Objects by ID
    object_positions: Dict[str, Tuple[int, int]] = field(default_factory=dict)
    object_types: Dict[str, str] = field(default_factory=dict)
    object_properties: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Inventory
    inventory: List[str] = field(default_factory=list)

    # Door states
    door_states: Dict[str, bool] = field(default_factory=dict)

    # Metadata
    grid_size: int = 8
    step: int = 0

    @classmethod
    def from_env_state(cls, state: dict, step: int = 0) -> "SymbolicState":
        """Extract symbolic state from GridWorld state dict.

        Args:
            state: GridWorld.get_state() output.
            step: Current step index in the trajectory.

        Returns:
            SymbolicState with structured access to all information.
        """
        return cls(
            agent_pos=tuple(state["agent_pos"]),
            object_positions={
                oid: tuple(obj["pos"])
                for oid, obj in state.get("objects", {}).items()
            },
            object_types={
                oid: obj["type"]
                for oid, obj in state.get("objects", {}).items()
            },
            object_properties={
                oid: deepcopy(obj.get("properties", {}))
                for oid, obj in state.get("objects", {}).items()
            },
            inventory=list(state.get("inventory", [])),
            door_states=dict(state.get("door_states", {})),
            grid_size=state.get("grid_size", 8),
            step=step,
        )

    def get_objects_of_type(self, obj_type: str) -> Dict[str, Tuple[int, int]]:
        """Get all objects of a specific type with their positions."""
        return {oid: self.object_positions[oid]
                for oid, t in self.object_types.items() if t == obj_type}

    def get_keys(self) -> Dict[str, Tuple[int, int]]:
        return self.get_objects_of_type("key")

    def get_doors(self) -> Dict[str, Tuple[int, int]]:
        return self.get_objects_of_type("door")

    def get_boxes(self) -> Dict[str, Tuple[int, int]]:
        return self.get_objects_of_type("box")

    def get_occluders(self) -> Dict[str, Tuple[int, int]]:
        return self.get_objects_of_type("occluder")

    def has_object(self, obj_id: str) -> bool:
        return obj_id in self.object_positions

    def object_type(self, obj_id: str) -> Optional[str]:
        return self.object_types.get(obj_id)


@dataclass
class SymbolicTrajectory:
    """A full trajectory of symbolic states.

    Supports all constraint violation checks needed for AC3 oracle.
    """
    states: List[SymbolicState] = field(default_factory=list)
    actions: List[int] = field(default_factory=list)
    horizon: int = 0

    def __len__(self) -> int:
        return len(self.states)

    @classmethod
    def from_env_trajectory(
        cls, states: List[dict], actions: List[int]
    ) -> "SymbolicTrajectory":
        """Construct from a sequence of GridWorld states and actions.

        Args:
            states: List of GridWorld state dicts.
            actions: List of integer actions.

        Returns:
            SymbolicTrajectory for constraint checking.
        """
        symbolic = [
            SymbolicState.from_env_state(s, i)
            for i, s in enumerate(states)
        ]
        return cls(states=symbolic, actions=list(actions), horizon=len(states))


def extract_symbolic_state(state: dict, step: int = 0) -> SymbolicState:
    """Convenience function: extract SymbolicState from env state dict."""
    return SymbolicState.from_env_state(state, step)


def compare_symbolic_states(
    s1: SymbolicState, s2: SymbolicState
) -> Dict[str, Any]:
    """Detailed comparison of two symbolic states.

    Returns a dict identifying what changed between states:
    - objects_added, objects_removed, objects_moved
    - identity_changes (type mismatches for same ID)
    - inventory_changes
    - door_state_changes

    Args:
        s1: First symbolic state (reference).
        s2: Second symbolic state (comparison).

    Returns:
        Dict with change categories and affected object IDs.
    """
    changes: Dict[str, Any] = {
        "objects_added": [],
        "objects_removed": [],
        "objects_moved": [],
        "identity_changes": [],
        "inventory_changes": {"added": [], "removed": []},
        "door_state_changes": [],
    }

    ids1 = set(s1.object_positions.keys())
    ids2 = set(s2.object_positions.keys())

    # Detect added/removed objects
    changes["objects_added"] = list(ids2 - ids1)
    changes["objects_removed"] = list(ids1 - ids2)

    # Detect movement and identity changes
    for oid in ids1 & ids2:
        if s1.object_positions[oid] != s2.object_positions[oid]:
            changes["objects_moved"].append(oid)
        if s1.object_types.get(oid) != s2.object_types.get(oid):
            changes["identity_changes"].append(oid)

    # Inventory changes
    inv1 = set(s1.inventory)
    inv2 = set(s2.inventory)
    changes["inventory_changes"]["added"] = list(inv2 - inv1)
    changes["inventory_changes"]["removed"] = list(inv1 - inv2)

    # Door state changes
    for door_id in set(s1.door_states.keys()) | set(s2.door_states.keys()):
        if s1.door_states.get(door_id) != s2.door_states.get(door_id):
            changes["door_state_changes"].append(door_id)

    return changes


def symbolic_to_state_dict(ss: SymbolicState, goal: dict = None) -> dict:
    gs = ss.grid_size

    def _clamp(pos):
        return (max(0, min(gs - 1, int(pos[0]))), max(0, min(gs - 1, int(pos[1]))))

    objects = {}
    for oid in ss.object_positions:
        clamped_pos = _clamp(ss.object_positions[oid])
        objects[oid] = {
            "type": ss.object_types.get(oid, "unknown"),
            "pos": list(clamped_pos),
            "properties": ss.object_properties.get(oid, {}),
        }
    return {
        "agent_pos": _clamp(ss.agent_pos),
        "objects": objects,
        "door_states": dict(ss.door_states),
        "inventory": list(ss.inventory),
        "grid_size": gs,
        "goal": goal or {"type": "position", "pos": (0, 0)},
    }


def encode_symbolic_trajectory(
    sym_traj, grid_size: int = 8, horizon: int = 25,
) -> tuple:
    """Encode a SymbolicTrajectory into (z0, A, Z) numpy arrays.

    Returns tensors suitable for IWCM model input.
    """
    from .data import encode_state, encode_action
    import numpy as np

    states_dict = [symbolic_to_state_dict(s) for s in sym_traj.states]
    if len(states_dict) < horizon + 1:
        horizon = len(states_dict) - 1
    if horizon < 1:
        return None

    z0 = encode_state(states_dict[0], grid_size)
    A = np.zeros((horizon, 11), dtype=np.float32)
    for i, a in enumerate(sym_traj.actions[:horizon]):
        if 0 <= a < 11:
            A[i, a] = 1.0
    Z = np.array([encode_state(s, grid_size) for s in states_dict[1:horizon + 1]])
    return z0, A, Z
