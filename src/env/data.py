"""PyTorch Dataset for grid world trajectories.

Provides TrajectoryDataset and CounterfactualDataset classes for
training the IWCM world model and AC3 corruptor.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import pickle

from .scenarios import (
    Scenario, generate_trajectories, generate_counterfactual_pairs,
    PREDEFINED_SCENARIOS,
)


# ═══════════════════════════════════════════════════════════
# State Encoding
# ═══════════════════════════════════════════════════════════

def encode_state(state: dict, grid_size: int = 8) -> np.ndarray:
    """Encode a GridWorld state dict into a fixed-size numpy array.

    Layout per cell (4 channels):
      [0]: agent presence (1.0 or 0.0)
      [1]: object type encoding (key=1, door=2, box=3, occluder=4)
      [2]: door open state (1.0 if open door at cell, 0 otherwise)
      [3]: goal indicator (1.0 if goal cell)

    Plus inventory channels (one per possible item slot).

    Args:
        state: GridWorld.get_state() dict.
        grid_size: Width/height of the grid.

    Returns:
        numpy array of shape (grid_size, grid_size, 4).
    """
    encoding = np.zeros((grid_size, grid_size, 4), dtype=np.float32)

    # Agent
    ar, ac = state["agent_pos"]
    encoding[ar, ac, 0] = 1.0

    # Objects
    object_type_map = {"key": 1.0, "door": 2.0, "box": 3.0, "occluder": 4.0}
    for obj_id, obj in state.get("objects", {}).items():
        r, c = obj["pos"]
        encoding[r, c, 1] = object_type_map.get(obj["type"], 0.0)

    # Door states
    door_states = state.get("door_states", {})
    for door_id, is_open in door_states.items():
        if door_id in state.get("objects", {}):
            r, c = state["objects"][door_id]["pos"]
            encoding[r, c, 2] = 1.0 if is_open else 0.0

    # Goal
    goal = state.get("goal", {})
    if goal.get("type") == "position":
        gr, gc = goal["pos"]
        encoding[gr, gc, 3] = 1.0

    return encoding


def encode_action(action: int, num_actions: int = 11) -> np.ndarray:
    """One-hot encode an action.

    Args:
        action: Integer action index.
        num_actions: Size of action space.

    Returns:
        numpy array of shape (num_actions,).
    """
    onehot = np.zeros(num_actions, dtype=np.float32)
    onehot[action] = 1.0
    return onehot


# ═══════════════════════════════════════════════════════════
# TrajectoryDataset
# ═══════════════════════════════════════════════════════════

class TrajectoryDataset(Dataset):
    """Dataset of (z_0, A, Z) tuples for training the IWCM world model.

    Each item:
      z_0: encoded initial state, shape (grid_size, grid_size, 4)
      A:   action sequence, shape (horizon, num_actions)
      Z:   state sequence, shape (horizon, grid_size, grid_size, 4)
    """

    def __init__(
        self,
        trajectories: List[Tuple[List[dict], List[int]]],
        horizon: int,
        grid_size: int = 8,
        num_actions: int = 11,
    ):
        self.grid_size = grid_size
        self.horizon = horizon
        self.num_actions = num_actions
        self._data: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []

        for states, actions in trajectories:
            if len(states) < horizon + 1:
                continue

            z0 = encode_state(states[0], grid_size)
            A = np.array([encode_action(a, num_actions) for a in actions[:horizon]])
            Z = np.array([encode_state(s, grid_size) for s in states[1:horizon + 1]])

            self._data.append((z0, A, Z))

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z0, A, Z = self._data[idx]
        return (
            torch.from_numpy(z0),
            torch.from_numpy(A),
            torch.from_numpy(Z),
        )


class CounterfactualDataset(Dataset):
    """Dataset of counterfactual trajectory pairs for C_counterfactual training.

    Each item: (z0, A, Z, A', Z') — two futures from the same initial state.
    """

    def __init__(
        self,
        pairs: List[Tuple[List[dict], List[int], List[dict], List[int]]],
        horizon: int,
        grid_size: int = 8,
        num_actions: int = 11,
    ):
        self.grid_size = grid_size
        self.horizon = horizon
        self.num_actions = num_actions
        self._data: List[Tuple[np.ndarray, np.ndarray, np.ndarray,
                               np.ndarray, np.ndarray]] = []

        for sA, aA, sB, aB in pairs:
            h = min(horizon, len(sA) - 1, len(sB) - 1)
            if h < 2:
                continue

            z0 = encode_state(sA[0], grid_size)
            A = np.array([encode_action(a, num_actions) for a in aA[:h]])
            Z = np.array([encode_state(s, grid_size) for s in sA[1:h + 1]])
            Ap = np.array([encode_action(a, num_actions) for a in aB[:h]])
            Zp = np.array([encode_state(s, grid_size) for s in sB[1:h + 1]])

            self._data.append((z0, A, Z, Ap, Zp))

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor,
        torch.Tensor, torch.Tensor,
    ]:
        z0, A, Z, Ap, Zp = self._data[idx]
        return (
            torch.from_numpy(z0),
            torch.from_numpy(A),
            torch.from_numpy(Z),
            torch.from_numpy(Ap),
            torch.from_numpy(Zp),
        )


# ═══════════════════════════════════════════════════════════
# Data Generation Functions
# ═══════════════════════════════════════════════════════════

def generate_dataset(
    scenario_names: List[str],
    horizon: int,
    num_trajectories: int,
    policy: str = "mixed",
    grid_size: int = 8,
    seed: int = 42,
) -> TrajectoryDataset:
    """Generate a TrajectoryDataset from named scenarios.

    Args:
        scenario_names: List of scenario names from PREDEFINED_SCENARIOS.
        horizon: Trajectory length.
        num_trajectories: Total number of trajectories to generate.
        policy: "random", "expert", or "mixed" (80% expert, 20% random).
        grid_size: Grid dimensions.
        seed: Random seed.

    Returns:
        TrajectoryDataset ready for DataLoader.
    """
    all_trajectories = []
    rng = np.random.RandomState(seed)

    per_scenario = num_trajectories // len(scenario_names)
    remainder = num_trajectories % len(scenario_names)

    for i, name in enumerate(scenario_names):
        scenario = Scenario.from_preset(name, grid_size)
        n = per_scenario + (1 if i < remainder else 0)

        for j in range(n):
            traj_seed = int(rng.randint(0, 2**31))
            use_expert = policy == "expert" or (policy == "mixed" and rng.random() < 0.8)
            p = "expert" if use_expert else "random"

            states, actions, _ = Scenario.__dict__.get(
                "generate_trajectory", generate_trajectory
            ).__wrapped__(scenario, horizon, policy=p, seed=traj_seed)
            all_trajectories.append((states, actions))

    return TrajectoryDataset(all_trajectories, horizon, grid_size)


# Fix the reference issue
generate_dataset.__globals__["generate_trajectory"] = generate_trajectory  # type: ignore


def save_dataset(dataset: Dataset, path: str) -> None:
    """Save dataset to disk as pickle file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(dataset, f)


def load_dataset(path: str) -> Dataset:
    """Load dataset from pickle file."""
    with open(path, "rb") as f:
        return pickle.load(f)
