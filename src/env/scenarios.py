"""Scenario definitions and data generation for grid world experiments.

Defines predefined scenarios (key-door, box push, occlusion, etc.) and
a PyTorch Dataset class for training the IWCM world model.

Generated trajectories consist of (z_0, A, Z) tuples where z_0 is the
initial state, A is an action sequence, and Z is the resulting state sequence.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Iterator
from dataclasses import dataclass, field
from copy import deepcopy
import random

# Lazy imports to avoid circular dependencies
_GRID_WORLD = None


def _get_gridworld_class():
    global _GRID_WORLD
    if _GRID_WORLD is None:
        from .grid_world import GridWorld
        _GRID_WORLD = GridWorld
    return _GRID_WORLD


# ═══════════════════════════════════════════════════════════
# Predefined Scenarios
# ═══════════════════════════════════════════════════════════

PREDEFINED_SCENARIOS: Dict[str, dict] = {
    "key_door_simple": {
        "description": "1 key next to agent, 1 door between agent and goal",
        "agent_start": (6, 1),
        "goal": {"type": "position", "pos": (0, 3)},
        "objects": [
            {"type": "key", "id": "key_0", "pos": (6, 3)},
            {"type": "door", "id": "door_0", "pos": (3, 3), "key_id": "key_0"},
        ],
    },
    "key_door_long": {
        "description": "Long-range: key far from door, requiring traversal",
        "agent_start": (7, 1),
        "goal": {"type": "position", "pos": (0, 0)},
        "objects": [
            {"type": "key", "id": "key_0", "pos": (6, 7)},
            {"type": "door", "id": "door_0", "pos": (0, 4), "key_id": "key_0"},
        ],
    },
    "box_push": {
        "description": "Two boxes blocking path, must push out of the way",
        "agent_start": (4, 1),
        "goal": {"type": "position", "pos": (4, 6)},
        "objects": [
            {"type": "box", "id": "box_0", "pos": (4, 3)},
            {"type": "box", "id": "box_1", "pos": (4, 4)},
        ],
    },
    "multi_object": {
        "description": "Multiple keys, doors, and boxes requiring complex planning",
        "agent_start": (7, 1),
        "goal": {"type": "position", "pos": (0, 6)},
        "objects": [
            {"type": "key", "id": "key_0", "pos": (6, 5)},
            {"type": "key", "id": "key_1", "pos": (2, 3)},
            {"type": "door", "id": "door_0", "pos": (5, 5), "key_id": "key_0"},
            {"type": "door", "id": "door_1", "pos": (1, 3), "key_id": "key_1"},
            {"type": "box", "id": "box_0", "pos": (7, 4)},
            {"type": "box", "id": "box_1", "pos": (3, 6)},
        ],
    },
    "occlusion_test": {
        "description": "Occluders forming a wall — tests invariant tracking under occlusion",
        "agent_start": (7, 3),
        "goal": {"type": "position", "pos": (0, 4)},
        "objects": [
            {"type": "occluder", "id": "occ_0", "pos": (3, 1)},
            {"type": "occluder", "id": "occ_1", "pos": (3, 2)},
            {"type": "occluder", "id": "occ_2", "pos": (3, 3)},
            {"type": "occluder", "id": "occ_3", "pos": (3, 5)},
            {"type": "occluder", "id": "occ_4", "pos": (3, 6)},
        ],
    },
    "conservation_test": {
        "description": "Multiple keys to test conservation — keys must not duplicate/disappear",
        "agent_start": (7, 0),
        "goal": {"type": "position", "pos": (0, 7)},
        "objects": [
            {"type": "key", "id": "key_0", "pos": (7, 3)},
            {"type": "key", "id": "key_1", "pos": (7, 5)},
            {"type": "door", "id": "door_0", "pos": (3, 3), "key_id": "key_0"},
            {"type": "door", "id": "door_1", "pos": (3, 5), "key_id": "key_1"},
        ],
    },
    "splice_test": {
        "description": "Tests splice detection — two valid segments that form invalid whole",
        "agent_start": (7, 1),
        "goal": {"type": "position", "pos": (0, 7)},
        "objects": [
            {"type": "key", "id": "key_0", "pos": (7, 3)},
            {"type": "door", "id": "door_0", "pos": (3, 3), "key_id": "key_0"},
            {"type": "box", "id": "box_0", "pos": (3, 7)},
        ],
    },
    "counterfactual_test": {
        "description": "Two futures from same start — tests counterfactual constraint",
        "agent_start": (7, 1),
        "goal": {"type": "position", "pos": (0, 1)},
        "objects": [
            {"type": "key", "id": "key_0", "pos": (6, 3)},
            {"type": "key", "id": "key_1", "pos": (6, 5)},
            {"type": "door", "id": "door_0", "pos": (3, 3), "key_id": "key_0"},
        ],
    },
}


# ═══════════════════════════════════════════════════════════
# Scenario Class
# ═══════════════════════════════════════════════════════════

@dataclass
class Scenario:
    """A named grid world configuration with agent start, objects, and goal."""
    name: str
    grid_size: int = 8
    agent_start: Tuple[int, int] = (7, 1)
    goal: Dict[str, Any] = field(default_factory=lambda: {"type": "position", "pos": (0, 0)})
    objects: List[Dict[str, Any]] = field(default_factory=list)
    description: str = ""

    @classmethod
    def from_preset(cls, name: str, grid_size: int = 8) -> "Scenario":
        """Create a Scenario from a predefined layout.

        Args:
            name: One of the PREDEFINED_SCENARIOS keys.
            grid_size: Grid size (default 8).

        Returns:
            Scenario instance.
        """
        preset = PREDEFINED_SCENARIOS[name]
        return cls(
            name=name,
            grid_size=grid_size,
            agent_start=tuple(preset["agent_start"]),
            goal=deepcopy(preset["goal"]),
            objects=deepcopy(preset.get("objects", [])),
            description=preset.get("description", ""),
        )

    @classmethod
    def random(
        cls,
        grid_size: int = 8,
        num_keys: int = 1,
        num_doors: int = 1,
        num_boxes: int = 1,
        num_occluders: int = 0,
        seed: Optional[int] = None,
    ) -> "Scenario":
        """Generate a random scenario.

        Args:
            grid_size: Grid size.
            num_keys: Number of keys to place.
            num_doors: Number of doors (each matched to a key).
            num_boxes: Number of pushable boxes.
            num_occluders: Number of occluder obstacles.
            seed: Random seed.

        Returns:
            Randomly generated Scenario.
        """
        rng = np.random.RandomState(seed)
        occupied = set()

        def _random_empty_cell() -> Tuple[int, int]:
            for _ in range(100):
                r = rng.randint(0, grid_size - 1)
                c = rng.randint(0, grid_size - 1)
                if (r, c) not in occupied:
                    occupied.add((r, c))
                    return (r, c)
            raise RuntimeError("Could not find empty cell")

        # Agent start
        agent_start = _random_empty_cell()

        # Goal (far from agent)
        occupied.add(agent_start)
        goal_candidates = []
        for r in range(grid_size):
            for c in range(grid_size):
                if (r, c) not in occupied:
                    dist = abs(r - agent_start[0]) + abs(c - agent_start[1])
                    goal_candidates.append(((r, c), dist))
        goal_candidates.sort(key=lambda x: -x[1])
        goal_pos = goal_candidates[0][0] if goal_candidates else (0, 0)
        occupied.add(goal_pos)

        objects = []

        # Keys
        for i in range(num_keys):
            pos = _random_empty_cell()
            objects.append({"type": "key", "id": f"key_{i}", "pos": pos})

        # Doors (matched to keys)
        for i in range(min(num_doors, num_keys)):
            pos = _random_empty_cell()
            objects.append({
                "type": "door", "id": f"door_{i}",
                "pos": pos, "key_id": f"key_{i}",
            })

        # Boxes
        for i in range(num_boxes):
            pos = _random_empty_cell()
            objects.append({"type": "box", "id": f"box_{i}", "pos": pos})

        # Occluders
        for i in range(num_occluders):
            pos = _random_empty_cell()
            objects.append({"type": "occluder", "id": f"occ_{i}", "pos": pos})

        return cls(
            name=f"random_{seed or 'auto'}",
            grid_size=grid_size,
            agent_start=agent_start,
            goal={"type": "position", "pos": goal_pos},
            objects=objects,
            description=f"Random scenario: {num_keys}k, {num_doors}d, {num_boxes}b, {num_occluders}o",
        )

    def to_env_config(self) -> dict:
        """Convert to format accepted by GridWorld.__init__."""
        return {
            "grid_size": self.grid_size,
            "agent_start": self.agent_start,
            "goal": deepcopy(self.goal),
            "objects": deepcopy(self.objects),
        }


# ═══════════════════════════════════════════════════════════
# Trajectory Generation
# ═══════════════════════════════════════════════════════════

def generate_trajectory(
    scenario: Scenario,
    horizon: int,
    action_sequence: Optional[List[int]] = None,
    policy: str = "random",
    seed: Optional[int] = None,
    max_steps: int = 500,
) -> Tuple[List[dict], List[int], float]:
    """Generate a single trajectory from a scenario.

    Args:
        scenario: The scenario configuration.
        horizon: Number of steps to record.
        action_sequence: Predefined action list (used if provided).
        policy: "random" for random walk, "expert" for hand-crafted.
        seed: Random seed for reproducibility.
        max_steps: Maximum environment steps before truncation.

    Returns:
        Tuple of (states_list, actions_list, total_reward).
    """
    GW = _get_gridworld_class()
    env = GW(
        grid_size=scenario.grid_size,
        objects_config=scenario.to_env_config(),
        seed=seed,
    )
    env.reset()

    states: List[dict] = []
    actions: List[int] = []

    rng = np.random.RandomState(seed)
    state = env.get_state()
    states.append(deepcopy(state))

    total_reward = 0.0

    for t in range(horizon):
        if action_sequence is not None and t < len(action_sequence):
            a = action_sequence[t]
        elif policy == "random":
            valid = env.get_valid_actions()
            if not valid:
                break
            a = int(rng.choice(valid))
        elif policy == "expert":
            a = _expert_policy(env, rng)
        else:
            valid = env.get_valid_actions()
            if not valid:
                break
            a = int(rng.choice(valid))

        state, reward, done, _ = env.step(a)
        states.append(deepcopy(state))
        actions.append(a)
        total_reward += reward

        if done:
            break

    return states, actions, total_reward


def generate_trajectories(
    scenario: Scenario,
    horizon: int,
    num_trajectories: int,
    policy: str = "random",
    seed: Optional[int] = None,
) -> List[Tuple[List[dict], List[int]]]:
    """Generate multiple trajectories from the same scenario.

    Args:
        scenario: The scenario configuration.
        horizon: Number of steps per trajectory.
        num_trajectories: How many to generate.
        policy: Action selection policy.
        seed: Base seed (each trajectory gets seed + i).

    Returns:
        List of (states_list, actions_list) tuples.
    """
    rng = np.random.RandomState(seed)
    trajectories = []

    for i in range(num_trajectories):
        traj_seed = int(rng.randint(0, 2**31))
        states, actions, _ = generate_trajectory(
            scenario, horizon, policy=policy, seed=traj_seed,
        )
        trajectories.append((states, actions))

    return trajectories


def generate_counterfactual_pairs(
    scenario: Scenario,
    horizon: int,
    num_pairs: int,
    seed: Optional[int] = None,
) -> List[Tuple[List[dict], List[int], List[dict], List[int]]]:
    """Generate pairs of trajectories from the same z_0 but different actions.

    Used for testing the counterfactual consistency constraint (C_counterfactual).

    Returns:
        List of (states_A, actions_A, states_B, actions_B) tuples.
    """
    rng = np.random.RandomState(seed)
    pairs = []

    for i in range(num_pairs):
        traj_seed_a = int(rng.randint(0, 2**31))
        traj_seed_b = int(rng.randint(0, 2**31))

        env = _get_gridworld_class()(
            grid_size=scenario.grid_size,
            objects_config=scenario.to_env_config(),
            seed=traj_seed_a,
        )
        env.reset()
        z0 = deepcopy(env.get_state())

        # Trajectory A
        states_a = [deepcopy(z0)]
        actions_a = []
        rng_a = np.random.RandomState(traj_seed_a)
        for t in range(horizon):
            valid = env.get_valid_actions()
            if not valid:
                break
            a = int(rng_a.choice(valid))
            state, _, done, _ = env.step(a)
            states_a.append(deepcopy(state))
            actions_a.append(a)
            if done:
                break

        # Trajectory B — reset to same z_0, different actions
        env.reset()
        env.state = deepcopy(z0)  # force same start
        states_b = [deepcopy(z0)]
        actions_b = []
        rng_b = np.random.RandomState(traj_seed_b)
        for t in range(horizon):
            valid = env.get_valid_actions()
            if not valid:
                break
            a = int(rng_b.choice(valid))
            state, _, done, _ = env.step(a)
            states_b.append(deepcopy(state))
            actions_b.append(a)
            if done:
                break

        # Pad to same length
        min_len = min(len(states_a), len(states_b))
        pairs.append((states_a[:min_len], actions_a[:min_len-1],
                       states_b[:min_len], actions_b[:min_len-1]))

    return pairs


# ═══════════════════════════════════════════════════════════
# Expert Policy (heuristic for generating solvable trajectories)
# ═══════════════════════════════════════════════════════════

def _expert_policy(env, rng: np.random.RandomState) -> int:
    """Simple heuristic policy that moves toward the goal.

    Falls back to random walk when goal-directed movement is blocked.
    """
    from .actions import Action, MOVE_ACTIONS

    agent_r, agent_c = env.state["agent_pos"]
    goal = env.state.get("goal", {})

    # Move toward goal position
    if goal.get("type") == "position":
        gr, gc = goal["pos"]
        moves = []
        if agent_r > gr:
            moves.append(int(Action.MOVE_UP))
        if agent_r < gr:
            moves.append(int(Action.MOVE_DOWN))
        if agent_c > gc:
            moves.append(int(Action.MOVE_LEFT))
        if agent_c < gc:
            moves.append(int(Action.MOVE_RIGHT))

        # Try goal-directed moves first
        valid = env.get_valid_actions()
        for m in moves:
            if m in valid and rng.random() < 0.8:
                return m

    # Try pickup if adjacent to key
    if int(Action.PICKUP) in env.get_valid_actions() and rng.random() < 0.5:
        return int(Action.PICKUP)

    # Try open if adjacent to door (with key)
    if int(Action.OPEN) in env.get_valid_actions() and rng.random() < 0.5:
        return int(Action.OPEN)

    # Random walk
    valid = env.get_valid_actions()
    if not valid:
        return int(Action.MOVE_UP)  # fallback

    # Prefer movement over other actions
    move_actions = [a for a in valid if a in [int(a) for a in MOVE_ACTIONS]]
    if move_actions:
        return int(rng.choice(move_actions))
    return int(rng.choice(valid))


def is_solvable(
    scenario: Scenario, horizon: int, max_attempts: int = 20
) -> bool:
    """Check if a scenario is likely solvable using the expert policy.

    Args:
        scenario: The scenario to test.
        horizon: Maximum steps for solving.
        max_attempts: Number of expert policy attempts.

    Returns:
        True if at least one attempt succeeded.
    """
    for _ in range(max_attempts):
        _, _, reward = generate_trajectory(
            scenario, horizon, policy="expert", seed=random.randint(0, 2**31),
        )
        if reward > 0:
            return True
    return False
