"""AC3 symbolic mutation grammar — 7 corruption types for near-miss worlds.

Section 5.5: Defines 7 symbolic mutation operators that generate
causally invalid trajectories from valid ones by applying minimal edits.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from copy import deepcopy
from dataclasses import dataclass

from ...env.symbolic_state import SymbolicState, SymbolicTrajectory


# ═══════════════════════════════════════════════════════════
# Symbolic Trajectory wrapper for mutations
# ═══════════════════════════════════════════════════════════

@dataclass
class SymbolicTrajectory:
    """A full symbolic trajectory — states and actions."""
    states: List[SymbolicState]
    actions: List[int]
    horizon: int

    @property
    def z0(self) -> SymbolicState:
        return self.states[0]

    def __len__(self) -> int:
        return len(self.states)


# ═══════════════════════════════════════════════════════════
# Mutation Base
# ═══════════════════════════════════════════════════════════

class Mutation:
    """Base class for symbolic trajectory mutations."""
    name: str = "base"

    def apply(self, traj: SymbolicTrajectory, rng: np.random.RandomState) -> SymbolicTrajectory:
        raise NotImplementedError


# ═══════════════════════════════════════════════════════════
# Identity Corruptions
# ═══════════════════════════════════════════════════════════

class IdentitySwapMutation(Mutation):
    """Swap identities of two objects of the same type.

    Tests object permanence: can the model track which key is which?
    """

    name = "identity_swap"

    def apply(self, traj: SymbolicTrajectory, rng: np.random.RandomState) -> SymbolicTrajectory:
        states = [deepcopy(s) for s in traj.states]

        # Find two objects of same type
        obj_types = {}
        for oid, otype in states[0].object_types.items():
            if oid not in obj_types:
                obj_types[otype] = []
            obj_types[otype].append(oid)

        # Pick a type with >= 2 instances
        swappable = [t for t, ids in obj_types.items() if len(ids) >= 2]
        if not swappable:
            return SymbolicTrajectory(states=states, actions=list(traj.actions),
                                     horizon=traj.horizon)

        obj_type = rng.choice(swappable)
        candidates = obj_types[obj_type]
        id1, id2 = rng.choice(candidates, size=2, replace=False)

        # Swap their IDs in every state
        for state in states:
            if id1 in state.object_positions and id2 in state.object_positions:
                # Swap positions
                state.object_positions[id1], state.object_positions[id2] = \
                    state.object_positions[id2], state.object_positions[id1]

        return SymbolicTrajectory(states=states, actions=list(traj.actions),
                                  horizon=traj.horizon)


# ═══════════════════════════════════════════════════════════
# Conservation Corruptions
# ═══════════════════════════════════════════════════════════

class ConservationMutation(Mutation):
    """Violate conservation: duplicate or delete objects.

    Tests count/mass/inventory conservation.
    """

    name = "conservation"

    def apply(self, traj: SymbolicTrajectory, rng: np.random.RandomState) -> SymbolicTrajectory:
        states = [deepcopy(s) for s in traj.states]

        obj_ids = list(states[0].object_positions.keys())
        if not obj_ids:
            return SymbolicTrajectory(states=states, actions=list(traj.actions),
                                     horizon=traj.horizon)

        operation = rng.choice(["duplicate", "delete", "transform"])

        if operation == "duplicate":
            # Duplicate an object at a random step
            dup_id = rng.choice(obj_ids)
            new_id = dup_id + "_dup"
            t_insert = rng.randint(0, len(states))

            for t in range(t_insert, len(states)):
                if dup_id in states[t].object_positions:
                    states[t].object_positions[new_id] = states[t].object_positions[dup_id]
                    states[t].object_types[new_id] = states[t].object_types[dup_id]

        elif operation == "delete":
            del_id = rng.choice(obj_ids)
            t_delete = rng.randint(0, len(states))

            for t in range(t_delete, len(states)):
                states[t].object_positions.pop(del_id, None)
                states[t].object_types.pop(del_id, None)

        elif operation == "transform":
            trans_id = rng.choice(obj_ids)
            new_type = rng.choice(["key", "box"])
            t_transform = rng.randint(0, len(states))

            for t in range(t_transform, len(states)):
                if trans_id in states[t].object_types:
                    states[t].object_types[trans_id] = new_type

        return SymbolicTrajectory(states=states, actions=list(traj.actions),
                                  horizon=traj.horizon)


# ═══════════════════════════════════════════════════════════
# Action-Locality Corruptions
# ═══════════════════════════════════════════════════════════

class ActionLocalityMutation(Mutation):
    """Make an action affect an object outside its causal scope.

    Tests causal locality: can actions affect distant objects?
    """

    name = "action_locality"

    def apply(self, traj: SymbolicTrajectory, rng: np.random.RandomState) -> SymbolicTrajectory:
        states = [deepcopy(s) for s in traj.states]
        if len(states) < 2:
            return SymbolicTrajectory(states=states, actions=list(traj.actions),
                                     horizon=traj.horizon)

        # Pick a random timestep
        t = rng.randint(0, len(states) - 1)
        s_t = states[t]
        s_tp1 = states[t + 1]

        # Find an object that didn't change (stable)
        stable = []
        for oid in s_t.object_positions:
            if oid in s_tp1.object_positions:
                if s_t.object_positions[oid] == s_tp1.object_positions[oid]:
                    stable.append(oid)

        if stable:
            # Move a stable object — action shouldn't have affected it
            target = rng.choice(stable)
            dr = rng.choice([-1, 0, 1])
            dc = rng.choice([-1, 0, 1])
            old_pos = s_tp1.object_positions[target]
            new_pos = (old_pos[0] + dr, old_pos[1] + dc)

            # Apply the illegal change from t+1 onward
            for tt in range(t + 1, len(states)):
                if target in states[tt].object_positions:
                    # Move object to a random nearby position
                    states[tt].object_positions[target] = new_pos

        return SymbolicTrajectory(states=states, actions=list(traj.actions),
                                  horizon=traj.horizon)


# ═══════════════════════════════════════════════════════════
# Splice Corruptions
# ═══════════════════════════════════════════════════════════

class SpliceMutation(Mutation):
    """Join two valid trajectory halves into a globally invalid sequence.

    Tests long-range consistency.
    """

    name = "splice"

    def apply(self, traj: SymbolicTrajectory, rng: np.random.RandomState) -> SymbolicTrajectory:
        states = [deepcopy(s) for s in traj.states]
        if len(states) < 4:
            return SymbolicTrajectory(states=states, actions=list(traj.actions),
                                     horizon=traj.horizon)

        # Pick splice point (not at start or end)
        splice_t = rng.randint(2, len(states) - 2)

        # Shuffle order of objects at the splice
        before = states[splice_t - 1]

        # Reorder objects after splice
        obj_positions = list(before.object_positions.items())
        if len(obj_positions) >= 2:
            rng.shuffle(obj_positions)
            for t in range(splice_t, len(states)):
                for i, (oid, _) in enumerate(obj_positions):
                    if oid in states[t].object_positions:
                        states[t].object_positions[oid] = obj_positions[i][1]

        return SymbolicTrajectory(states=states, actions=list(traj.actions),
                                  horizon=traj.horizon)


class CounterfactualMismatchMutation(Mutation):
    """Given two futures differing only in action, make unaffected objects change."""

    name = "counterfactual_mismatch"

    def apply(self, traj: SymbolicTrajectory, rng: np.random.RandomState) -> SymbolicTrajectory:
        states = [deepcopy(s) for s in traj.states]
        if len(states) < 3:
            return SymbolicTrajectory(states=states, actions=list(traj.actions),
                                     horizon=traj.horizon)

        # Pick an object that stays still, move it mid-trajectory
        obj_ids = list(states[0].object_positions.keys())
        if not obj_ids:
            return SymbolicTrajectory(states=states, actions=list(traj.actions),
                                     horizon=traj.horizon)

        target = rng.choice(obj_ids)
        t_change = rng.randint(1, len(states) - 1)
        dr, dc = rng.choice([-1, 1]), rng.choice([-1, 1])

        for t in range(t_change, len(states)):
            if target in states[t].object_positions:
                old = states[t].object_positions[target]
                states[t].object_positions[target] = (old[0] + dr, old[1] + dc)

        return SymbolicTrajectory(states=states, actions=list(traj.actions),
                                  horizon=traj.horizon)


class TemporalOrderMutation(Mutation):
    """Reverse cause-effect: effect appears before cause in worldline."""

    name = "temporal_order"

    def apply(self, traj: SymbolicTrajectory, rng: np.random.RandomState) -> SymbolicTrajectory:
        states = [deepcopy(s) for s in traj.states]
        if len(states) < 3:
            return SymbolicTrajectory(states=states, actions=list(traj.actions),
                                     horizon=traj.horizon)

        # Swap two consecutive state-action pairs
        t = rng.randint(0, len(states) - 2)
        states[t], states[t + 1] = states[t + 1], states[t]
        actions = list(traj.actions)
        if t < len(actions) - 1:
            actions[t], actions[t + 1] = actions[t + 1], actions[t]

        return SymbolicTrajectory(states=states, actions=actions,
                                  horizon=traj.horizon)


class OcclusionMutation(Mutation):
    """Change object identity/velocity silently during occlusion."""

    name = "occlusion"

    def apply(self, traj: SymbolicTrajectory, rng: np.random.RandomState) -> SymbolicTrajectory:
        states = [deepcopy(s) for s in traj.states]
        if len(states) < 3:
            return SymbolicTrajectory(states=states, actions=list(traj.actions),
                                     horizon=traj.horizon)

        # Find objects, swap properties during "occlusion" (middle of trajectory)
        obj_ids = list(states[0].object_positions.keys())
        movable = [oid for oid in obj_ids
                   if states[0].object_types.get(oid) in ("key", "box")]
        if len(movable) < 2:
            return SymbolicTrajectory(states=states, actions=list(traj.actions),
                                     horizon=traj.horizon)

        id1, id2 = rng.choice(movable, size=2, replace=False)
        t_start = rng.randint(1, len(states) - 2)
        t_end = rng.randint(t_start + 1, len(states))

        # Swap identities during occlusion window
        for t in range(t_start, t_end):
            if id1 in states[t].object_positions and id2 in states[t].object_positions:
                states[t].object_positions[id1], states[t].object_positions[id2] = \
                    states[t].object_positions[id2], states[t].object_positions[id1]

        return SymbolicTrajectory(states=states, actions=list(traj.actions),
                                  horizon=traj.horizon)


# ═══════════════════════════════════════════════════════════
# Symbolic Mutation Grammar
# ═══════════════════════════════════════════════════════════

class SymbolicMutationGrammar:
    """Aggregates all 7 symbolic mutation types.

    Provides weighted sampling of mutation operators to generate
    diverse near-miss worlds for AC3 training.
    """

    # Default weights from paper (Section 5.5)
    DEFAULT_WEIGHTS = [1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5]

    def __init__(self, weights: Optional[List[float]] = None):
        self.weights = np.array(weights or self.DEFAULT_WEIGHTS, dtype=np.float64)
        self.weights = self.weights / self.weights.sum()

        self.mutations: List[Mutation] = [
            IdentitySwapMutation(),
            ConservationMutation(),
            ActionLocalityMutation(),
            SpliceMutation(),
            CounterfactualMismatchMutation(),
            TemporalOrderMutation(),
            OcclusionMutation(),
        ]

        self.mutation_names = [m.name for m in self.mutations]

    def apply(
        self, traj: SymbolicTrajectory, rng: Optional[np.random.RandomState] = None
    ) -> SymbolicTrajectory:
        """Apply a randomly selected mutation.

        Args:
            traj: Valid trajectory to corrupt.
            rng: Random state for reproducibility.

        Returns:
            Corrupted (near-miss) trajectory.
        """
        if rng is None:
            rng = np.random.RandomState()

        mutation = rng.choice(self.mutations, p=self.weights)
        return mutation.apply(traj, rng)

    def apply_multiple(
        self, traj: SymbolicTrajectory, num: int = 4,
        rng: Optional[np.random.RandomState] = None,
    ) -> List[SymbolicTrajectory]:
        """Generate multiple corrupted versions of the same trajectory.

        Args:
            traj: Valid trajectory.
            num: Number of corruptions to generate.
            rng: Random state.

        Returns:
            List of corrupted trajectories.
        """
        if rng is None:
            rng = np.random.RandomState()
        return [self.apply(traj, rng) for _ in range(num)]
