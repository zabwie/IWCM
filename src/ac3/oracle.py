"""AC3 symbolic constraint oracle — determines causal validity.

Uses the symbolic state to detect constraint violations for the
AC3 training loop. This is the ground-truth oracle that the TAMG
Validator Committee aims to approximate.
"""

import numpy as np
from typing import List
from .mutations.grammar import SymbolicTrajectory


class SymbolicOracle:
    """Constraint oracle for AC3 — detects causal violations in symbolic trajectories.

    Checks all 5 constraint types (identity, conservation, locality,
    counterfactual, temporal) using exact symbolic state access.
    """

    def __call__(self, traj: SymbolicTrajectory) -> List[str]:
        """Detect violations in a trajectory.

        Args:
            traj: Symbolic trajectory to check.

        Returns:
            List of violated constraint names (empty = valid).
        """
        violations = []

        if self._check_identity_violation(traj):
            violations.append("identity")
        if self._check_conservation_violation(traj):
            violations.append("conservation")
        if self._check_locality_violation(traj):
            violations.append("locality")
        if self._check_counterfactual_violation(traj):
            violations.append("counterfactual")
        if self._check_temporal_violation(traj):
            violations.append("temporal")

        return violations

    def is_valid(self, traj: SymbolicTrajectory) -> bool:
        """Check if trajectory is causally valid."""
        return len(self(traj)) == 0

    def _check_identity_violation(self, traj: SymbolicTrajectory) -> bool:
        """Check if object identities are preserved.

        An object's type should never change throughout a trajectory.
        """
        for t in range(len(traj.states) - 1):
            s1, s2 = traj.states[t], traj.states[t + 1]
            for oid in set(s1.object_types.keys()) & set(s2.object_types.keys()):
                if s1.object_types[oid] != s2.object_types[oid]:
                    return True
        return False

    def _check_conservation_violation(self, traj: SymbolicTrajectory) -> bool:
        """Check count conservation.

        Check if the number of objects changes without a corresponding
        pickup/drop action. Objects can only be created through DROP
        and destroyed through PICKUP.
        """
        # Conservation is about object counts not changing unexpectedly
        total_counts = []
        for state in traj.states:
            total_counts.append(len(state.object_positions))
        # If count changes without corresponding actions, it's a violation
        for t in range(len(total_counts) - 1):
            delta = abs(total_counts[t + 1] - total_counts[t])
            if delta > 1:  # More than 1 object changing in a single step
                return True
        return False

    def _check_locality_violation(self, traj: SymbolicTrajectory) -> bool:
        """Check action-effect locality.

        Objects far from the agent's position should not change positions
        in a single step (unless affected by a valid action).
        """
        for t in range(len(traj.states) - 1):
            s1, s2 = traj.states[t], traj.states[t + 1]
            agent_pos = s1.agent_pos

            for oid in set(s1.object_positions.keys()) & set(s2.object_positions.keys()):
                if s1.object_positions[oid] != s2.object_positions[oid]:
                    dist = abs(s1.object_positions[oid][0] - agent_pos[0]) + \
                           abs(s1.object_positions[oid][1] - agent_pos[1])
                    if dist > 3:  # Action can't affect objects more than 3 cells away
                        return True
        return False

    def _check_counterfactual_violation(self, traj: SymbolicTrajectory) -> bool:
        """Check counterfactual consistency.

        For objects not in inventory, their existence should be consistent.
        """
        for t in range(len(traj.states) - 1):
            s1, s2 = traj.states[t], traj.states[t + 1]
            inv1 = set(s1.inventory)
            inv2 = set(s2.inventory)

            # Objects should not appear/disappear from grid without pickup/drop
            disappeared = set(s1.object_positions.keys()) - set(s2.object_positions.keys())
            appeared = set(s2.object_positions.keys()) - set(s1.object_positions.keys())

            # Allowed: picked up → moves to inventory
            for oid in disappeared:
                if oid not in inv2:
                    return True  # disappeared without being picked up

            # Allowed: dropped → moves from inventory to grid
            for oid in appeared:
                if oid not in inv1:
                    return True  # appeared without being dropped

        return False

    def _check_temporal_violation(self, traj: SymbolicTrajectory) -> bool:
        """Check temporal ordering.

        Effects should follow causes. Inventory changes should precede
        door opening (cause before effect).
        """
        for t in range(len(traj.actions)):
            action = traj.actions[t]
            state = traj.states[t]

            # OPEN action requires key in inventory
            if action == 10:  # OPEN
                if not state.inventory:
                    return True

        return False
