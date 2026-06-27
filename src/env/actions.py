"""Action definitions for the grid world environment.

Defines the action space, direction deltas, and action applicability
checks used by GridWorld.step().
"""

from enum import IntEnum
from typing import Dict, Tuple, List


class Action(IntEnum):
    """Grid world action space with 11 discrete actions."""
    MOVE_UP = 0
    MOVE_DOWN = 1
    MOVE_LEFT = 2
    MOVE_RIGHT = 3
    PICKUP = 4
    DROP = 5
    PUSH_UP = 6
    PUSH_DOWN = 7
    PUSH_LEFT = 8
    PUSH_RIGHT = 9
    OPEN = 10


N_ACTIONS = 11

# Action name lookup
ACTION_NAMES: Dict[int, str] = {a.value: a.name for a in Action}

# Direction deltas for movement and push actions
DIRECTION_DELTAS: Dict[int, Tuple[int, int]] = {
    Action.MOVE_UP: (-1, 0),
    Action.MOVE_DOWN: (1, 0),
    Action.MOVE_LEFT: (0, -1),
    Action.MOVE_RIGHT: (0, 1),
}

# Push → corresponding Move direction
PUSH_TO_MOVE: Dict[int, int] = {
    Action.PUSH_UP: Action.MOVE_UP,
    Action.PUSH_DOWN: Action.MOVE_DOWN,
    Action.PUSH_LEFT: Action.MOVE_LEFT,
    Action.PUSH_RIGHT: Action.MOVE_RIGHT,
}

# Movement actions only
MOVE_ACTIONS: List[int] = [
    Action.MOVE_UP, Action.MOVE_DOWN,
    Action.MOVE_LEFT, Action.MOVE_RIGHT,
]

# Push actions only
PUSH_ACTIONS: List[int] = [
    Action.PUSH_UP, Action.PUSH_DOWN,
    Action.PUSH_LEFT, Action.PUSH_RIGHT,
]

# All directional actions (move + push)
DIRECTIONAL_ACTIONS: List[int] = MOVE_ACTIONS + PUSH_ACTIONS
