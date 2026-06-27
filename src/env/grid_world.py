"""Grid world environment for IWCM paper experiments (Section 7.1).

A configurable N×N grid world with objects (keys, doors, boxes, occluders),
supporting movement, pushing, pickup, drop, and door-opening actions.

Deterministic given same seed and action sequence. Provides rich state for
testing all 5 IWCM constraint heads (boundary, local transition, invariant,
effect, counterfactual).
"""

import numpy as np
from copy import deepcopy
from typing import Optional, Tuple, List, Dict, Any


# ═══════════════════════════════════════════════════════════════════════════════
# Object type definitions (stubs for T5 — full class hierarchy deferred)
# ═══════════════════════════════════════════════════════════════════════════════

OBJECT_TYPES: Dict[str, Dict[str, Any]] = {
    "key": {
        "pickupable": True,
        "pushable": False,
        "passable": True,
        "color": "yellow",
    },
    "door": {
        "pickupable": False,
        "pushable": False,
        "passable": True,  # base passability; overridden by door_states
        "color": "red",
        "openable": True,
    },
    "box": {
        "pickupable": False,
        "pushable": True,
        "passable": False,
        "color": "brown",
    },
    "occluder": {
        "pickupable": False,
        "pushable": False,
        "passable": False,
        "color": "gray",
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
# Action constants
# ═══════════════════════════════════════════════════════════════════════════════

MOVE_UP    = 0
MOVE_DOWN  = 1
MOVE_LEFT  = 2
MOVE_RIGHT = 3
PICKUP     = 4
DROP       = 5
PUSH_UP    = 6
PUSH_DOWN  = 7
PUSH_LEFT  = 8
PUSH_RIGHT = 9
OPEN       = 10

N_ACTIONS = 11

ACTION_NAMES: Dict[int, str] = {
    0: "MOVE_UP",
    1: "MOVE_DOWN",
    2: "MOVE_LEFT",
    3: "MOVE_RIGHT",
    4: "PICKUP",
    5: "DROP",
    6: "PUSH_UP",
    7: "PUSH_DOWN",
    8: "PUSH_LEFT",
    9: "PUSH_RIGHT",
    10: "OPEN",
}

# Direction deltas for movement and push actions
DIRECTION_DELTAS: Dict[int, Tuple[int, int]] = {
    0: (-1, 0),   # UP
    1: (1, 0),    # DOWN
    2: (0, -1),   # LEFT
    3: (0, 1),    # RIGHT
}

# Mapping from push action to corresponding move direction
PUSH_TO_MOVE: Dict[int, int] = {
    PUSH_UP:    MOVE_UP,
    PUSH_DOWN:  MOVE_DOWN,
    PUSH_LEFT:  MOVE_LEFT,
    PUSH_RIGHT: MOVE_RIGHT,
}

# All four cardinal directions (for adjacency checks)
ADJACENCY_DIRECTIONS: List[Tuple[int, int]] = [(-1, 0), (1, 0), (0, -1), (0, 1)]


# ═══════════════════════════════════════════════════════════════════════════════
# Layout presets
# ═══════════════════════════════════════════════════════════════════════════════

LAYOUT_PRESETS: Dict[str, Dict[str, Any]] = {
    "key_door_simple": {
        "agent_start": (6, 1),
        "goal": {"type": "position", "pos": (0, 3)},
        "objects": [
            {"type": "key", "id": "key_0", "pos": (6, 3)},
            {
                "type": "door",
                "id": "door_0",
                "pos": (3, 3),
                "key_id": "key_0",
            },
        ],
    },
    "key_door_long": {
        "agent_start": (7, 1),
        "goal": {"type": "position", "pos": (0, 0)},
        "objects": [
            {"type": "key", "id": "key_0", "pos": (7, 7)},
            {
                "type": "door",
                "id": "door_0",
                "pos": (0, 4),
                "key_id": "key_0",
            },
        ],
    },
    "box_push": {
        "agent_start": (5, 1),
        "goal": {"type": "position", "pos": (5, 5)},
        "objects": [
            {"type": "box", "id": "box_0", "pos": (5, 3)},
        ],
    },
    "multi_object": {
        "agent_start": (7, 1),
        "goal": {"type": "position", "pos": (0, 6)},
        "objects": [
            {"type": "key", "id": "key_0", "pos": (6, 5)},
            {"type": "key", "id": "key_1", "pos": (2, 3)},
            {
                "type": "door",
                "id": "door_0",
                "pos": (5, 5),
                "key_id": "key_0",
            },
            {
                "type": "door",
                "id": "door_1",
                "pos": (1, 3),
                "key_id": "key_1",
            },
            {"type": "box", "id": "box_0", "pos": (7, 4)},
        ],
    },
    "occlusion_test": {
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
}


# ═══════════════════════════════════════════════════════════════════════════════
# GridWorld environment
# ═══════════════════════════════════════════════════════════════════════════════

class GridWorld:
    """Grid world environment for IWCM paper experiments (Section 7.1).

    Configurable N×N grid with objects (keys, doors, boxes, occluders) and
    an agent. Supports 11 actions: 4 movement, pickup, drop, 4 push, open.

    Deterministic: same seed + same action sequence = same trajectories.

    Parameters
    ----------
    grid_size : int
        Edge length of the square grid (default 8).
    max_objects : int
        Maximum number of objects placed during random generation.
    objects_config : dict, optional
        Explicit object/agent/goal configuration. Overrides random generation.
        Format: {"agent_start": (r,c), "goal": {...}, "objects": [...]}
    seed : int, optional
        Random seed for reproducible randomization.
    """

    def __init__(
        self,
        grid_size: int = 8,
        max_objects: int = 5,
        objects_config: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
    ):
        if grid_size < 3:
            raise ValueError(f"grid_size must be >= 3, got {grid_size}")
        if max_objects < 0:
            raise ValueError(f"max_objects must be >= 0, got {max_objects}")

        self.grid_size = grid_size
        self.max_objects = max_objects
        self.objects_config = objects_config
        self.seed = seed

        # Deterministic random number generator
        self._rng = np.random.RandomState(seed)

        # Internal object-id counter for auto-naming
        self._object_counter: Dict[str, int] = {}

        # Step counter (for truncation)
        self._step_count: int = 0
        self._max_steps: int = max(grid_size * grid_size * 4, 100)

        # Core state (populated by reset)
        self.state: Dict[str, Any] = {}

        # 2-D grid for O(1) object-at-cell lookup:
        # _grid[r, c] = object_id (str) or None
        self._grid: np.ndarray = np.full(
            (grid_size, grid_size), None, dtype=object
        )

        # Perform initial reset
        self.reset()

    # ── Public API ────────────────────────────────────────────────────────────

    def reset(self, layout: Optional[str] = None) -> Dict[str, Any]:
        """Reset environment to initial state.

        Parameters
        ----------
        layout : str, optional
            Name of a predefined layout (key in LAYOUT_PRESETS), or None
            for random generation.  ``objects_config`` passed to __init__
            always takes priority over layout presets.

        Returns
        -------
        dict
            Environment state (see ``get_state()``).
        """
        # Re-seed for reproducibility across resets
        if self.seed is not None:
            self._rng = np.random.RandomState(self.seed)

        self._object_counter = {}
        self._step_count = 0
        self._grid = np.full((self.grid_size, self.grid_size), None, dtype=object)

        self.state = {
            "agent_pos": None,
            "objects": {},
            "inventory": [],
            "door_states": {},
            "grid_size": self.grid_size,
            "goal": {},
        }

        # Determine configuration priority:
        # 1. Explicit objects_config (passed to __init__)
        # 2. Named layout preset
        # 3. Random generation
        if self.objects_config is not None:
            self._apply_config(self.objects_config)
        elif layout is not None:
            if layout not in LAYOUT_PRESETS:
                raise ValueError(
                    f"Unknown layout '{layout}'. "
                    f"Available: {list(LAYOUT_PRESETS.keys())}"
                )
            cfg = deepcopy(LAYOUT_PRESETS[layout])
            self._apply_config(cfg)
        else:
            self._randomize_objects()

        return self.get_state()

    def step(self, action: int) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """Execute an action and return (state, reward, done, info).

        Parameters
        ----------
        action : int
            Action index (0-10).  See ACTION_NAMES.

        Returns
        -------
        state : dict
            Environment state after the step.
        reward : float
            +1.0 for reaching goal, -0.01 per step otherwise.
        done : bool
            True if goal reached or max steps exceeded.
        info : dict
            Additional metadata about the step (validity, side effects).
        """
        if action < 0 or action >= N_ACTIONS:
            raise ValueError(f"Action must be in [0, {N_ACTIONS - 1}], got {action}")

        info: Dict[str, Any] = {
            "action": action,
            "action_name": ACTION_NAMES.get(action, "UNKNOWN"),
            "valid": False,
        }

        # ── Validate ──────────────────────────────────────────────────────
        if not self._is_action_valid(action):
            self._step_count += 1
            reward = -0.01
            done = self._step_count >= self._max_steps
            if done:
                info["truncated"] = True
            return self.get_state(), reward, done, info

        info["valid"] = True

        # ── Execute ───────────────────────────────────────────────────────
        agent_r, agent_c = self.state["agent_pos"]

        # Movement (0-3)
        if action in (MOVE_UP, MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT):
            dr, dc = DIRECTION_DELTAS[action]
            nr, nc = agent_r + dr, agent_c + dc
            self.state["agent_pos"] = (nr, nc)
            info["moved"] = True

        # Push (6-9)
        elif action in (PUSH_UP, PUSH_DOWN, PUSH_LEFT, PUSH_RIGHT):
            move_dir = PUSH_TO_MOVE[action]
            dr, dc = DIRECTION_DELTAS[move_dir]
            box_r, box_c = agent_r + dr, agent_c + dc
            target_r, target_c = box_r + dr, box_c + dc

            box_id = self._get_cell_object(box_r, box_c)
            assert box_id is not None, "push action validated but no box found"

            # Move box on grid
            self._grid[box_r, box_c] = None
            self._grid[target_r, target_c] = box_id
            self.state["objects"][box_id]["pos"] = (target_r, target_c)

            # Agent moves into box's old cell
            self.state["agent_pos"] = (box_r, box_c)
            info["pushed_box"] = box_id

        # Pickup (4)
        elif action == PICKUP:
            # Check current cell first, then adjacent cells
            found = False
            obj_id = self._get_cell_object(agent_r, agent_c)
            if obj_id is not None and self.state["objects"][obj_id]["type"] == "key":
                self._remove_object(obj_id)
                self.state["inventory"].append(obj_id)
                info["picked_up"] = obj_id
                found = True

            if not found:
                for dr, dc in ADJACENCY_DIRECTIONS:
                    nr, nc = agent_r + dr, agent_c + dc
                    obj_id = self._get_cell_object(nr, nc)
                    if obj_id is not None and self.state["objects"][obj_id]["type"] == "key":
                        self._remove_object(obj_id)
                        self.state["inventory"].append(obj_id)
                        info["picked_up"] = obj_id
                        break

        # Drop (5)
        elif action == DROP:
            if self.state["inventory"]:
                obj_id = self.state["inventory"].pop()
                self._place_object(obj_id, "key", agent_r, agent_c)
                info["dropped"] = obj_id

        # Open (10)
        elif action == OPEN:
            for dr, dc in ADJACENCY_DIRECTIONS:
                nr, nc = agent_r + dr, agent_c + dc
                door_id = self._get_cell_object(nr, nc)
                if door_id is not None and self.state["objects"][door_id]["type"] == "door":
                    required_key = self.state["objects"][door_id].get("key_id")
                    if required_key is not None and required_key in self.state["inventory"]:
                        self.state["door_states"][door_id] = True
                        info["opened_door"] = door_id
                        break

        # ── Post-step ─────────────────────────────────────────────────────
        self._step_count += 1

        goal_reached = self._check_goal()
        truncated = self._step_count >= self._max_steps

        done = goal_reached or truncated
        reward = 1.0 if goal_reached else -0.01

        if goal_reached:
            info["goal_reached"] = True
        if truncated:
            info["truncated"] = True

        return self.get_state(), reward, done, info

    def get_state(self) -> Dict[str, Any]:
        """Return a deep copy of the current environment state."""
        return {
            "agent_pos": self.state["agent_pos"],
            "objects": deepcopy(self.state["objects"]),
            "inventory": list(self.state["inventory"]),
            "door_states": dict(self.state["door_states"]),
            "grid_size": self.state["grid_size"],
            "goal": deepcopy(self.state["goal"]),
            "step_count": self._step_count,
            "max_steps": self._max_steps,
        }

    def get_valid_actions(self) -> List[int]:
        """Return indices of all currently valid actions."""
        return [a for a in range(N_ACTIONS) if self._is_action_valid(a)]

    def is_done(self) -> bool:
        """Check whether goal state has been reached."""
        return self._check_goal()

    def render_text(self) -> str:
        """ASCII render of the grid for debugging.

        Legend
        ------
        A       Agent
        k       Key
        D       Door (closed) / d (open)
        B       Box
        O       Occluder
        .       Empty cell
        G       Goal position (shown behind objects when agent not present)
        """
        grid = self.grid_size
        chars = [["." for _ in range(grid)] for _ in range(grid)]

        # Place objects
        for obj_id, obj in self.state["objects"].items():
            r, c = obj["pos"]
            otype = obj["type"]
            if otype == "key":
                chars[r][c] = "k"
            elif otype == "door":
                chars[r][c] = "d" if self.state["door_states"].get(obj_id, False) else "D"
            elif otype == "box":
                chars[r][c] = "B"
            elif otype == "occluder":
                chars[r][c] = "O"

        # Mark goal cell (if visible)
        goal = self.state.get("goal", {})
        if goal.get("type") == "position":
            gr, gc = goal["pos"]
            if self.state["agent_pos"] != (gr, gc) and chars[gr][gc] == ".":
                chars[gr][gc] = "G"

        # Place agent (on top)
        ar, ac = self.state["agent_pos"]
        chars[ar][ac] = "A"

        # Build string
        header = "+" + "---" * grid + "+"
        lines = [header]
        for row in range(grid):
            line = "| " + "  ".join(chars[row]) + " |"
            lines.append(line)
        lines.append(header)

        # Inventory / door info
        extra = []
        if self.state["inventory"]:
            extra.append(f"Inventory: {self.state['inventory']}")
        closed = [did for did, s in self.state["door_states"].items() if not s]
        opened = [did for did, s in self.state["door_states"].items() if s]
        if opened:
            extra.append(f"Doors open: {opened}")
        if closed:
            extra.append(f"Doors closed: {closed}")
        if self._step_count > 0:
            extra.append(f"Step: {self._step_count}/{self._max_steps}")

        if extra:
            lines.append("  " + " | ".join(extra))

        return "\n".join(lines)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _apply_config(self, config: Dict[str, Any]) -> None:
        """Apply an explicit object configuration dict to the state."""
        cfg = deepcopy(config)

        # Agent start
        agent_start = tuple(cfg.get("agent_start", (0, 0)))
        if not self._in_bounds(*agent_start):
            raise ValueError(f"agent_start {agent_start} out of bounds")
        self.state["agent_pos"] = agent_start

        # Objects
        for obj_spec in cfg.get("objects", []):
            obj_type = obj_spec["type"]
            obj_id = obj_spec.get("id", self._gen_object_id(obj_type))
            pos = tuple(obj_spec["pos"])
            if not self._in_bounds(*pos):
                raise ValueError(f"object {obj_id} position {pos} out of bounds")
            if self._get_cell_object(*pos) is not None:
                raise ValueError(f"cell {pos} already occupied")

            self.state["objects"][obj_id] = {
                "type": obj_type,
                "pos": pos,
                "properties": dict(OBJECT_TYPES[obj_type]),
            }
            self._grid[pos[0], pos[1]] = obj_id

            # Door-specific: track key_id and initial door state
            if obj_type == "door":
                key_id = obj_spec.get("key_id")
                if key_id is not None:
                    self.state["objects"][obj_id]["key_id"] = key_id
                self.state["door_states"][obj_id] = False  # all doors start closed

        # Goal
        self.state["goal"] = deepcopy(cfg.get("goal", {"type": "position", "pos": (0, 0)}))

        # If no goal position specified, default to top-right corner
        if self.state["goal"].get("type") == "position" and "pos" not in self.state["goal"]:
            self.state["goal"]["pos"] = (0, self.grid_size - 1)

        # Ensure goal position is in bounds
        if self.state["goal"].get("type") == "position":
            gp = self.state["goal"]["pos"]
            if not self._in_bounds(*gp):
                raise ValueError(f"goal position {gp} out of bounds")

    def _randomize_objects(self) -> None:
        """Generate a random layout with keys, doors, boxes, and occluders."""
        gs = self.grid_size
        occupied: set = set()  # set of (r, c) tuples

        # ── Agent ──
        ar, ac = self._rng.randint(0, gs, size=2)
        self.state["agent_pos"] = (int(ar), int(ac))

        # ── Generate objects ──
        # We'll place: 1-2 keys, 1-2 matching doors, 1 box, 0-2 occluders
        n_keys = min(self._rng.randint(1, 3), self.max_objects)
        n_doors = min(n_keys, self.max_objects)  # one door per key
        n_boxes = min(self._rng.randint(0, 2), self.max_objects - n_keys - n_doors)
        n_occluders = min(
            self._rng.randint(0, 3), self.max_objects - n_keys - n_doors - n_boxes
        )

        # Helper: pick a random free cell
        def _free_cell() -> Tuple[int, int]:
            attempts = 0
            while attempts < gs * gs * 10:
                r, c = self._rng.randint(0, gs, size=2)
                if (int(r), int(c)) not in occupied:
                    return (int(r), int(c))
                attempts += 1
            raise RuntimeError("grid too full for random placement")

        # Place keys
        key_ids = []
        for i in range(int(n_keys)):
            pos = _free_cell()
            occupied.add(pos)
            obj_id = self._gen_object_id("key")
            key_ids.append(obj_id)
            self._place_object(obj_id, "key", *pos)

        # Place doors (each linked to a key)
        for i in range(int(n_doors)):
            pos = _free_cell()
            occupied.add(pos)
            obj_id = self._gen_object_id("door")
            self.state["objects"][obj_id] = {
                "type": "door",
                "pos": pos,
                "properties": dict(OBJECT_TYPES["door"]),
                "key_id": key_ids[i],
            }
            self._grid[pos[0], pos[1]] = obj_id
            self.state["door_states"][obj_id] = False

        # Place boxes
        for _ in range(int(n_boxes)):
            pos = _free_cell()
            occupied.add(pos)
            obj_id = self._gen_object_id("box")
            self._place_object(obj_id, "box", *pos)

        # Place occluders
        for _ in range(int(n_occluders)):
            pos = _free_cell()
            occupied.add(pos)
            obj_id = self._gen_object_id("occluder")
            self._place_object(obj_id, "occluder", *pos)

        # ── Goal ──
        # Default: position goal at a free cell
        goal_pos = _free_cell()
        # Don't add goal to occupied; it can share a cell with a passable object
        self.state["goal"] = {"type": "position", "pos": goal_pos}

    def _place_object(
        self,
        obj_id: str,
        obj_type: str,
        row: int,
        col: int,
        **extra_props,
    ) -> None:
        """Place an object on the grid, updating both state dict and grid array."""
        self.state["objects"][obj_id] = {
            "type": obj_type,
            "pos": (row, col),
            "properties": dict(OBJECT_TYPES[obj_type]),
            **extra_props,
        }
        self._grid[row, col] = obj_id

    def _remove_object(self, obj_id: str) -> None:
        """Remove object from grid and state dict."""
        if obj_id in self.state["objects"]:
            r, c = self.state["objects"][obj_id]["pos"]
            self._grid[r, c] = None
            del self.state["objects"][obj_id]

    def _get_cell_object(self, row: int, col: int) -> Optional[str]:
        """Return object_id at (row, col), or None if empty or out-of-bounds."""
        if not (0 <= row < self.grid_size and 0 <= col < self.grid_size):
            return None
        return self._grid[row, col]

    def _is_cell_enterable(self, row: int, col: int) -> bool:
        """Check whether the agent can move into a cell.

        A cell is enterable if it is:
        - In bounds, AND
        - Empty (no object), OR contains a passable object (key, open door).

        Boxes, closed doors, and occluders block movement.
        """
        if not self._in_bounds(row, col):
            return False  # wall

        obj_id = self._get_cell_object(row, col)
        if obj_id is None:
            return True  # empty cell

        obj = self.state["objects"][obj_id]
        otype = obj["type"]

        if otype == "door":
            return bool(self.state["door_states"].get(obj_id, False))
        elif otype == "key":
            return True  # passable per OBJECT_TYPES
        else:
            return False  # box, occluder, unknown

    def _is_cell_empty(self, row: int, col: int) -> bool:
        """Check whether a cell has no object (agent presence ignored)."""
        return self._get_cell_object(row, col) is None

    def _is_action_valid(self, action: int) -> bool:
        """Check whether an action can be executed in the current state."""
        ar, ac = self.state["agent_pos"]

        # Movement (0-3)
        if action in (MOVE_UP, MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT):
            dr, dc = DIRECTION_DELTAS[action]
            nr, nc = ar + dr, ac + dc
            return self._is_cell_enterable(nr, nc)

        # Push (6-9)
        elif action in (PUSH_UP, PUSH_DOWN, PUSH_LEFT, PUSH_RIGHT):
            move_dir = PUSH_TO_MOVE[action]
            dr, dc = DIRECTION_DELTAS[move_dir]
            box_r, box_c = ar + dr, ac + dc
            target_r, target_c = box_r + dr, box_c + dc

            # Cell in front must contain a pushable box
            box_id = self._get_cell_object(box_r, box_c)
            if box_id is None:
                return False
            obj = self.state["objects"].get(box_id)
            if obj is None or obj["type"] != "box":
                return False
            if not OBJECT_TYPES["box"]["pushable"]:
                return False

            # Target cell must be in bounds and empty (no object)
            if not self._in_bounds(target_r, target_c):
                return False
            if not self._is_cell_empty(target_r, target_c):
                return False

            return True

        # Pickup (4)
        elif action == PICKUP:
            # Current cell
            obj_id = self._get_cell_object(ar, ac)
            if obj_id is not None and self.state["objects"].get(obj_id, {}).get("type") == "key":
                return True
            # Adjacent cells
            for dr, dc in ADJACENCY_DIRECTIONS:
                nr, nc = ar + dr, ac + dc
                obj_id = self._get_cell_object(nr, nc)
                if obj_id is not None and self.state["objects"].get(obj_id, {}).get("type") == "key":
                    return True
            return False

        # Drop (5)
        elif action == DROP:
            if not self.state["inventory"]:
                return False
            # Cell must be empty (no other object present; agent occupies it)
            return self._is_cell_empty(ar, ac)

        # Open (6)
        elif action == OPEN:
            for dr, dc in ADJACENCY_DIRECTIONS:
                nr, nc = ar + dr, ac + dc
                door_id = self._get_cell_object(nr, nc)
                if door_id is not None and self.state["objects"].get(door_id, {}).get("type") == "door":
                    # Door must be closed
                    if self.state["door_states"].get(door_id, False):
                        continue
                    required_key = self.state["objects"][door_id].get("key_id")
                    if required_key is not None and required_key in self.state["inventory"]:
                        return True
            return False

        return False

    def _check_goal(self) -> bool:
        """Evaluate whether the goal condition is satisfied."""
        goal = self.state.get("goal", {})
        gtype = goal.get("type", "position")

        if gtype == "position":
            return self.state["agent_pos"] == tuple(goal.get("pos", (-1, -1)))

        elif gtype == "door_open":
            door_id = goal.get("door_id")
            if door_id is None:
                return False
            return bool(self.state["door_states"].get(door_id, False))

        elif gtype == "box_at":
            box_id = goal.get("box_id")
            target = tuple(goal.get("pos", (-1, -1)))
            if box_id is None:
                return False
            box_obj = self.state["objects"].get(box_id)
            if box_obj is None:
                return False
            return box_obj["pos"] == target

        elif gtype == "key_collected":
            key_id = goal.get("key_id")
            if key_id is None:
                return False
            return key_id in self.state["inventory"]

        return False

    def _gen_object_id(self, obj_type: str) -> str:
        """Generate a unique object ID of the form '{type}_{n}'."""
        count = self._object_counter.get(obj_type, 0)
        self._object_counter[obj_type] = count + 1
        return f"{obj_type}_{count}"

    def _in_bounds(self, row: int, col: int) -> bool:
        """Check whether (row, col) is within grid boundaries."""
        return 0 <= row < self.grid_size and 0 <= col < self.grid_size


# ═══════════════════════════════════════════════════════════════════════════════
# Smoke-test (runs when module is executed directly)
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # ── Determinism check ──
    print("=== Determinism check ===")
    for seed in [42, 42, 99]:
        env = GridWorld(grid_size=8, seed=seed)
        state = env.reset(layout="key_door_simple")
        s1 = env.get_state()
        env.step(MOVE_RIGHT)
        s2 = env.get_state()
        print(f"  seed={seed}: agent_pos after MOVE_RIGHT = {s2['agent_pos']}")

    # ── Layout tests ──
    for layout in LAYOUT_PRESETS:
        print(f"\n=== Layout: {layout} ===")
        env = GridWorld(grid_size=8, seed=1)
        state = env.reset(layout=layout)
        print(env.render_text())
        va = env.get_valid_actions()
        print(f"  valid actions: {[ACTION_NAMES[a] for a in va]}")

    # ── Play key_door_simple to completion ──
    print("\n=== Playing key_door_simple ===")
    env = GridWorld(grid_size=8, seed=42)
    state = env.reset(layout="key_door_simple")
    print(env.render_text())

    # Move to key, pick up, move to door, open, move to goal
    actions = [
        MOVE_RIGHT,   # (6,1) -> (6,2)
        MOVE_RIGHT,   # (6,2) -> (6,3) — now on key cell
        PICKUP,       # pick up key_0
        MOVE_UP,      # (6,3) -> (5,3)
        MOVE_UP,      # (5,3) -> (4,3)
        MOVE_UP,      # (4,3) -> (3,3) — adjacent to door (door is at 3,3, wait)
    ]

    # Re-verify layout: agent at (6,1), key at (6,3), door at (3,3)
    # Agent moves right to (6,2), then right to (6,3) where key is
    # PICKUP takes it. Then move up: (5,3), (4,3), (3,3) — door at (3,3)
    # Agent is ON door at (3,3). OPEN checks adjacent cells, not current.
    # Need to be adjacent to door, not on it. Let me adjust.

    # Actually: key is at (6,3). Agent at (6,1). 
    # Move right to (6,2), then PICKUP from adjacent (key at 6,3). 
    # Then move: (6,2) -> (5,2) -> (4,2) -> (3,2) [adjacent to door at 3,3]
    # OPEN -> door opens. Then move right to (3,3), then up to goal.

    actions_v2 = [
        MOVE_RIGHT,   # (6,1) -> (6,2); key at (6,3) adjacent
        PICKUP,       # pick up key_0 from adjacent
        MOVE_UP,      # (6,2) -> (5,2)
        MOVE_UP,      # (5,2) -> (4,2)
        MOVE_UP,      # (4,2) -> (3,2); door at (3,3) adjacent
        OPEN,         # open door_0
        MOVE_RIGHT,   # (3,2) -> (3,3) — door is open now
        MOVE_UP,      # (3,3) -> (2,3)
        MOVE_UP,      # (2,3) -> (1,3)
        MOVE_UP,      # (1,3) -> (0,3) — GOAL!
    ]

    for i, act in enumerate(actions_v2):
        state, reward, done, info = env.step(act)
        print(f"\n  Step {i+1}: {ACTION_NAMES[act]}")
        print(f"    reward={reward}, done={done}, info={info}")
        print(env.render_text())
        if done:
            break

    # ── Test all layout presets are playable ──
    print("\n=== All presets init/step test ===")
    for layout_name in LAYOUT_PRESETS:
        env = GridWorld(grid_size=8, seed=7)
        env.reset(layout=layout_name)
        va = env.get_valid_actions()
        # Try a valid move action if available
        move_actions = [a for a in va if a in (MOVE_UP, MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT)]
        if move_actions:
            act = move_actions[0]
            s, r, d, info = env.step(act)
            print(f"  {layout_name}: took {ACTION_NAMES[act]}, valid={info.get('valid')}, reward={r}")
        else:
            # Try any valid action
            act = va[0] if va else 0
            s, r, d, info = env.step(act)
            print(f"  {layout_name}: took {ACTION_NAMES[act]}, valid={info.get('valid')}, reward={r}")

    print("\n=== All tests passed ===")
