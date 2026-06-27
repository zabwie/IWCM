"""Video frame renderer for grid world (Section 7.1, Experiment 2).

Renders grid world states as image frames with:
- Distinct colors per object type
- Occlusion handling (occluders hide objects behind them from camera view)
- Configurable resolution and color scheme

Generates image sequences suitable for training the video encoder
in Experiment 2 (TAMG + Validator Committee).
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from PIL import Image, ImageDraw


# ═══════════════════════════════════════════════════════════
# Color Scheme
# ═══════════════════════════════════════════════════════════

DEFAULT_COLORS: Dict[str, Tuple[int, int, int]] = {
    "background": (240, 240, 240),  # light gray
    "agent": (0, 100, 200),         # blue
    "key": (255, 215, 0),           # gold/yellow
    "door_closed": (220, 50, 50),   # red
    "door_open": (100, 200, 100),   # green
    "box": (139, 90, 43),           # brown
    "occluder": (80, 80, 80),       # dark gray
    "goal": (0, 200, 0),            # bright green
    "wall": (40, 40, 40),           # almost black
    "grid_line": (200, 200, 200),   # subtle grid lines
}

AGENT_RADIUS_RATIO = 0.35       # agent circle radius as fraction of cell size
OBJECT_RADIUS_RATIO = 0.25      # object indicator radius
DOOR_THICKNESS_RATIO = 0.15      # door line thickness


class GridWorldRenderer:
    """Renders grid world states as image frames.

    Handles camera placement (top-down view), occlusion (objects
    behind occluders are not rendered), and produces clean frames
    for video-based experiments.

    Usage:
        renderer = GridWorldRenderer(grid_size=8, cell_px=32)
        frame = renderer.render_frame(state)  # (H, W, 3) numpy array
    """

    def __init__(
        self,
        grid_size: int = 8,
        cell_px: int = 32,
        colors: Optional[Dict[str, Tuple[int, int, int]]] = None,
        show_grid: bool = True,
        camera_angle: str = "top_down",
    ):
        self.grid_size = grid_size
        self.cell_px = cell_px
        self.img_size = grid_size * cell_px
        self.show_grid = show_grid
        self.camera_angle = camera_angle
        self.colors = {**DEFAULT_COLORS, **(colors or {})}

        # Precompute half sizes for drawing
        self._cell_half = cell_px // 2
        self._agent_r = int(cell_px * AGENT_RADIUS_RATIO)
        self._obj_r = int(cell_px * OBJECT_RADIUS_RATIO)
        self._door_t = max(1, int(cell_px * DOOR_THICKNESS_RATIO))

        # Build occlusion map (cache)
        self._occlusion_map = self._build_occlusion_map()

    def _build_occlusion_map(self) -> np.ndarray:
        """Precompute which cells are occluded from top-down view.

        For top-down view, an occluder at row r blocks cells in rows r-1, r-2, ...
        directly behind it (same column) from the camera perspective.
        """
        occlusion = np.zeros((self.grid_size, self.grid_size), dtype=bool)
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                # Top-down: occluder at (r,c) blocks (r-1,c), (r-2,c), ...
                for br in range(r - 1, -1, -1):
                    occlusion[br, c] = True
        return occlusion

    def render_frame(
        self, state: dict, render_goal: bool = True
    ) -> np.ndarray:
        """Render a single state as an RGB image.

        Args:
            state: GridWorld state dict with agent_pos, objects, door_states.
            render_goal: Whether to highlight the goal position.

        Returns:
            numpy array of shape (img_size, img_size, 3) with uint8 RGB pixels.
        """
        img = Image.new("RGB", (self.img_size, self.img_size), self.colors["background"])
        draw = ImageDraw.Draw(img)

        # Draw grid lines
        if self.show_grid:
            for i in range(1, self.grid_size):
                x = i * self.cell_px
                draw.line([(x, 0), (x, self.img_size)], fill=self.colors["grid_line"], width=1)
                y = i * self.cell_px
                draw.line([(0, y), (self.img_size, y)], fill=self.colors["grid_line"], width=1)

        # Determine which cells are visible (not occluded)
        visible_mask = self._compute_visibility(state)
        occupied_cells = self._get_occupied_cells(state)

        # Draw objects in order: background first, then occluders block
        objects = state.get("objects", {})
        door_states = state.get("door_states", {})

        for obj_id, obj in objects.items():
            r, c = obj["pos"]
            obj_type = obj["type"]

            if not (0 <= r < self.grid_size and 0 <= c < self.grid_size):
                continue
            if not visible_mask[r, c]:
                continue

            cx, cy = c * self.cell_px + self._cell_half, r * self.cell_px + self._cell_half

            if obj_type == "key":
                self._draw_key(draw, cx, cy)
            elif obj_type == "door":
                is_open = door_states.get(obj_id, False)
                self._draw_door(draw, cx, cy, is_open)
            elif obj_type == "box":
                self._draw_box(draw, cx, cy)
            elif obj_type == "occluder":
                self._draw_occluder(draw, r, c)

        # Draw agent
        ar, ac = state["agent_pos"]
        if visible_mask[ar, ac]:
            ax, ay = ac * self.cell_px + self._cell_half, ar * self.cell_px + self._cell_half
            self._draw_agent(draw, ax, ay)

        # Draw goal
        if render_goal:
            goal = state.get("goal", {})
            if goal.get("type") == "position":
                gr, gc = goal["pos"]
                if visible_mask[gr, gc] and (gr, gc) not in occupied_cells:
                    gx, gy = gc * self.cell_px + self._cell_half, gr * self.cell_px + self._cell_half
                    _draw_circle(draw, gx, gy, self._obj_r, self.colors["goal"], fill=False, width=2)

        return np.array(img)

    def render_trajectory(
        self, states: List[dict], render_goal: bool = True
    ) -> np.ndarray:
        """Render a sequence of states as an image stack.

        Args:
            states: List of state dicts (length H).
            render_goal: Whether to highlight goal position.

        Returns:
            numpy array of shape (H, img_size, img_size, 3).
        """
        frames = [self.render_frame(s, render_goal) for s in states]
        return np.stack(frames, axis=0)

    def _compute_visibility(self, state: dict) -> np.ndarray:
        """Compute which cells are visible (not behind an occluder)."""
        visible = np.ones((self.grid_size, self.grid_size), dtype=bool)

        objects = state.get("objects", {})
        for obj in objects.values():
            if obj["type"] == "occluder":
                r, c = obj["pos"]
                # Occluder blocks cells behind it (lower row index = "above")
                for br in range(r - 1, -1, -1):
                    visible[br, c] = False

        return visible

    def _get_occupied_cells(self, state: dict) -> set:
        """Get set of (r, c) tuples occupied by any object."""
        occupied = set()
        for obj in state.get("objects", {}).values():
            occupied.add(tuple(obj["pos"]))
        return occupied

    def _draw_agent(self, draw: ImageDraw.ImageDraw, x: int, y: int) -> None:
        _draw_circle(draw, x, y, self._agent_r, self.colors["agent"], fill=True)
        # Border
        _draw_circle(draw, x, y, self._agent_r, (0, 0, 0), fill=False, width=1)

    def _draw_key(self, draw: ImageDraw.ImageDraw, x: int, y: int) -> None:
        r = self._obj_r
        pts = [
            (x, y - r),           # top
            (x + r, y + r // 2),  # right
            (x + r // 3, y),      # inner right
            (x, y + r // 2),      # bottom point
            (x - r // 3, y),      # inner left
            (x - r, y + r // 2),  # left
        ]
        draw.polygon(pts, fill=self.colors["key"], outline=(0, 0, 0))

    def _draw_door(
        self, draw: ImageDraw.ImageDraw, x: int, y: int, is_open: bool
    ) -> None:
        half = self._cell_half
        color = self.colors["door_open"] if is_open else self.colors["door_closed"]
        # Draw as a thick horizontal line across the cell
        draw.line(
            [(x - half, y), (x + half, y)],
            fill=color, width=self._door_t * 2,
        )
        if not is_open:
            draw.line(
                [(x - half, y), (x + half, y)],
                fill=(0, 0, 0), width=1,
            )

    def _draw_box(self, draw: ImageDraw.ImageDraw, x: int, y: int) -> None:
        r = self._obj_r
        draw.rounded_rectangle(
            [x - r, y - r, x + r, y + r],
            radius=2, fill=self.colors["box"], outline=(0, 0, 0), width=1,
        )

    def _draw_occluder(
        self, draw: ImageDraw.ImageDraw, row: int, col: int
    ) -> None:
        """Draw occluder as a filled cell with a subtle pattern."""
        x0, y0 = col * self.cell_px, row * self.cell_px
        x1, y1 = x0 + self.cell_px, y0 + self.cell_px
        draw.rectangle([x0, y0, x1, y1], fill=self.colors["occluder"])
        # Diagonal cross pattern
        draw.line([(x0, y0), (x1, y1)], fill=(60, 60, 60), width=1)
        draw.line([(x1, y0), (x0, y1)], fill=(60, 60, 60), width=1)

    def save_frame(self, state: dict, path: str) -> None:
        """Render a frame and save to file."""
        frame = self.render_frame(state)
        img = Image.fromarray(frame)
        img.save(path)

    def save_trajectory_gif(
        self, states: List[dict], path: str, duration: int = 200
    ) -> None:
        """Render a trajectory and save as animated GIF."""
        frames_arr = self.render_trajectory(states)
        frames = [Image.fromarray(f) for f in frames_arr]
        frames[0].save(
            path, save_all=True, append_images=frames[1:],
            duration=duration, loop=0,
        )


def _draw_circle(
    draw: ImageDraw.ImageDraw,
    x: int, y: int, r: int,
    color: Tuple[int, int, int],
    fill: bool = True,
    width: int = 1,
) -> None:
    """Helper to draw a circle centered at (x, y)."""
    draw.ellipse([x - r, y - r, x + r, y + r], fill=color if fill else None, outline=color, width=width)
