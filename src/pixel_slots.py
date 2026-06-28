"""Pixel → object-slot extraction via color segmentation and tracking.

No oracle. No neural network. No simulator state access.
Works on rendered grid-world frames (H, W, 3) numpy arrays.

Approach:
  1. Color thresholding to find objects per frame
  2. Connected components to separate same-color objects
  3. Hungarian matching across frames for object identity
  4. Encode as slot tensor: type, position, velocity, existence

This enables the "from pixels" claim: SimpleTAMG operates on slots
extracted from rendered frames, not from oracle simulator state.
"""

import numpy as np
from scipy.ndimage import label as connected_components
from scipy.optimize import linear_sum_assignment

MAX_OBJECTS = 8
SLOT_DIM = 19

COLOR_MAP = {
    "agent": np.array([0, 100, 200]),
    "key": np.array([255, 215, 0]),
    "door_closed": np.array([220, 50, 50]),
    "door_open": np.array([100, 200, 100]),
    "box": np.array([139, 90, 43]),
    "occluder": np.array([80, 80, 80]),
}

TYPE_INDEX = {"agent": 0, "key": 1, "door": 2, "box": 3, "occluder": 4}
THRESHOLD = 40


def _color_mask(frame, target_color):
    diff = np.abs(frame.astype(np.float32) - target_color.astype(np.float32))
    return (diff.max(axis=-1) < THRESHOLD)


def _extract_objects(frame, grid_size):
    objects = []
    for obj_type, color in COLOR_MAP.items():
        mask = _color_mask(frame, color)
        labeled, n = connected_components(mask)
        for i in range(1, n + 1):
            ys, xs = np.where(labeled == i)
            if len(ys) < 3:
                continue
            cy, cx = ys.mean(), xs.mean()
            ny = cy / frame.shape[0]
            nx = cx / frame.shape[1]
            if obj_type == "door_closed":
                obj_type_key = "door"
                door_state = 1.0
            elif obj_type == "door_open":
                obj_type_key = "door"
                door_state = 0.0
            else:
                obj_type_key = obj_type
                door_state = 0.0
            objects.append({
                "type": obj_type_key,
                "pos": np.array([ny, nx], dtype=np.float32),
                "area": len(ys),
                "door_state": door_state,
            })
    return objects


def _match_objects(prev_objects, curr_objects, max_dist=0.3):
    if not prev_objects or not curr_objects:
        return {}
    n_prev, n_curr = len(prev_objects), len(curr_objects)
    cost = np.ones((n_prev, n_curr)) * 1e6
    for i, po in enumerate(prev_objects):
        for j, co in enumerate(curr_objects):
            if po["type"] == co["type"]:
                dist = np.linalg.norm(po["pos"] - co["pos"])
                if dist < max_dist:
                    cost[i, j] = dist
    row_ind, col_ind = linear_sum_assignment(cost)
    matches = {}
    for r, c in zip(row_ind, col_ind):
        if cost[r, c] < max_dist:
            matches[r] = c
    return matches


def extract_slots(frames, grid_size=8):
    """Extract object slots from a sequence of rendered frames.

    Args:
        frames: list of (H, W, 3) numpy arrays, one per timestep
        grid_size: grid world size (for normalization context)

    Returns:
        Z: (H, MAX_OBJECTS, SLOT_DIM) slot tensor
    """
    H = len(frames)
    all_objects = [_extract_objects(f, grid_size) for f in frames]

    tracks = {}
    next_track_id = 0

    for t in range(H):
        curr = all_objects[t]
        if t == 0:
            for obj in curr:
                tracks[next_track_id] = {"start": t, "end": t, "type": obj["type"],
                                          "positions": {t: obj["pos"]}}
                next_track_id += 1
        else:
            prev = all_objects[t - 1]
            matches = _match_objects(prev, curr)
            matched_curr = set(matches.values())
            prev_to_track = {}
            for track_id, track in tracks.items():
                if track["end"] == t - 1:
                    for pi, ci in matches.items():
                        if ci not in matched_curr:
                            continue
            for pi, ci in matches.items():
                prev_type = prev[pi]["type"]
                for tid, track in tracks.items():
                    if track["end"] == t - 1 and track["type"] == prev_type and tid not in prev_to_track.values():
                        last_pos = track["positions"][t - 1]
                        if np.linalg.norm(last_pos - prev[pi]["pos"]) < 0.05:
                            prev_to_track[pi] = tid
                            break
                if pi not in prev_to_track:
                    for tid, track in tracks.items():
                        if track["end"] == t - 1 and track["type"] == prev_type and tid not in prev_to_track.values():
                            prev_to_track[pi] = tid
                            break
            for pi, ci in matches.items():
                tid = prev_to_track.get(pi)
                if tid is not None:
                    tracks[tid]["end"] = t
                    tracks[tid]["positions"][t] = curr[ci]["pos"]
            for ci, obj in enumerate(curr):
                if ci not in matched_curr:
                    tracks[next_track_id] = {"start": t, "end": t, "type": obj["type"],
                                              "positions": {t: obj["pos"]}}
                    next_track_id += 1

    track_list = sorted(tracks.items(), key=lambda x: x[1]["start"])[:MAX_OBJECTS]
    Z = np.zeros((H, MAX_OBJECTS, SLOT_DIM), dtype=np.float32)

    for slot_idx, (track_id, track) in enumerate(track_list):
        ttype = track["type"]
        type_idx = TYPE_INDEX.get(ttype, 0)
        for t in range(H):
            if t in track["positions"] and track["start"] <= t <= track["end"]:
                pos = track["positions"][t]
                Z[t, slot_idx, type_idx] = 1.0
                Z[t, slot_idx, 5] = pos[0]
                Z[t, slot_idx, 6] = pos[1]
                Z[t, slot_idx, 15] = 1.0
                if t > track["start"] and (t - 1) in track["positions"]:
                    prev_pos = track["positions"][t - 1]
                    Z[t, slot_idx, 7] = pos[0] - prev_pos[0]
                    Z[t, slot_idx, 8] = pos[1] - prev_pos[1]
                if ttype == "agent":
                    Z[t, slot_idx, 0] = 1.0
                    Z[t, slot_idx, 1:5] = 0.0
                elif ttype == "key":
                    Z[t, slot_idx, 1] = 1.0
                elif ttype == "door":
                    Z[t, slot_idx, 2] = 1.0
                elif ttype == "box":
                    Z[t, slot_idx, 3] = 1.0
                elif ttype == "occluder":
                    Z[t, slot_idx, 4] = 1.0

    return Z
