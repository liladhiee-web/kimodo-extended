# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Parse Kimodo SOMA BVH motion files into the Kimodo NPZ dict shape — numpy only.

Kimodo BVH layout (see kimodo/assets/skeletons/.../somaskel77_standard_tpose.bvh):
  ROOT Root      { OFFSET 0 0 0; CHANNELS 6 Xpos Ypos Zpos Zrot Yrot Xrot }
    JOINT Hips   { OFFSET 0 100 0; CHANNELS 6 Xpos Ypos Zpos Zrot Yrot Xrot }
      ... joints with 3 rotation channels in "Z Y X" order, End Sites (no channels).

This parses hierarchy + per-frame channel data and runs FK in meters, producing:
  posed_joints     (T, J, 3)   global joint positions
  global_rot_mats  (T, J, 3,3) global joint rotations
  local_rot_mats   (T, J, 3,3) local rotations
  fps              float
Joint rows are mapped by joint NAME to the canonical SOMA-77 order (soma.SOMA77_NAMES).
"""

import numpy as np


def rot_axis(axis: str, angle_rad: float) -> np.ndarray:
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    if axis == "X":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == "Y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _rotation_from_channels(order: list[str], angles: np.ndarray) -> np.ndarray:
    """order maps 1:1 to angles: e.g. ['Zrotation','Yrotation','Xrotation'].

    Matches Kimodo's own loader (kimodo/skeleton/bvh.py:530) which uses
    scipy ``Rotation.from_euler("ZYX", ...)`` — uppercase order string means
    INTRINSIC rotations applied in listed order, i.e.
    R = R_first @ ... @ R_last = Rz @ Ry @ Rx for the Z→Y→X listing.
    """
    R = np.eye(3)
    for axis_name, ang in zip(order, angles):
        # intrinsic: first-listed rotation sits leftmost in the product
        R = R @ rot_axis(axis_name[0], ang)
    return R


def parse_bvh_frames(path: str, scale: float | None = None):
    """Parse a BVH file with a MOTION section.

    Returns dict with keys: posed_joints (T,J,3), global_rot_mats (T,J,3,3),
    local_rot_mats (T,J,3,3), joint_names (list of J strings), fps (float), scale (m/cm factor).
    """
    text = open(path, "rt", encoding="utf-8").read()
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    # ---- pass 1: hierarchy ------------------------------------------------------
    nodes = []  # list of dicts: name, parentIndex, offset, rot_order(list), has_pos
    stack = []
    in_motion = False
    frame_time = None
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("MOTION"):
            in_motion = True
            i += 1
            continue
        if in_motion:
            if line.startswith("Frame Time") or line.startswith("Frames:"):
                # handled later
                i += 1
                continue
            break
        if line.startswith(("ROOT", "JOINT")):
            parent_i = stack[-1] if stack else -1
            nodes.append(
                {
                    "name": line.split()[1],
                    "parent": parent_i,
                    "offset": None,
                    "order": [],
                    "has_pos": False,
                    "depth": len(stack),
                }
            )
            stack.append(len(nodes) - 1)
        elif line.startswith("End Site"):
            parent_i = stack[-1] if stack else -1
            nodes.append(
                {
                    "name": f"__end_{len(nodes)}",
                    "parent": parent_i,
                    "offset": None,
                    "order": [],
                    "has_pos": False,
                    "depth": len(stack),
                }
            )
            stack.append(len(nodes) - 1)
        elif line.startswith("OFFSET"):
            idx = stack[-1]
            nodes[idx]["offset"] = np.array([float(v) for v in line.split()[1:4]])
        elif line.startswith("CHANNELS"):
            idx = stack[-1]
            toks = line.split()[2:]
            nodes[idx]["order"] = toks
            nodes[idx]["has_pos"] = any("position" in t for t in toks)
        elif line == "}":
            if stack:
                stack.pop()
        i += 1

    # scale: BVH is cm by convention when values are sizeable; keep a heuristic
    auto_scale = None
    for n in nodes:
        if n["offset"] is not None and np.max(np.abs(n["offset"])) > 0.0001:
            if np.max(np.abs(n["offset"])) > 1.0:  # e.g., 100 == 1 m
                auto_scale = 0.01
            break
    if scale is None:
        scale = auto_scale if auto_scale is not None else 1.0

    # ---- pass 2: frames -----------------------------------------------------------
    j = i  # continue from break point
    frames: list[np.ndarray] = []
    n_ch = sum(len(n["order"]) for n in nodes)
    while j < len(lines):
        line = lines[j]
        if line.startswith("Frames:"):
            j += 1
            continue
        if line.startswith("Frame Time") or line.startswith("Frames Time"):
            parts = line.split(":")
            frame_time = float(parts[1]) if len(parts) > 1 else None
            j += 1
            continue
        vals = np.array(line.split(), dtype=np.float64)
        if vals.size == 0:
            j += 1
            continue
        if vals.size != n_ch:
            raise ValueError(
                f"BVH frame value count mismatch: expected {n_ch}, got {vals.size}"
            )
        frames.append(vals)
        j += 1

    if not frames:
        raise ValueError("BVH has no MOTION frames (rest-pose-only file?)")
    if frame_time is None:
        frame_time = 1.0 / 30.0
    fps = 1.0 / frame_time

    T = len(frames)
    J = len(nodes)
    posed = np.zeros((T, J, 3))
    g_rot = np.zeros((T, J, 3, 3))

    offsets = {k: (n["offset"] if n["offset"] is not None else np.zeros(3)) for k, n in enumerate(nodes)}

    for t in range(T):
        vals = frames[t]
        pos_i = 0
        for k, node in enumerate(nodes):
            order = node["order"]
            rot_order = [o for o in order if "rotation" in o]
            pos_vals = None
            rot_vals = None
            if order:
                seg = vals[pos_i : pos_i + len(order)]
                pos_i += len(order)
                p_names = [o for o in order if "position" in o]
                if p_names:
                    pos_vals = np.array(
                        [seg[i] for i, o in enumerate(order) if "position" in o], dtype=np.float64
                    )
                r_names = [o for o in order if "rotation" in o]
                if r_names:
                    rot_vals = np.array(
                        [seg[k2] for k2, o in enumerate(order) if "rotation" in o], dtype=np.float64
                    )
            local_t = offsets[k] * scale
            if pos_vals is not None:
                local_t = local_t + pos_vals * scale
            if rot_vals is not None:
                # BVH euler channels are in degrees (Kimodo does deg2rad in skeleton/bvh.py)
                local_r = _rotation_from_channels(rot_order, np.deg2rad(rot_vals))
            else:
                local_r = np.eye(3)
            p = node["parent"]
            if p == -1:
                posed[t, k] = local_t
                g_rot[t, k] = local_r
            else:
                posed[t, k] = posed[t, p] + g_rot[t, p] @ local_t
                g_rot[t, k] = g_rot[t, p] @ local_r

    joint_names = [n["name"] for n in nodes if not n["name"].startswith("__end_")]
    keep = [k for k, n in enumerate(nodes) if not n["name"].startswith("__end_")]
    return {
        "posed_joints": posed[:, keep],
        "global_rot_mats": g_rot[:, keep],
        "local_rot_mats": None,  # filled below
        "joint_names": joint_names,
        "fps": float(fps),
        "scale": float(scale),
        "T": T,
    }


def parse_bvh_to_kimodo_rows(path: str, soma_index: dict[str, int], soma_names: list[str]):
    """Parse BVH and remap joint rows to the SOMA-77 canonical order.

    Returns dict suited for the retargeter:
      posed_joints (T,77,3), global_rot_mats (T,77,3,3), fps, (foot_contacts = None)
    """
    from soma import SOMA77_INDEX, SOMA77_NAMES

    data = parse_bvh_frames(path)
    names = data["joint_names"]
    # rows map back by joint NAME: BVH names should be SOMA names ("Hips", "Spine1", ...)
    row_by_name = {n: i for i, n in enumerate(names)}
    T = data["T"]
    J77 = len(soma_names)
    posed77 = np.zeros((T, J77, 3))
    grot77 = np.zeros((T, J77, 3, 3))

    missing = [n for n in soma_names if n not in row_by_name]
    if missing:
        raise ValueError(f"BVH is missing these SOMA-77 joints: {missing[:8]}...")
    extra = [n for n in names if n not in soma_index]
    if extra:
        # wrapper nodes like "Root" are expected; anything else is unusual but ignorable
        print("BVH extra joints (ignored):", [e for e in extra if e != "Root"][:6] or "Root wrapper")

    seen = np.zeros(J77, dtype=bool)
    for name, row_i in row_by_name.items():
        if name not in soma_index:
            continue
        tgt = soma_index[name]
        posed77[:, tgt] = data["posed_joints"][:, row_i]
        grot77[:, tgt] = data["global_rot_mats"][:, row_i]
        seen[tgt] = True
    if not seen.all():
        raise ValueError("not all SOMA-77 rows were filled from the BVH")

    # local rotations via parent conjugation
    from soma import SOMA77_PARENT

    lrot77 = np.zeros_like(grot77)
    for name, j in SOMA77_INDEX.items():
        p = SOMA77_PARENT[name]
        if p is None:
            lrot77[:, j] = grot77[:, j]
        else:
            pi = SOMA77_INDEX[p]
            lrot77[:, j] = np.einsum("tji,tik->tjk", np.swapaxes(grot77[:, pi], -1, -2), grot77[:, j])

    return {
        "posed_joints": posed77,
        "global_rot_mats": grot77,
        "local_rot_mats": lrot77,
        "fps": data["fps"],
        "foot_contacts": None,
        "T": T,
    }
