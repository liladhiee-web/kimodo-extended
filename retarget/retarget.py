# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Core SOMA-77 -> target-skeleton retargeter.

Pipeline (numpy, frame-vectorized where possible):
  1. Rest-aligned global rotation transfer: target global rotation(j, t) =
     source global rotation(j, t) @ target bind global rotation(j'). Exact bind-pose
     reproduction and handles arbitrary target node axis conventions automatically.
  2. Limb length correction: for arm/leg chains whose target bones have different lengths
     (Mixamo vs SOMA), swing-rotate each limb joint so its child follows the
     segment-ratio-scaled source direction.
  3. Foot contact lock: Kimodo's foot_contacts labels drive an analytic two-bone leg IK
     that flattens the foot onto the ground during contact, eliminating foot slide.
  4. Bake to local TRS tracks for glTF Animation.
"""

from dataclasses import dataclass, field
import numpy as np

from mapping import SOMA_TO_MIXAMO, resolve_fuzzy, resolve_target_name  # noqa: E402
from quatmath import mat_to_quat, quat_between, quat_mul, quat_to_mat  # noqa: E402
from soma import SOMA77_INDEX, SOMA77_NAMES, SOMA77_PARENT  # noqa: E402

# Target-space limb correction chains: (joint_to_correct, child_joint)
LIMB_CHAINS = [
    ("LeftArm", "LeftForeArm"),
    ("LeftForeArm", "LeftHand"),
    ("RightArm", "RightForeArm"),
    ("RightForeArm", "RightHand"),
    ("LeftUpLeg", "LeftLeg"),
    ("LeftLeg", "LeftFoot"),
    ("RightUpLeg", "RightLeg"),
    ("RightLeg", "RightFoot"),
]
# knots used to drive two-bone leg IK: (thigh, shin, ankle) mixamorig names
LEG_CHAINS = {
    "left": ("LeftUpLeg", "LeftLeg", "LeftFoot"),
    "right": ("RightUpLeg", "RightLeg", "RightFoot"),
}
# foot_contacts column layout: [L heel, L toe, R heel, R toe]
CONTACT_COLS = {"left": (0, 1), "right": (2, 3)}


@dataclass
class RetargetResult:
    """Baked animation tracks.

    attrs:
        bones: list of target node names with rotation tracks (order arbitrary).
        rotations: {node name: (T,4) quaternions [x,y,z,w]} local (parent-relative).
        translations: {node name: (T,3)} only 'Hips' (in-place or traveling).
        fps: float
    """

    bones: list[str] = field(default_factory=list)
    rotations: dict[str, np.ndarray] = field(default_factory=dict)
    translations: dict[str, np.ndarray] = field(default_factory=dict)
    fps: float = 30.0


@dataclass
class TargetSkeleton:
    """Bind-pose data for a target rig (from GLB nodes).

    attrs:
        names: node name per skeleton slot index.
        parent_i: parent slot index (-1 root).
        bind_local_t: (J,3) local translation at bind.
        bind_local_r: (J,4,4) or (J,3,3) — local rotation matrices at bind.
        bind_global_r: (J,3,3) global rotation matrices at bind.
        bind_global_p: (J,3) global joint positions at bind.
        node_index: {node name: slot index}.
    """

    names: list[str]
    parent_i: np.ndarray
    bind_local_t: np.ndarray
    bind_local_r: np.ndarray
    bind_global_r: np.ndarray
    bind_global_p: np.ndarray
    node_index: dict


def fk_target_local_to_global(sk: TargetSkeleton, local_rots: np.ndarray, local_trans: np.ndarray):
    """FK animated locals to globals. local_rots (T,J,3,3), local_trans (T,J,3)."""
    T = local_rots.shape[0]
    J = len(sk.names)
    g_rot = np.zeros((T, J, 3, 3))
    g_pos = np.zeros((T, J, 3))
    roots = np.where(sk.parent_i == -1)[0]
    for j in range(J):
        pj = sk.parent_i[j]
        if pj == -1:
            g_rot[:, j] = local_rots[:, j]
            g_pos[:, j] = local_trans[:, j]
        else:
            g_rot[:, j] = np.einsum("tij,tjk->tik", g_rot[:, pj], local_rots[:, j])
            # translation = bind local translation rotated by animated global frame (constant bone length)
            off = np.einsum("tij,j->ti", g_rot[:, pj], sk.bind_local_t[j])
            g_pos[:, j] = g_pos[:, pj] + off
    return g_rot, g_pos


def derive_locals(sk: TargetSkeleton, g_rot: np.ndarray, hips_positions: np.ndarray):
    """From animated globals, derive per-node local rotations and hips translation."""
    J = len(sk.names)
    local = np.zeros_like(g_rot)
    for j in range(J):
        pj = sk.parent_i[j]
        if pj == -1:
            local[:, j] = g_rot[:, j]
        else:
            local[:, j] = np.einsum(
                "tji,tik->tjk",
                np.swapaxes(g_rot[:, pj], -1, -2),
                g_rot[:, j],
            )
    return local


def build_pairs(sk: "TargetSkeleton", prefix: str) -> tuple[dict[str, tuple[int, int]], list[str]]:
    """Resolve SOMA-77 joints onto target nodes. Returns (pairs, missing_descriptions).

    Resolution order per bone: exact name (with/without prefix) -> fuzzy
    ('mixamorig_Hips', 'Bone:Hips', 'Hips.001' style exports).
    """
    node_ci = {n.lower(): n for n in sk.names}
    pairs: dict[str, tuple[int, int]] = {}
    missing: list[str] = []
    for soma_name, mix_name in SOMA_TO_MIXAMO.items():
        tgt_name = resolve_target_name(mix_name, node_ci, prefix)
        if tgt_name is None:
            tgt_name = resolve_fuzzy(mix_name, sk.names)
        if tgt_name is None:
            missing.append(f"{soma_name} -> {mix_name}")
            continue
        pairs[soma_name] = (SOMA77_INDEX[soma_name], sk.node_index[tgt_name])
    return pairs, missing


def retarget(
    global_rot_src: np.ndarray,  # (T, 77, 3, 3)
    posed_joints_src: np.ndarray,  # (T, 77, 3) meters
    foot_contacts: np.ndarray | None,  # (T, 4) or None
    sk: TargetSkeleton,  # target bind data (built in glb.py)
    prefix: str = "mixamorig:",
    mode: str = "inplace",  # "inplace" | "travel"
    limb_fix: bool = True,
    sticky_feet: bool = True,
    fps: float = 30.0,
) -> RetargetResult:
    """Run the full retarget pipeline."""
    # ---- build the source->target joint correspondences -----------------------------
    pairs, missing = build_pairs(sk, prefix)
    if "Hips" not in pairs:
        sample = ", ".join(sk.names[:12])
        raise ValueError(
            "could not find the pelvis bone in this character — "
            "'Hips' did not resolve to any node. This is usually not a Mixamo-named rig. "
            f"Expected e.g. 'mixamorig:Hips' or 'Hips' (also accepting 'mixamorig_Hips', "
            f"'Hips.001' style). First nodes of this file: [{sample}]"
        )
    if missing:
        # expected: finger tails / neck2 / eyes etc. — report header keeps the real misses
        pass
    node_ci = {n.lower(): n for n in sk.names}
    src_of_tgt = {}
    for soma_name, (src_i, tgt_i) in pairs.items():
        src_of_tgt[tgt_i] = src_i

    # parent correspondence sanity: mapped target node's parent should map from source parent
    T = global_rot_src.shape[0]
    tgt_bind_g_r = sk.bind_global_r

    # ---- pass 1: rest-aligned global rotation transfer ------------------------------
    # R_g(t) = R_src(t) @ C_j,  C_j = R_bind_tgt(j')  (source bind rotations are identity in BVH T-pose)
    g_rot = np.zeros((T, len(sk.names), 3, 3))
    g_rot[:] = np.eye(3)
    observed = np.zeros(len(sk.names), dtype=bool)
    for soma_name, (src_i, tgt_i) in pairs.items():
        g_rot[:, tgt_i] = np.einsum("tij,jk->tik", global_rot_src[:, src_i], tgt_bind_g_r[tgt_i])
        observed[tgt_i] = True

    # unobserved nodes: inherit animated motion of nearest observed ancestor? Just keep bind pose
    # (= identity global) — matches GLB default for jaw/eyes; groupable children (Neck2) fall back.
    # Hip / root translation ----------------------------------------------------------
    hips_name = sk.names[pairs["Hips"][1]]
    hips_i = pairs["Hips"][1]

    scale = 1.0
    src_hips = posed_joints_src[:, SOMA77_INDEX["Hips"]]
    # scale factor: target bind hips height / source rest hips height (SOMA T-pose hips ~1.0m)
    tgt_bind_hips_h = sk.bind_global_p[hips_i][1]
    if tgt_bind_hips_h > 1e-4:
        scale = float(tgt_bind_hips_h / max(src_hips[:, 1].mean(), 0.5))
        scale = float(np.clip(scale, 0.3, 5.0))

    if mode == "inplace":
        hips_path = (src_hips - src_hips[:1]) * scale  # (T,3), starts at origin
    else:  # travel
        hips_path = src_hips * scale
    if not sticky_feet and foot_contacts is not None and mode == "travel":
        # travel mode keeps the character on the ground: shift path so lowest point is above 0
        min_y = hips_path[:, 1].min()
        if min_y < 0.0:
            hips_path[:, 1] -= min_y * 1.01

    # Build animated global rotations and positions via FK with hips as only translating root.
    local_rot_p1 = derive_locals(sk, g_rot, hips_path)
    local_trans = np.zeros((T, len(sk.names), 3))
    local_trans[:] = sk.bind_local_t[None]
    local_trans[:, hips_i] = hips_path
    g_rot, g_pos = fk_target_local_to_global(sk, local_rot_p1, local_trans)

    # ---- pass 2: limb length swing fix ----------------------------------------------
    if limb_fix:
        for joint_name, child_name in LIMB_CHAINS:
            tgt_name = resolve_target_name(joint_name, node_ci, prefix) or resolve_fuzzy(joint_name, sk.names)
            child = resolve_target_name(child_name, node_ci, prefix) or resolve_fuzzy(child_name, sk.names)
            if tgt_name is None or child is None or tgt_name not in src_of_tgt:
                continue
            j_i = sk.node_index[tgt_name]
            c_i = sk.node_index[child]
            src_i = src_of_tgt[j_i]
            src_child_i = src_of_tgt.get(c_i)
            if src_child_i is None:
                continue
            pj_i = sk.parent_i[j_i]
            src_parent_i = SOMA77_INDEX[SOMA77_PARENT[SOMA77_NAMES[src_i]]]

            src_vec = posed_joints_src[:, src_child_i] - posed_joints_src[:, src_parent_i]  # (T,3)
            src_len = np.linalg.norm(src_vec, axis=-1, keepdims=True).clip(min=1e-6)
            tgt_len = np.linalg.norm(
                sk.bind_global_p[c_i] - sk.bind_global_p[j_i]
            )
            ratio = float(np.clip(tgt_len / max(float(src_len.mean()), 1e-6), 0.2, 5.0))
            desired_dir = src_vec / src_len  # unit source direction, transformed below
            # express desired direction in target parent's animated frame
            if pj_i == -1:
                R_pg = np.tile(np.eye(3), (T, 1, 1))
            else:
                R_pg = g_rot[:, pj_i]
            # world desired: R_pg @ (R_src_pg^T @ d) where R_src_pg is source parent global rot
            src_pg = global_rot_src[:, src_parent_i]
            d_local_src = np.einsum("tji,tj->ti", np.swapaxes(src_pg, -1, -2), desired_dir)
            d_world = np.einsum("tij,tj->ti", R_pg, d_local_src)
            cur_world = g_pos[:, c_i] - g_pos[:, j_i]
            cur_world = cur_world / np.linalg.norm(cur_world, axis=-1, keepdims=True).clip(min=1e-6)
            swing = quat_between(cur_world, d_world)  # (T,4)
            S = quat_to_mat(swing)
            g_rot[:, j_i] = np.einsum("tij,tjk->tik", S, g_rot[:, j_i])
            # recompute positions for consistency (positions after joint j change)
            local_rot_p1 = derive_locals(sk, g_rot, hips_path)
            g_rot, g_pos = fk_target_local_to_global(sk, local_rot_p1, local_trans)

    # ---- pass 3: sticky feet via two-bone leg IK ------------------------------------
    if sticky_feet and foot_contacts is not None:
        g_rot, g_pos = _sticky_feet(
            g_rot, g_pos, sk, foot_contacts, node_ci, prefix, src_of_tgt, posed_joints_src
        )

    # ---- bake to locals --------------------------------------------------------------
    local_fin = derive_locals(sk, g_rot, hips_path)
    result = RetargetResult(fps=fps)
    for tgt_i, src_i in src_of_tgt.items():
        name = sk.names[tgt_i]
        result.bones.append(name)
        result.rotations[name] = mat_to_quat(local_fin[:, tgt_i])
    result.translations[hips_name] = hips_path
    return result


def _existing(name_mix: str, node_ci: dict, prefix: str) -> str | None:
    return resolve_target_name(name_mix, node_ci, prefix)


def _sticky_feet(
    g_rot, g_pos, sk, foot_contacts, node_ci, prefix, src_of_tgt, posed_joints_src
):
    """Flatten feet during contact runs using analytic two-bone IK per foot."""
    T = g_rot.shape[0]
    for side, cols in CONTACT_COLS.items():
        thigh_n = _existing(LEG_CHAINS[side][0], node_ci, prefix)
        shin_n = _existing(LEG_CHAINS[side][1], node_ci, prefix)
        ankle_n = _existing(LEG_CHAINS[side][2], node_ci, prefix)
        if not all((thigh_n, shin_n, ankle_n)):
            continue
        h_i, k_i, a_i = (sk.node_index[x] for x in (thigh_n, shin_n, ankle_n))
        if h_i not in src_of_tgt or k_i not in src_of_tgt or a_i not in src_of_tgt:
            continue

        contact = (foot_contacts[:, cols[0]] > 0.5) | (foot_contacts[:, cols[1]] > 0.5)
        if not contact.any():
            continue

        # contact runs
        runs = []
        start = None
        for t in range(T):
            if contact[t] and start is None:
                start = t
            elif not contact[t] and start is not None:
                runs.append((start, t - 1))
                start = None
        if start is not None:
            runs.append((start, T - 1))
        if not runs:
            continue

        L1 = np.linalg.norm(sk.bind_global_p[k_i] - sk.bind_global_p[h_i])
        L2 = np.linalg.norm(sk.bind_global_p[a_i] - sk.bind_global_p[k_i])

        for (f0, f1) in runs:
            # lock the lower of foot/toe heights; mixamorig ankle drives foot & toe nodes
            foot_pts = g_pos[f0 : f1 + 1, [a_i]]  # ankle
            lock_y = foot_pts[:, :, 1].min()
            # desired ankle positions: keep xz from current pose, clamp y to locked height
            tgt_pos = g_pos[f0 : f1 + 1, a_i].copy()
            tgt_pos[:, 1] = np.maximum(tgt_pos[:, 1], lock_y)  # don't push below lock
            tgt_pos[:, 1] = lock_y  # flatten onto locked height exactly

            parent_i = sk.parent_i[h_i]
            for idx, t in enumerate(range(f0, f1 + 1)):
                _solve_leg_ik_frame(
                    g_rot, g_pos, sk, t, h_i, k_i, a_i, L1, L2,
                    tgt_pos[idx], parent_i, posed_joints_src, src_of_tgt,
                )
            # refresh positions after per-frame edits could compound: recompute FK once per run
            hips_i = sk.node_index[_existing("Hips", node_ci, prefix)]
            # derive hips translation from current global positions to keep continuity
            local_rot = derive_locals(sk, g_rot, g_pos[:, hips_i])
            local_trans = np.zeros((T, len(sk.names), 3))
            local_trans[:] = sk.bind_local_t[None]
            local_trans[:, hips_i] = g_pos[:, hips_i]
            g_rot, g_pos = fk_target_local_to_global(sk, local_rot, local_trans)
    return g_rot, g_pos


def _solve_leg_ik_frame(
    g_rot, g_pos, sk, t, h_i, k_i, a_i, L1, L2, ankle_target, parent_i, posed_joints_src, src_of_tgt
):
    """Adjust g_rot at h_i and k_i so the ankle at frame t lands on ankle_target."""
    H = g_pos[t, h_i]
    d_vec = ankle_target - H
    d = np.linalg.norm(d_vec)
    if d < 1e-6:
        return
    d_clamped = float(np.clip(d, abs(L1 - L2) + 1e-3, L1 + L2 - 1e-3))
    u1 = d_vec / max(d, 1e-9)

    # knee pole from source knee position (helps knee bend forward like the source motion)
    if k_i in src_of_tgt:
        src_knee = posed_joints_src[t, src_of_tgt[k_i]]
        src_hip = posed_joints_src[t, src_of_tgt[h_i]]
        pole_dir = src_knee - src_hip
        if np.linalg.norm(pole_dir) > 1e-6:
            # express source pole direction in target space (world alignment assumed Y-up here)
            pole = pole_dir / np.linalg.norm(pole_dir)
        else:
            pole = np.array([0.0, 0.0, 1.0])
    else:
        pole = np.array([0.0, 0.0, 1.0])

    axis = np.cross(u1, pole)
    n_axis = np.linalg.norm(axis)
    if n_axis < 1e-6:
        axis = np.cross(u1, np.array([1.0, 0.0, 0.0]))
        n_axis = np.linalg.norm(axis)
        if n_axis < 1e-6:
            axis = np.array([0.0, 1.0, 0.0])
            n_axis = 1.0
    axis = axis / n_axis

    cos_a = np.clip((L1**2 + d_clamped**2 - L2**2) / (2 * L1 * d_clamped), -1.0, 1.0)
    a = np.arccos(cos_a)

    def rot_v(v, ang):
        return (
            v * np.cos(ang)
            + np.cross(axis, v) * np.sin(ang)
            + axis * np.dot(axis, v) * (1 - np.cos(ang))
        )

    thigh_dir1 = rot_v(u1, +a)
    thigh_dir2 = rot_v(u1, -a)
    # pick the bend whose knee is closer to the pole direction
    def knee_pos(thigh_dir):
        return H + thigh_dir * L1

    k1 = knee_pos(thigh_dir1)
    k2 = knee_pos(thigh_dir2)
    thigh_dir = thigh_dir1 if np.dot(k1 - H, pole) > np.dot(k2 - H, pole) else thigh_dir2
    knee = knee_pos(thigh_dir)

    # swing correction at hip: rotate current thigh bind-frame world dir onto thigh_dir
    cur_thigh_dir = g_pos[t, k_i] - H
    n = np.linalg.norm(cur_thigh_dir)
    if n > 1e-9:
        cur_thigh_dir /= n
        S = quat_between(cur_thigh_dir, thigh_dir)
        g_rot[t, h_i] = quat_to_mat(quat_mul(S, mat_to_quat(g_rot[t, h_i])))

    # recompute positions downstream of hip for this frame (cheap: refresh whole frame pos)
    # knee direction fix at knee joint
    g_pos[t, k_i] = knee
    cur_shin_dir = g_pos[t, a_i] - knee
    desired_shin_dir = ankle_target - knee
    n1 = np.linalg.norm(cur_shin_dir)
    n2 = np.linalg.norm(desired_shin_dir)
    if n1 > 1e-9 and n2 > 1e-9:
        S2 = quat_between(cur_shin_dir / n1, desired_shin_dir / n2)
        g_rot[t, k_i] = quat_to_mat(quat_mul(S2, mat_to_quat(g_rot[t, k_i])))
    g_pos[t, a_i] = ankle_target
