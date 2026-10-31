# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end self test for the Kimodo->Mixamo retargeter.

Builds a synthetic SOMA-77 motion procedurally (gait cycle) using the repo's BVH rest offsets,
builds a synthetic 'Mixamo-like' GLB skeleton, runs the full pipeline, saves a GLB, reloads
it, and asserts:
  1. frame 0 reproduces the target bind pose exactly,
  2. the output GLB round-trips with correct channel counts and unit quaternions,
  3. foot-contact frames produce sticky (height-locked) ankles,
  4. no large per-frame rotation discontinuities.
Run:  python retarget/run_selftest.py
"""

import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from glb import add_animation, build_target_skeleton, detect_prefix, load_gltf, save_gltf  # noqa: E402
from retarget import derive_locals, fk_target_local_to_global, retarget  # noqa: E402
from quatmath import quat_to_mat  # noqa: E402
from soma import SOMA77_INDEX, SOMA77_PARENT, parse_bvh_rest  # noqa: E402

from pygltflib import GLTF2, Node, Scene  # noqa: E402


def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


def rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def make_synthetic_soma_motion(bvh_path: str, T: int = 90, fps: float = 30.0):
    """Gait-like procedural motion on the SOMA-77 hierarchy (meters, Y-up)."""
    offsets = parse_bvh_rest(bvh_path)
    missing = [n for n in SOMA77_INDEX if n not in offsets]
    if missing:
        raise RuntimeError(f"BVH missing joints: {missing[:5]}...")

    period = fps * 0.8  # stride period in frames
    loc = {n: np.eye(3) for n in SOMA77_INDEX}
    g_rot = np.zeros((T, 77, 3, 3))
    g_pos = np.zeros((T, 77, 3))

    for t in range(T):
        ph = 2 * np.pi * (t % period) / period
        sw_phase = (t % period) / period

        loc = {n: np.eye(3) for n in SOMA77_INDEX}
        hips_pos = np.array([
            (t / period) * 0.8,
            max(0.5, 1.0 + 0.01 * np.sin(ph)),
            0.0,
        ])

        def setr(name, ax, ang):
            loc[name] = rot_x(ang) if ax == "x" else rot_z(ang)

        # left leg: planted for sw_phase in [0.5, 1), swinging in [0, 0.5)
        if sw_phase < 0.5:
            a = 2 * np.pi * sw_phase
            setr("LeftLeg", "x", -0.35 * np.sin(a))
            setr("LeftShin", "x", -0.5 * max(0.0, np.sin(a)))
            setr("LeftFoot", "x", 0.35 * np.sin(a) + 0.5 * max(0.0, np.sin(a)))
        # right leg: opposite phase
        if sw_phase >= 0.5:
            a = 2 * np.pi * (sw_phase - 0.5)
            setr("RightLeg", "x", -0.35 * np.sin(a))
            setr("RightShin", "x", -0.5 * max(0.0, np.sin(a)))
            setr("RightFoot", "x", 0.35 * np.sin(a) + 0.5 * max(0.0, np.sin(a)))

        # arm swing opposite to same-side leg, slight spine twist
        setr("LeftArm", "x", 0.25 * np.sin(ph))
        setr("RightArm", "x", -0.25 * np.sin(ph))
        setr("Spine1", "z", 0.06 * np.sin(2 * ph))

        # FK
        for name in SOMA77_INDEX:
            parent = SOMA77_PARENT[name]
            j = SOMA77_INDEX[name]
            if parent is None:
                g_rot[t, j] = loc[name]
                g_pos[t, j] = hips_pos
            else:
                pi = SOMA77_INDEX[parent]
                g_rot[t, j] = g_rot[t, pi] @ loc[name]
                g_pos[t, j] = g_pos[t, pi] + g_rot[t, pi] @ offsets[name]

    # foot contacts: [L heel, L toe, R heel, R toe] — planted leg in contact
    contacts = np.zeros((T, 4))
    for t in range(T):
        sw_phase = (t % period) / period
        if sw_phase >= 0.5:
            contacts[t, 0] = contacts[t, 1] = 1.0
        else:
            contacts[t, 2] = contacts[t, 3] = 1.0

    return g_rot, g_pos, contacts


def _mixamo_hierarchy() -> dict[int, list[int]]:
    H, SP, SP1, SP2, NE, HE = 0, 1, 2, 3, 4, 5
    LSH, LAR, LFA, LHA = 6, 7, 8, 9
    LTH = [10, 11, 12]
    LIX = [13, 14, 15]
    LMD = [16, 17, 18]
    LRG = [19, 20, 21]
    LPK = [22, 23, 24]
    RSH, RAR, RFA, RHA = 25, 26, 27, 28
    RTH = [29, 30, 31]
    RIX = [32, 33, 34]
    RMD = [35, 36, 37]
    RRG = [38, 39, 40]
    RPK = [41, 42, 43]
    LUL, LLG, LFT, LTB, LTE = 44, 45, 46, 47, 48
    RUL, RLG, RFT, RTB, RTE = 49, 50, 51, 52, 53

    h: dict[int, list[int]] = {}
    h[H] = [SP, LUL, RUL]
    h[SP] = [SP1]
    h[SP1] = [SP2]
    h[SP2] = [NE, LSH, RSH]
    h[NE] = [HE]
    h[LAR] = [LFA]
    h[LFA] = [LHA]
    h[LHA] = LTH + LIX + LMD + LRG + LPK
    h[RAR] = [RFA]
    h[RFA] = [RHA]
    h[RHA] = RTH + RIX + RMD + RRG + RPK
    h[LUL] = [LLG]
    h[LLG] = [LFT]
    h[LFT] = [LTB]
    h[LTB] = [LTE]
    h[RUL] = [RLG]
    h[RLG] = [RFT]
    h[RFT] = [RTB]
    h[RTB] = [RTE]
    return h


NODE_NAMES = (
    ["Hips", "Spine", "Spine1", "Spine2", "Neck", "Head"]
    + [f"Left Shoulder".replace(" ", "")]  # placeholder fixed below
    if False
    else [
        "Hips", "Spine", "Spine1", "Spine2", "Neck", "Head",
        "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",
        "LeftHandThumb1", "LeftHandThumb2", "LeftHandThumb3",
        "LeftHandIndex1", "LeftHandIndex2", "LeftHandIndex3",
        "LeftHandMiddle1", "LeftHandMiddle2", "LeftHandMiddle3",
        "LeftHandRing1", "LeftHandRing2", "LeftHandRing3",
        "LeftHandPinky1", "LeftHandPinky2", "LeftHandPinky3",
        "RightShoulder", "RightArm", "RightForeArm", "RightHand",
        "RightHandThumb1", "RightHandThumb2", "RightHandThumb3",
        "RightHandIndex1", "RightHandIndex2", "RightHandIndex3",
        "RightHandMiddle1", "RightHandMiddle2", "RightHandMiddle3",
        "RightHandRing1", "RightHandRing2", "RightHandRing3",
        "RightHandPinky1", "RightHandPinky2", "RightHandPinky3",
        "LeftUpLeg", "LeftLeg", "LeftFoot", "LeftToeBase", "LeftToe_End",
        "RightUpLeg", "RightLeg", "RightFoot", "RightToeBase", "RightToe_End",
    ]
)

NODE_TRANSLATIONS = {
    "Hips": (0, 0.95, 0),
    "Spine": (0, 0.08, 0),
    "Spine1": (0, 0.10, 0),
    "Spine2": (0, 0.12, 0),
    "Neck": (0, 0.16, 0),
    "Head": (0, 0.10, 0),
    "LeftShoulder": (0.04, 0.12, 0),
    "LeftArm": (0.11, 0.02, 0),
    "LeftForeArm": (0.27, 0, 0),
    "LeftHand": (0.25, 0, 0),
    "LeftHandThumb1": (0.03, 0.01, 0.02),
    "LeftHandThumb2": (0.03, 0, 0.01),
    "LeftHandThumb3": (0.025, 0, 0.01),
    "LeftHandIndex1": (0.08, 0, 0.02),
    "LeftHandIndex2": (0.04, 0, 0),
    "LeftHandIndex3": (0.035, 0, 0),
    "LeftHandMiddle1": (0.085, 0, 0.005),
    "LeftHandMiddle2": (0.043, 0, 0),
    "LeftHandMiddle3": (0.038, 0, 0),
    "LeftHandRing1": (0.08, 0, -0.01),
    "LeftHandRing2": (0.038, 0, 0),
    "LeftHandRing3": (0.034, 0, 0),
    "LeftHandPinky1": (0.07, 0, -0.025),
    "LeftHandPinky2": (0.032, 0, 0),
    "LeftHandPinky3": (0.028, 0, 0),
    "RightShoulder": (-0.04, 0.12, 0),
    "RightArm": (-0.11, 0.02, 0),
    "RightForeArm": (-0.27, 0, 0),
    "RightHand": (-0.25, 0, 0),
    "RightHandThumb1": (-0.03, 0.01, 0.02),
    "RightHandThumb2": (-0.03, 0, 0.01),
    "RightHandThumb3": (-0.025, 0, 0.01),
    "RightHandIndex1": (-0.08, 0, 0.02),
    "RightHandIndex2": (-0.04, 0, 0),
    "RightHandIndex3": (-0.035, 0, 0),
    "RightHandMiddle1": (-0.085, 0, 0.005),
    "RightHandMiddle2": (-0.043, 0, 0),
    "RightHandMiddle3": (-0.038, 0, 0),
    "RightHandRing1": (-0.08, 0, -0.01),
    "RightHandRing2": (-0.038, 0, 0),
    "RightHandRing3": (-0.034, 0, 0),
    "RightHandPinky1": (-0.07, 0, -0.025),
    "RightHandPinky2": (-0.032, 0, 0),
    "RightHandPinky3": (-0.028, 0, 0),
    "LeftUpLeg": (-0.09, -0.04, 0),
    "LeftLeg": (0, -0.42, 0),
    "LeftFoot": (0, -0.40, 0),
    "LeftToeBase": (0, -0.06, 0.09),
    "LeftToe_End": (0, -0.01, 0.08),
    "RightUpLeg": (0.09, -0.04, 0),
    "RightLeg": (0, -0.42, 0),
    "RightFoot": (0, -0.40, 0),
    "RightToeBase": (0, -0.06, 0.09),
    "RightToe_End": (0, -0.01, 0.08),
}


def make_synthetic_mixamo_glb(tmp_dir: Path) -> Path:
    """Minimal Mixamo-like GLB: prefix nodes, T-pose, meters, Y-up."""
    prefix = "mixamorig:"
    nodes = [
        Node(
            name=prefix + name,
            translation=[float(c) for c in NODE_TRANSLATIONS[name]],
        )
        for name in NODE_NAMES
    ]
    for p, children in _mixamo_hierarchy().items():
        for c in children:
            nodes[p].children.append(c)
    for nd in nodes:
        if nd.children is not None and len(nd.children) == 0:
            nd.children = None

    gltf = GLTF2()
    gltf.nodes = nodes
    gltf.asset.version = "2.0"
    gltf.scenes = [Scene(nodes=[0])]
    gltf.scene = 0
    out = Path(tmp_dir) / "synthetic_char.glb"
    gltf.save(str(out))
    return out


def main():
    repo_root = Path(__file__).resolve().parent.parent
    bvh = repo_root / "kimodo" / "assets" / "skeletons" / "somaskel77" / "somaskel77_standard_tpose.bvh"
    assert bvh.exists(), f"BVH not found: {bvh}"

    T, fps = 90, 30.0
    g_rot_src, g_pos_src, contacts = make_synthetic_soma_motion(str(bvh), T=T, fps=fps)

    with tempfile.TemporaryDirectory() as td:
        char_path = make_synthetic_mixamo_glb(td)

        gltf = load_gltf(str(char_path))
        prefix = detect_prefix(gltf)
        sk, warns = build_target_skeleton(gltf)
        for w in warns:
            print("WARN:", w)
        assert prefix == "mixamorig:", f"prefix detect failed: {prefix!r}"

        # Pure transfer first (no limb fix): must reproduce bind pose at t=0 exactly.
        result_plain = retarget(
            g_rot_src, g_pos_src, contacts, sk,
            prefix=prefix, mode="inplace", limb_fix=False, sticky_feet=False, fps=fps,
        )
        max_q_err = 0.0
        for bone, quats in result_plain.rotations.items():
            j = sk.node_index[bone]
            R_from_q = quat_to_mat(quats[0])
            err = np.abs(R_from_q - sk.bind_local_r[j]).max()
            max_q_err = max(max_q_err, err)
        assert max_q_err < 1e-6, f"bind pose mismatch at t=0: {max_q_err}"
        hips_name0 = next(iter(result_plain.translations))
        assert float(np.abs(result_plain.translations[hips_name0][0]).max()) < 1e-12

        # Full pipeline (limb fix + sticky feet) for the functional checks.
        result = retarget(
            g_rot_src, g_pos_src, contacts, sk,
            prefix=prefix, mode="inplace", limb_fix=True, sticky_feet=True, fps=fps,
        )

        # --- 2. save + reload roundtrip
        out_path = Path(td) / "out.glb"
        gltf_out = load_gltf(str(char_path))
        add_animation(gltf_out, result, "test")
        save_gltf(gltf_out, str(out_path))
        re_gltf = GLTF2().load(str(out_path))
        assert len(re_gltf.animations) == 1
        anim = re_gltf.animations[0]
        rot_paths = [c for c in anim.channels if c.target.path == "rotation"]
        trans_paths = [c for c in anim.channels if c.target.path == "translation"]
        assert len(trans_paths) == 1
        assert len(rot_paths) >= 40, f"too few rotation channels: {len(rot_paths)}"
        out_acc = anim.samplers[rot_paths[0].sampler].output
        acc = re_gltf.accessors[out_acc]
        bv = re_gltf.bufferViews[acc.bufferView]
        blob_fn = getattr(re_gltf, "binary_blob", None)
        blob_bytes = blob_fn() if callable(blob_fn) else blob_fn
        raw = np.frombuffer(
            blob_bytes[bv.byteOffset : bv.byteOffset + bv.byteLength], dtype=np.float32
        ).reshape(-1, 4)
        norms = np.linalg.norm(raw.astype(np.float64), axis=1)
        assert np.abs(norms - 1.0).max() < 1e-3, f"quat normalization drift: {norms.min()}, {norms.max()}"

        # --- 3. rebuild worlds from baked locals; verify sticky feet + continuity
        sk2, _ = build_target_skeleton(load_gltf(str(char_path)))
        locals_all = np.zeros((T, len(sk2.names), 3, 3))
        hips_i = sk2.node_index[hips_name0]
        hips_track = result.translations[hips_name0]
        unmapped = []
        for bone, quats in result.rotations.items():
            j = sk2.node_index.get(bone)
            if j is None:
                unmapped.append(bone)
                continue
            for q_i in range(T):
                locals_all[q_i, j] = quat_to_mat(quats[q_i])
        if unmapped:
            print("WARN tracks without node:", unmapped)
        local_trans = np.repeat(sk2.bind_local_t[None], T, axis=0)
        local_trans[:, hips_i] = hips_track
        g_rot_f, g_pos_f = fk_target_local_to_global(sk2, locals_all, local_trans)

        ankles = {
            "left": sk2.node_index.get("mixamorig:LeftFoot"),
            "right": sk2.node_index.get("mixamorig:RightFoot"),
        }

        def contact_runs(side):
            cols = {"left": (0, 1), "right": (2, 3)}[side]
            c = (contacts[:, cols[0]] > 0.5) | (contacts[:, cols[1]] > 0.5)
            runs = []
            start = None
            for t in range(T):
                if c[t] and start is None:
                    start = t
                elif not c[t] and start is not None:
                    runs.append((start, t - 1))
                    start = None
            if start is not None:
                runs.append((start, T - 1))
            return runs

        n_runs = 0
        for side, aj in ankles.items():
            assert aj is not None
            for (f0, f1) in contact_runs(side):
                if f1 - f0 >= 1:
                    ys = g_pos_f[f0 : f1 + 1, aj, 1]
                    span = float(ys.max() - ys.min())
                    assert span < 0.01, f"{side} contact run height span {span*100:.2f}cm > 1cm"
                    n_runs += 1

        worst_deg = 0.0
        for bone, quats in result.rotations.items():
            q = quats.astype(np.float64)
            d = np.clip(np.abs(np.sum(q[:-1] * q[1:], axis=1)), -1.0, 1.0)
            worst_deg = max(worst_deg, float(np.degrees(2 * np.arccos(d)).max()))
        assert worst_deg < 30, f"rotation jump {worst_deg:.1f} deg/frame"

        print("ALL SELF-TESTS PASSED")
        print(f"  bind-pose reproduction error: {max_q_err:.2e}")
        print(f"  rotation channels: {len(rot_paths)}; hips translation channel: {len(trans_paths)}")
        print(f"  sticky feet verified over {n_runs} contact run(s)")
        print(f"  largest per-frame rotation: {worst_deg:.2f} deg")


if __name__ == "__main__":
    main()
