# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bake Kimodo-generated motions (SOMA-77 NPZ) onto a Mixamo-rig character GLB.

Usage:
  python retarget/kimodo_to_mixamo.py --motion clip1.npz [clip2.npz ...] --character char.glb --out char_kimodo.glb
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from glb import add_animation, build_target_skeleton, detect_prefix, load_gltf, save_gltf  # noqa: E402
from mapping import SOMA_TO_MIXAMO, resolve_target_name  # noqa: E402
from retarget import retarget  # noqa: E402


def load_kimodo_npz(path: str) -> dict:
    """Load a Kimodo NPZ and normalize batch dims. Returns dict of arrays."""
    z = np.load(path)
    keys = set(z.files)
    required = ["posed_joints", "global_rot_mats"]
    missing = [k for k in required if k not in keys]
    if missing:
        raise KeyError(f"{path} is not a Kimodo NPZ; missing keys: {missing} (found: {sorted(keys)})")

    def squeeze(x):
        x = np.asarray(z[x], dtype=np.float64)
        while x.ndim > 3 and x.shape[0] == 1:  # (1, T, J, 3)
            x = x[0]
        return x

    posed = squeeze("posed_joints")
    grots = squeeze("global_rot_mats")
    if posed.ndim == 4 and posed.shape[-1] == 3 and posed.shape[-2] != 3:
        posed = posed[0]  # (1, T, J, 3) case
    while grots.ndim > 4 and grots.shape[0] == 1:
        grots = grots[0]
    T_p, J_p = posed.shape[0], posed.shape[1]
    T_r = grots.shape[0]
    if T_p != T_r:
        raise ValueError(f"posed_joints length {T_p} != global_rot_mats length {T_r}")

    contacts = None
    if "foot_contacts" in keys:
        contacts = np.asarray(z["foot_contacts"], dtype=np.float64)
        while contacts.ndim > 2 and contacts.shape[0] == 1:
            contacts = contacts[0]
        if contacts.shape[0] != T_p:
            contacts = None

    if J_p != 77:
        raise ValueError(
            f"This converter expects the SOMA-77 NPZ (77 joints), got {J_p}. "
            "Use `kimodo_convert` to convert other formats to Kimodo NPZ first."
        )
    return {
        "posed_joints": posed,
        "global_rot_mats": grots,
        "foot_contacts": contacts,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--motion", nargs="+", help="Kimodo NPZ file(s) (SOMA-77)")
    ap.add_argument("--character", required=True, help="Mixamo-rigged character .glb")
    ap.add_argument("--out", help="output .glb with appended animation clips")
    ap.add_argument("--fps", type=float, default=30.0, help="motion sampling rate (Kimodo default 30)")
    ap.add_argument("--mode", choices=["inplace", "travel"], default="inplace",
                    help="inplace: hips translation is relative (game logic moves the character); "
                         "travel: absolute position carried by the root")
    ap.add_argument("--no-limb-fix", action="store_true", help="disable arm/leg length correction")
    ap.add_argument("--no-sticky-feet", action="store_true", help="disable foot-contact IK lock")
    ap.add_argument("--clip-prefix", default="kimodo_", help="prefix for animation clip names")
    ap.add_argument("--check", action="store_true", help="print mapping diagnostics and exit")
    args = ap.parse_args()

    gltf = load_gltf(args.character)
    sk, warns = build_target_skeleton(gltf)
    prefix = detect_prefix(gltf)

    mapped, missing_tgt = [], []
    for soma_name, mix_name in SOMA_TO_MIXAMO.items():
        if resolve_target_name(mix_name, {n.lower(): n for n in sk.names}, prefix):
            mapped.append(soma_name)
        else:
            missing_tgt.append(f"{soma_name} -> {prefix}{mix_name}")

    if args.check:
        print(f"prefix: '{prefix}'")
        print(f"nodes: {len(sk.names)}, mapped joints: {len(mapped)}/77")
        print("unmapped target bones (no Mixamo counterpart in file):")
        for m in missing_tgt:
            print("  -", m)
        return

    if not args.motion or not args.out:
        ap.error("--motion and --out are required unless --check is used")

    for w in warns:
        print("WARN:", w)

    gltf_out = load_gltf(args.character)
    for motion_path in args.motion:
        data = load_kimodo_npz(motion_path)
        result = retarget(
            data["global_rot_mats"],
            data["posed_joints"],
            data["foot_contacts"],
            sk,
            prefix=prefix,
            mode=args.mode,
            limb_fix=not args.no_limb_fix,
            sticky_feet=not args.no_sticky_feet,
            fps=args.fps,
        )
        clip_name = args.clip_prefix + Path(motion_path).stem
        add_animation(gltf_out, result, clip_name)
        n_frames = data["posed_joints"].shape[0]
        print(f"baked '{clip_name}': {n_frames} frames @ {args.fps}fps "
              f"({n_frames / args.fps:.1f}s), {len(result.bones)} bone tracks, "
              f"foot KCs: {'on' if data['foot_contacts'] is not None and not args.no_sticky_feet else 'off'}")

    save_gltf(gltf_out, args.out)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
