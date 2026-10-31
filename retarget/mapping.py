# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SOMA-77 joint -> Mixamo bone mapping.

Mixamo bone node names usually carry the ``mixamorig:`` prefix in glTF/FBX exports, but some
pipelines strip it. The converter matches target nodes case-insensitively, with or without the
prefix; edit TARGET_EXTRA here or provide a JSON mapping to override at runtime.
"""

import re

# SOMA joint name -> Mixamo node name (without prefix; prefix added/removed as needed)
SOMA_TO_MIXAMO = {
    "Hips": "Hips",
    "Spine1": "Spine",
    "Spine2": "Spine1",
    "Chest": "Spine2",
    "Neck1": "Neck",
    "Head": "Head",
    "LeftShoulder": "LeftShoulder",
    "LeftArm": "LeftArm",
    "LeftForeArm": "LeftForeArm",
    "LeftHand": "LeftHand",
    "LeftLeg": "LeftUpLeg",
    "LeftShin": "LeftLeg",
    "LeftFoot": "LeftFoot",
    "LeftToeBase": "LeftToeBase",
    "LeftToeEnd": "LeftToe_End",
    "RightShoulder": "RightShoulder",
    "RightArm": "RightArm",
    "RightForeArm": "RightForeArm",
    "RightHand": "RightHand",
    "RightLeg": "RightUpLeg",
    "RightShin": "RightLeg",
    "RightFoot": "RightFoot",
    "RightToeBase": "RightToeBase",
    "RightToeEnd": "RightToe_End",
}

# Finger chains: SOMA has 4-5 phalanges per finger (Thumb x4, others x5, incl. *End tips);
# Mixamo has 3 per finger. Map the first three, drop the rest.
for side in ("Left", "Right"):
    SOMA_TO_MIXAMO[f"{side}HandThumb1"] = f"{side}HandThumb1"
    SOMA_TO_MIXAMO[f"{side}HandThumb2"] = f"{side}HandThumb2"
    SOMA_TO_MIXAMO[f"{side}HandThumb3"] = f"{side}HandThumb3"
    for finger in ("Index", "Middle", "Ring", "Pinky"):
        for i in (1, 2, 3):
            SOMA_TO_MIXAMO[f"{side}Hand{finger}{i}"] = f"{side}Hand{finger}{i}"

# SOMA joints intentionally not mapped (Mixamo rig has no counterpart):
UNMAPPED = [
    "Neck2",      # folded into Neck (Neck1 drives it)
    "HeadEnd", "Jaw", "LeftEye", "RightEye",
    "LeftHandThumbEnd", "LeftHandIndex4", "LeftHandIndexEnd",
    "LeftHandMiddle4", "LeftHandMiddleEnd", "LeftHandRing4", "LeftHandRingEnd",
    "LeftHandPinky4", "LeftHandPinkyEnd",
    "RightHandThumbEnd", "RightHandIndex4", "RightHandIndexEnd",
    "RightHandMiddle4", "RightHandMiddleEnd", "RightHandRing4", "RightHandRingEnd",
    "RightHandPinky4", "RightHandPinkyEnd",
]

# Limit chains whose target counterpart is a single node in Mixamo: rate-limit to the mapped
# parent so motion isn't lost when the child has no counterpart.
GROUPED_CHILDREN = {"Neck2": "Neck1"}


def resolve_target_name(name: str, node_names_ci: dict[str, str], prefix: str) -> str | None:
    """Find the actual target node for a Mixamo-style bone name, prefix-flexible."""
    candidates = [
        f"{prefix}{name}",
        f"{prefix}{name.lower()}",
        name,
        name.lower(),
    ]
    for c in candidates:
        hit = node_names_ci.get(c.lower())
        if hit is not None:
            return hit
    return None


# FBX/Blender dedup suffixes: "..._2", "... .001", "...:1"  (Mixamo names never end
# with "<sep><digits>"; digit-name bones like Thumb3/Spine1 have no separator before)
_DUP_SUFFIX_RE = re.compile(r"[:._|-]\d{1,4}$")


def normalize_bone_name(name: str) -> str:
    """Lowercase + strip FBX dedup suffixes, for tolerant matching."""
    return _DUP_SUFFIX_RE.sub("", (name or "").strip().lower())


def resolve_fuzzy(bone_name: str, names: list[str]) -> str | None:
    """Suffix-tolerant fallback: node may be 'mixamorig_Hips', 'Bone:Hips', 'Hips.001'...

    Matches when the normalized node name equals the bone name or ends with
    '<separator><bone>', separator in : _ | - space. Prefers the shortest name.
    """
    base = normalize_bone_name(bone_name)
    best = None
    for n in names:
        nm = normalize_bone_name(n)
        if nm == base:
            return n
        for sep in (":", "_", "|", "-", " "):
            if nm.endswith(sep + base):
                if best is None or len(n) < len(best):
                    best = n
                break
    return best


def detect_prefix(names: list[str]) -> str:
    """Guess the bone-name prefix from any node ending in a known bone word.

    Handles ':'-style (mixamorig:Hips) AND '_'-style (mixamorig_Hips) exports.
    """
    for n in names:
        if not n:
            continue
        low = n.lower()
        for bone in ("hips", "spine", "head", "leftfoot"):
            for sep in (":", "_", "|", "-", " "):
                if low.endswith(sep + bone):
                    i = low.rfind(sep + bone)
                    return n[: i + 1]
    return ""
