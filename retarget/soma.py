# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SOMA 77-joint skeleton: canonical joint order, parent map, and BVH rest pose parsing.

The canonical order matches ``SOMASkeleton77.bone_order_names_with_parents`` in
``kimodo/skeleton/definitions.py`` — this is the row order of ``posed_joints`` /
``local_rot_mats`` / ``global_rot_mats`` in Kimodo NPZ files for SOMA models.
"""

from pathlib import Path
import numpy as np

# Canonical 77-joint SOMA order (flattened depth-first from definitions.py).
SOMA77_NAMES = [
    "Hips",
    "Spine1", "Spine2", "Chest",
    "Neck1", "Neck2", "Head", "HeadEnd",
    "Jaw", "LeftEye", "RightEye",
    "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",
    "LeftHandThumb1", "LeftHandThumb2", "LeftHandThumb3", "LeftHandThumbEnd",
    "LeftHandIndex1", "LeftHandIndex2", "LeftHandIndex3", "LeftHandIndex4", "LeftHandIndexEnd",
    "LeftHandMiddle1", "LeftHandMiddle2", "LeftHandMiddle3", "LeftHandMiddle4", "LeftHandMiddleEnd",
    "LeftHandRing1", "LeftHandRing2", "LeftHandRing3", "LeftHandRing4", "LeftHandRingEnd",
    "LeftHandPinky1", "LeftHandPinky2", "LeftHandPinky3", "LeftHandPinky4", "LeftHandPinkyEnd",
    "RightShoulder", "RightArm", "RightForeArm", "RightHand",
    "RightHandThumb1", "RightHandThumb2", "RightHandThumb3", "RightHandThumbEnd",
    "RightHandIndex1", "RightHandIndex2", "RightHandIndex3", "RightHandIndex4", "RightHandIndexEnd",
    "RightHandMiddle1", "RightHandMiddle2", "RightHandMiddle3", "RightHandMiddle4", "RightHandMiddleEnd",
    "RightHandRing1", "RightHandRing2", "RightHandRing3", "RightHandRing4", "RightHandRingEnd",
    "RightHandPinky1", "RightHandPinky2", "RightHandPinky3", "RightHandPinky4", "RightHandPinkyEnd",
    "LeftLeg", "LeftShin", "LeftFoot", "LeftToeBase", "LeftToeEnd",
    "RightLeg", "RightShin", "RightFoot", "RightToeBase", "RightToeEnd",
]

SOMA77_PARENT = {
    "Hips": None,
    "Spine1": "Hips", "Spine2": "Spine1", "Chest": "Spine2",
    "Neck1": "Chest", "Neck2": "Neck1", "Head": "Neck2", "HeadEnd": "Head",
    "Jaw": "Head", "LeftEye": "Head", "RightEye": "Head",
    "LeftShoulder": "Chest", "LeftArm": "LeftShoulder",
    "LeftForeArm": "LeftArm", "LeftHand": "LeftForeArm",
    "LeftHandThumb1": "LeftHand", "LeftHandThumb2": "LeftHandThumb1",
    "LeftHandThumb3": "LeftHandThumb2", "LeftHandThumbEnd": "LeftHandThumb3",
    "LeftHandIndex1": "LeftHand", "LeftHandIndex2": "LeftHandIndex1",
    "LeftHandIndex3": "LeftHandIndex2", "LeftHandIndex4": "LeftHandIndex3",
    "LeftHandIndexEnd": "LeftHandIndex4",
    "LeftHandMiddle1": "LeftHand", "LeftHandMiddle2": "LeftHandMiddle1",
    "LeftHandMiddle3": "LeftHandMiddle2", "LeftHandMiddle4": "LeftHandMiddle3",
    "LeftHandMiddleEnd": "LeftHandMiddle4",
    "LeftHandRing1": "LeftHand", "LeftHandRing2": "LeftHandRing1",
    "LeftHandRing3": "LeftHandRing2", "LeftHandRing4": "LeftHandRing3",
    "LeftHandRingEnd": "LeftHandRing4",
    "LeftHandPinky1": "LeftHand", "LeftHandPinky2": "LeftHandPinky1",
    "LeftHandPinky3": "LeftHandPinky2", "LeftHandPinky4": "LeftHandPinky3",
    "LeftHandPinkyEnd": "LeftHandPinky4",
    "RightShoulder": "Chest", "RightArm": "RightShoulder",
    "RightForeArm": "RightArm", "RightHand": "RightForeArm",
    "RightHandThumb1": "RightHand", "RightHandThumb2": "RightHandThumb1",
    "RightHandThumb3": "RightHandThumb2", "RightHandThumbEnd": "RightHandThumb3",
    "RightHandIndex1": "RightHand", "RightHandIndex2": "RightHandIndex1",
    "RightHandIndex3": "RightHandIndex2", "RightHandIndex4": "RightHandIndex3",
    "RightHandIndexEnd": "RightHandIndex4",
    "RightHandMiddle1": "RightHand", "RightHandMiddle2": "RightHandMiddle1",
    "RightHandMiddle3": "RightHandMiddle2", "RightHandMiddle4": "RightHandMiddle3",
    "RightHandMiddleEnd": "RightHandMiddle4",
    "RightHandRing1": "RightHand", "RightHandRing2": "RightHandRing1",
    "RightHandRing3": "RightHandRing2", "RightHandRing4": "RightHandRing3",
    "RightHandRingEnd": "RightHandRing4",
    "RightHandPinky1": "RightHand", "RightHandPinky2": "RightHandPinky1",
    "RightHandPinky3": "RightHandPinky2", "RightHandPinky4": "RightHandPinky3",
    "RightHandPinkyEnd": "RightHandPinky4",
    "LeftLeg": "Hips", "LeftShin": "LeftLeg", "LeftFoot": "LeftShin",
    "LeftToeBase": "LeftFoot", "LeftToeEnd": "LeftToeBase",
    "RightLeg": "Hips", "RightShin": "RightLeg", "RightFoot": "RightShin",
    "RightToeBase": "RightFoot", "RightToeEnd": "RightToeBase",
}

# Depth-first traversal is implied; verify list consistency (parent appears before child).
def _check() -> None:
    seen = set()
    for name in SOMA77_NAMES:
        parent = SOMA77_PARENT[name]
        if parent is not None and parent not in seen:
            raise RuntimeError(f"SOMA77 order error: parent {parent} of {name} not seen yet")
        seen.add(name)


_check()
SOMA77_INDEX = {n: i for i, n in enumerate(SOMA77_NAMES)}


def parse_bvh_rest(bvh_path: str | Path) -> dict[str, np.ndarray]:
    """Parse a BVH file and return {joint name: rest offset (3,) in meters}.

    Handles the ``Root`` wrapper node Kimodo writes; converts cm -> m.
    """
    text = Path(bvh_path).read_text()
    lines = [ln.strip() for ln in text.splitlines()]

    offsets: dict[str, np.ndarray] = {}
    stack: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith(("ROOT", "JOINT")):
            name = line.split()[1]
            stack.append(name)
        elif line.startswith("End Site"):
            # End site uses "End Site" + name of parent + "_End" marker offset
            name = stack[-1] + "__end"
            stack.append(name)
        elif line.startswith("OFFSET"):
            vals = np.array([float(v) for v in line.split()[1:4]], dtype=np.float64) / 100.0
            offsets[stack[-1]] = vals
        elif line == "}":
            if stack:
                stack.pop()
        i += 1

    # strip internal End-Site nodes and the root wrapper (offset 0) entries
    result = {name: off for name, off in offsets.items() if name in SOMA77_INDEX}
    for name, off in offsets.items():
        if name.endswith("__end"):
            base = name[: -len("__end")]
            if base in SOMA77_INDEX:
                # Absolute position tracking happens during FK; store end offsets keyed plainly
                result[f"{base}#end"] = off
    return result


def rest_dirs_global(offsets: dict[str, np.ndarray], root_wrapped_offset: np.ndarray | None = None):
    """Compute T-pose (rest) global bone direction vectors in meters.

    Returns {joint: {'dir': unit vector from parent joint, 'len': bone length}}.
    Positions follow the hierarchy; hips rest height is encoded in the BVH offset of Hips
    (which sits under the 'Root' wrapper).
    """
    pos: dict[str, np.ndarray] = {}
    dirs = {}
    # Root wrapper offset shifts everything (Kimodo BVH: Hips OFFSET 0 100 0 cm)
    for name in SOMA77_NAMES:
        parent = SOMA77_PARENT[name]
        off = offsets.get(name, np.zeros(3))
        if parent is None:
            pos[name] = off
        else:
            pos[name] = pos[parent] + off
    # End sites extend leaf joints (HeadEnd, *End) — not needed for directions of mapped bones
    for name in SOMA77_NAMES:
        parent = SOMA77_PARENT[name]
        if parent is None:
            continue
        vec = pos[name] - pos[parent]
        length = float(np.linalg.norm(vec))
        dirs[name] = {
            "dir": vec / length if length > 1e-6 else np.array([0.0, 1.0, 0.0]),
            "len": length,
        }
    return dirs, pos
