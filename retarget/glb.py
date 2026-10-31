# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GLB skeleton loading and animation baking with pygltflib.

Loads node hierarchy + bind TRS from a Mixamo-style glTF/GLB, feeds it to the retargeter,
then appends one baked Animation per clip and saves a new GLB.
"""

import numpy as np
from pygltflib import (
    Animation,
    AnimationChannel,
    AnimationChannelTarget,
    AnimationSampler,
    Accessor,
    Buffer,
    BufferView,
    GLTF2,
    Scene,
    Node,
)

from quatmath import quat_to_mat, mat_to_quat  # noqa: E402
from retarget import RetargetResult, TargetSkeleton  # noqa: E402

SCALAR, VEC3, VEC4 = "SCALAR", "VEC3", "VEC4"
FLOAT = 5126  # Accessor.ComponentType.FLOAT


# --------------------------------------------------------------------------- loading
def load_gltf(path: str) -> GLTF2:
    return GLTF2().load(path)


def detect_prefix(gltf: GLTF2) -> str:
    """Guess the bone-name prefix. Mixamo exports use 'mixamorig:'."""
    for node in gltf.nodes:
        if node.name and ":" in node.name and len(node.name.split(":", 1)[0]) <= 12:
            # assume smth like mixamorig:Hips — verify against a known mixamorig name
            pre, base = node.name.split(":", 1)
            if base.lower().startswith(("hips", "spine", "head")):
                return pre + ":"
    return ""


def build_target_skeleton(gltf: GLTF2) -> tuple[TargetSkeleton, list[str]]:
    """Order nodes topologically (parents first), extract bind TRS, compute bind globals."""
    node_parent = {i: -1 for i in range(len(gltf.nodes))}
    for i, node in enumerate(gltf.nodes):
        for c in node.children or []:
            node_parent[c] = i

    order = []
    placed = set()
    remaining = set(range(len(gltf.nodes)))
    while remaining:
        for i in sorted(remaining):
            p = node_parent[i]
            if p == -1 or p in placed:
                order.append(i)
                placed.add(i)
                remaining.discard(i)
                break
        else:
            # cycle guard
            order.extend(sorted(remaining))
            break

    n = len(gltf.nodes)
    names = [gltf.nodes[i].name or f"node_{i}" for i in order]
    parent_i = np.array([node_parent[i] for i in order], dtype=np.int64)

    bind_local_t = np.zeros((n, 3))
    bind_local_r = np.tile(np.eye(3), (n, 1, 1))
    warns = []
    for k, node in enumerate(order):
        gl_node = gltf.nodes[node]
        t, q, s = _node_trs(gl_node)
        if s is not None and not np.allclose(s, 1.0):
            warns.append(f"node '{names[k]}' has non-unit bind scale {s}")
        bind_local_t[k] = t
        bind_local_r[k] = quat_to_mat(q)

    bind_global_r = np.tile(np.eye(3), (n, 1, 1))
    bind_global_p = np.zeros((n, 3))
    for k in range(n):
        p = parent_i[k]
        if p == -1:
            bind_global_r[k] = bind_local_r[k]
            bind_global_p[k] = bind_local_t[k]
        else:
            bind_global_r[k] = bind_global_r[p] @ bind_local_r[k]
            bind_global_p[k] = bind_global_p[p] + bind_global_r[p] @ bind_local_t[k]

    node_index = {name: k for k, name in enumerate(names)}
    sk = TargetSkeleton(
        names=names,
        parent_i=parent_i,
        bind_local_t=bind_local_t,
        bind_local_r=bind_local_r,
        bind_global_r=bind_global_r,
        bind_global_p=bind_global_p,
        node_index=node_index,
    )
    return sk, warns


def _node_trs(node: Node):
    """Extract (translation, rotation quaternion xyzw, scale) from a glTF node."""
    if node.matrix is not None and len(node.matrix) == 16:
        M = np.array(node.matrix, dtype=np.float64).reshape(4, 4).T  # column-major
        t = M[:3, 3]
        R = M[:3, :3]
        scales = np.linalg.norm(R, axis=0)
        s = scales
        R_unit = R / np.clip(s, 1e-12, None)[None, :]
        q = mat_to_quat(R_unit)
        return t, q, s
    t = np.array(node.translation or [0, 0, 0], dtype=np.float64)
    q_raw = node.rotation
    q = np.array(q_raw, dtype=np.float64) if q_raw else np.array([0.0, 0.0, 0.0, 1.0])
    s = np.array(node.scale) if node.scale else None
    return t, q, s


# --------------------------------------------------------------------------- writing
def add_animation(gltf: GLTF2, result: RetargetResult, clip_name: str) -> Animation:
    """Append a baked Animation (rotation quats per bone, hips translation) to the GLTF."""
    node_by_name = {}
    for i, node in enumerate(gltf.nodes):
        if node.name:
            node_by_name.setdefault(node.name, i)

    blob_fn = getattr(gltf, "binary_blob", None)
    blob = bytearray(bytes(blob_fn() if callable(blob_fn) else blob_fn) if blob_fn is not None else b"")
    if gltf.buffers:
        buf = gltf.buffers[0]
    else:
        buf = Buffer()
        gltf.buffers.append(buf)

    anim = Animation(name=clip_name)
    T = None
    times_accessor = None

    def push_data(arr: np.ndarray, ctype: str) -> int:
        nonlocal blob
        data = np.ascontiguousarray(arr, dtype=np.float32).tobytes()
        while len(blob) % 4:
            blob.append(0)
        bv = BufferView(buffer=0, byteOffset=len(blob), byteLength=len(data))
        gltf.bufferViews.append(bv)
        bv_index = len(gltf.bufferViews) - 1
        blob.extend(data)
        acc = Accessor(bufferView=bv_index, componentType=FLOAT, count=arr.shape[0], type=ctype)
        gltf.accessors.append(acc)
        return len(gltf.accessors) - 1

    def find_node(bone: str) -> int | None:
        if bone in node_by_name:
            return node_by_name[bone]
        for name, idx in node_by_name.items():
            if name.endswith(":" + bone) or name.lower() == bone.lower():
                return idx
        return None

    for bone in result.bones:
        node_i = find_node(bone)
        if node_i is None:
            continue
        quats = result.rotations[bone]
        if T is None:
            T = quats.shape[0]
            times = np.arange(T, dtype=np.float32) / np.float32(result.fps)
            times_accessor = push_data(times[:, None], SCALAR)
        elif quats.shape[0] != T:
            raise ValueError("all track arrays must share the same frame count")

        out_acc = push_data(quats, VEC4)
        anim.samplers.append(
            AnimationSampler(input=times_accessor, output=out_acc, interpolation="LINEAR")
        )
        anim.channels.append(
            AnimationChannel(
                sampler=len(anim.samplers) - 1,
                target=AnimationChannelTarget(node=node_i, path="rotation"),
            )
        )

    hips_name, hips_path = next(iter(result.translations.items()))
    node_i = find_node(hips_name)
    if node_i is None:
        raise RuntimeError(f"could not locate hips node '{hips_name}' in GLB")
    if T is None:
        T = hips_path.shape[0]
        times = np.arange(T, dtype=np.float32) / np.float32(result.fps)
        times_accessor = push_data(times[:, None], SCALAR)
    out_acc = push_data(hips_path, VEC3)
    anim.samplers.append(
        AnimationSampler(input=times_accessor, output=out_acc, interpolation="LINEAR")
    )
    anim.channels.append(
        AnimationChannel(
            sampler=len(anim.samplers) - 1,
            target=AnimationChannelTarget(node=node_i, path="translation"),
        )
    )

    gltf.animations.append(anim)

    while len(blob) % 4:
        blob.append(0)
    setter = getattr(gltf, "set_binary_blob", None)
    if callable(setter):
        setter(bytes(blob))
    else:
        gltf.binary_blob = bytes(blob)
    buf.uri = None
    if buf.byteLength is None or buf.byteLength < len(blob):
        buf.byteLength = len(blob)
    return anim


def save_gltf(gltf: GLTF2, out_path: str) -> None:
    if not gltf.scenes:
        gltf.scenes = [Scene(nodes=list(range(len(gltf.nodes))))]
    gltf.save(out_path)
