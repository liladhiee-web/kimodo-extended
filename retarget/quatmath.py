# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Numpy rotation utilities (quaternions as [x, y, z, w] 4-vectors, glTF convention)."""

import numpy as np


def mat_to_quat(R: np.ndarray) -> np.ndarray:
    """Rotation matrix (..., 3, 3) -> quaternion (..., 4) as [x, y, z, w]."""
    m = np.asarray(R)
    single = m.ndim == 2
    if single:
        m = m[None]
    batch = m.shape[0]

    trace = m[..., 0, 0] + m[..., 1, 1] + m[..., 2, 2]
    q = np.zeros((batch, 4))
    qw_positive = trace > 0.0

    s = np.sqrt(np.maximum(trace[qw_positive] + 1.0, 1e-12)) * 2.0
    q[qw_positive, 3] = 0.25 * s
    q[qw_positive, 0] = (m[qw_positive, 2, 1] - m[qw_positive, 1, 2]) / s
    q[qw_positive, 1] = (m[qw_positive, 0, 2] - m[qw_positive, 2, 0]) / s
    q[qw_positive, 2] = (m[qw_positive, 1, 0] - m[qw_positive, 0, 1]) / s

    # branch per largest diagonal
    cond0 = (~qw_positive) & (m[..., 0, 0] >= m[..., 1, 1]) & (m[..., 0, 0] >= m[..., 2, 2])
    s0 = np.sqrt(np.maximum(1.0 + m[cond0, 0, 0] - m[cond0, 1, 1] - m[cond0, 2, 2], 1e-12)) * 2.0
    q[cond0, 3] = (m[cond0, 2, 1] - m[cond0, 1, 2]) / s0
    q[cond0, 0] = 0.25 * s0
    q[cond0, 1] = (m[cond0, 0, 1] + m[cond0, 1, 0]) / s0
    q[cond0, 2] = (m[cond0, 0, 2] + m[cond0, 2, 0]) / s0

    cond1 = (~qw_positive) & (~cond0) & (m[..., 1, 1] >= m[..., 2, 2])
    s1 = np.sqrt(np.maximum(1.0 + m[cond1, 1, 1] - m[cond1, 0, 0] - m[cond1, 2, 2], 1e-12)) * 2.0
    q[cond1, 3] = (m[cond1, 0, 2] - m[cond1, 2, 0]) / s1
    q[cond1, 0] = (m[cond1, 0, 1] + m[cond1, 1, 0]) / s1
    q[cond1, 1] = 0.25 * s1
    q[cond1, 2] = (m[cond1, 1, 2] + m[cond1, 2, 1]) / s1

    cond2 = (~qw_positive) & (~cond0) & (~cond1)
    s2 = np.sqrt(np.maximum(1.0 + m[cond2, 2, 2] - m[cond2, 0, 0] - m[cond2, 1, 1], 1e-12)) * 2.0
    q[cond2, 3] = (m[cond2, 1, 0] - m[cond2, 0, 1]) / s2
    q[cond2, 0] = (m[cond2, 0, 2] + m[cond2, 2, 0]) / s2
    q[cond2, 1] = (m[cond2, 1, 2] + m[cond2, 2, 1]) / s2
    q[cond2, 2] = 0.25 * s2

    q /= np.linalg.norm(q, axis=-1, keepdims=True).clip(min=1e-12)
    return q[0] if single else q


def quat_to_mat(q: np.ndarray) -> np.ndarray:
    """Quaternion (..., 4) [x, y, z, w] -> rotation matrix (..., 3, 3)."""
    q = np.asarray(q, dtype=np.float64)
    single = q.ndim == 1
    if single:
        q = q[None]
    q = q / np.linalg.norm(q, axis=-1, keepdims=True).clip(min=1e-12)
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    R = np.empty(q.shape[:-1] + (3, 3))
    R[..., 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    R[..., 0, 1] = 2.0 * (x * y - z * w)
    R[..., 0, 2] = 2.0 * (x * z + y * w)
    R[..., 1, 0] = 2.0 * (x * y + z * w)
    R[..., 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    R[..., 1, 2] = 2.0 * (y * z - x * w)
    R[..., 2, 0] = 2.0 * (x * z - y * w)
    R[..., 2, 1] = 2.0 * (y * z + x * w)
    R[..., 2, 2] = 1.0 - 2.0 * (x * x + y * y)
    return R[0] if single else R


def quat_mul(qa: np.ndarray, qb: np.ndarray) -> np.ndarray:
    """Hamilton product; supports broadcasting of (..., 4) quaternions [x, y, z, w]."""
    ax, ay, az, aw = np.moveaxis(np.asarray(qa, dtype=np.float64), -1, 0)
    bx, by, bz, bw = np.moveaxis(np.asarray(qb, dtype=np.float64), -1, 0)
    out = np.stack(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        axis=-1,
    )
    return out / np.linalg.norm(out, axis=-1, keepdims=True).clip(min=1e-12)


def quat_conj(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    return q * np.array([1.0, -1.0, -1.0, -1.0])


def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector(s) v (..., 3) by quaternion(s) q (..., 4) [x, y, z, w]."""
    q = np.asarray(q, dtype=np.float64)
    R = quat_to_mat(q)
    return np.einsum(... if False else "...ij,...j->...i", R, np.asarray(v, dtype=np.float64))


def quat_between(va: np.ndarray, vb: np.ndarray) -> np.ndarray:
    """Shortest-arc quaternion rotating unit vector(s) va onto vb. [..., 3] -> [..., 4]."""
    va = np.asarray(va, dtype=np.float64)
    vb = np.asarray(vb, dtype=np.float64)
    va = va / np.linalg.norm(va, axis=-1, keepdims=True).clip(min=1e-12)
    vb = vb / np.linalg.norm(vb, axis=-1, keepdims=True).clip(min=1e-12)

    dot = np.clip(np.sum(va * vb, axis=-1), -1.0, 1.0)
    axis = np.cross(va, vb)
    axis_norm = np.linalg.norm(axis, axis=-1)

    q = np.empty(va.shape[:-1] + (4,))
    q[..., 3] = 1.0 + dot

    # near-opposite case: pick an orthogonal rotation axis
    near_opposite = (dot < -0.999) & (axis_norm < 1e-6)
    if near_opposite.any():
        fallback = np.zeros_like(va)
        ref = np.array([1.0, 0.0, 0.0])
        for i in range(va.shape[0]):
            if near_opposite[i]:
                a = ref if abs(va[i, 0]) < 0.9 else np.array([0.0, 1.0, 0.0])
                fallback[i] = np.cross(va[i], a)
        axis = np.where(np.broadcast_to(near_opposite[..., None], axis.shape), fallback, axis)
        axis_norm = np.linalg.norm(axis, axis=-1)
        # use 180-degree rotation on that axis
        q[near_opposite, 3] = 0.0

    good = ~near_opposite
    q[..., :3] = axis
    q = q / np.linalg.norm(q, axis=-1, keepdims=True).clip(min=1e-12)
    # sign convention: w positive where possible (q[...,3]=1+dot >= 0 on good frames)
    return q * np.where(q[..., 3:4] >= 0, 1.0, -1.0)
