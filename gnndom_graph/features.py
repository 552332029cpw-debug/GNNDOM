# SPDX-License-Identifier: BSD-3-Clause
"""Feature builders for ManiFabric-style dynamics graphs in z-up coordinates."""

from __future__ import annotations

import math

import numpy as np


def build_velocity_history(
    rollout_dir,
    timestep: int,
    *,
    n_his: int,
    pred_time_interval: int,
    downsample_idx: np.ndarray,
) -> np.ndarray:
    from gnndom_dataset.storage import load_step

    pos_list = []
    for i in range(timestep - n_his * pred_time_interval, timestep + pred_time_interval, pred_time_interval):
        step = load_step(rollout_dir, max(0, i))
        pos_list.append(step["positions"].astype(np.float32))
    vel_list = []
    for i in range(n_his):
        dt_steps = max(1, min(timestep - (timestep - (n_his - i) * pred_time_interval), pred_time_interval))
        vel_list.append((pos_list[i + 1] - pos_list[i]) / np.float32(dt_steps))
    return np.concatenate([vel[downsample_idx] for vel in vel_list], axis=1).astype(np.float32)


def apply_picker_action(
    pointcloud: np.ndarray,
    velocity_his: np.ndarray,
    picker_position: np.ndarray,
    action: np.ndarray,
    *,
    dt: float,
    pred_time_interval: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pointcloud = pointcloud.astype(np.float32).copy()
    velocity_his = velocity_his.astype(np.float32).copy()
    picked_particles = np.full((2,), -1, dtype=np.int64)
    action = action.astype(np.float32).reshape(2, 4)
    picker_position = picker_position.astype(np.float32)
    for picker_idx in range(2):
        if action[picker_idx, 3] <= 0.5:
            continue
        distances = np.linalg.norm(pointcloud - picker_position[picker_idx], axis=1)
        picked = int(np.argmin(distances))
        picked_particles[picker_idx] = picked
        old_pos = pointcloud[picked].copy()
        new_pos = old_pos + action[picker_idx, :3]
        new_vel = (new_pos - old_pos) / np.float32(dt * pred_time_interval)
        if velocity_his.shape[1] >= 6:
            velocity_his[picked, :-3] = velocity_his[picked, 3:].copy()
        velocity_his[picked, -3:] = new_vel
        pointcloud[picked] = new_pos
    return pointcloud, velocity_his, picked_particles


def compute_node_attr(
    pointcloud: np.ndarray,
    velocity_his: np.ndarray,
    picked_particles: np.ndarray,
    *,
    rollout_info: dict,
    use_es: bool,
) -> np.ndarray:
    node_one_hot = np.zeros((len(pointcloud), 2), dtype=np.float32)
    node_one_hot[:, 0] = 1.0
    for picked in picked_particles:
        if picked >= 0:
            node_one_hot[picked, 0] = 0.0
            node_one_hot[picked, 1] = 1.0

    if use_es and rollout_info.get("env_shape") is not None:
        distance_to_shape, vector_to_shape = compute_distance_to_shape(
            pointcloud,
            np.asarray(rollout_info["shape_pos"], dtype=np.float32),
            np.asarray(rollout_info["shape_size"], dtype=np.float32),
            np.asarray(rollout_info["shape_quat"], dtype=np.float32),
            str(rollout_info["env_shape"]),
        )
    else:
        distance_to_shape = pointcloud[:, 2:3].astype(np.float32)
        vector_to_shape = np.zeros((len(pointcloud), 3), dtype=np.float32)
        vector_to_shape[:, 2] = 1.0

    return np.concatenate([velocity_his, distance_to_shape, vector_to_shape, node_one_hot], axis=1).astype(np.float32)


def compute_distance_to_shape(
    pointcloud: np.ndarray,
    shape_pos: np.ndarray,
    shape_size: np.ndarray,
    shape_quat: np.ndarray,
    env_shape: str,
) -> tuple[np.ndarray, np.ndarray]:
    ground_dist = pointcloud[:, 2:3].astype(np.float32)
    ground_vec = np.zeros((len(pointcloud), 3), dtype=np.float32)
    ground_vec[:, 2] = 1.0

    if env_shape in {"platform", "table"}:
        rot = quat_to_matrix(shape_quat)
        local_point = (rot.T @ (pointcloud - shape_pos).T).T
        clamped = np.maximum(np.minimum(local_point, shape_size), -shape_size)
        closest = (rot @ clamped.T).T + shape_pos
        vector = pointcloud - closest
        distance = np.linalg.norm(vector.astype(np.float32), axis=1, keepdims=True)
    elif env_shape == "sphere":
        vector = pointcloud - shape_pos
        center_dist = np.linalg.norm(vector.astype(np.float32), axis=1, keepdims=True)
        distance = center_dist - float(shape_size[0])
    elif env_shape == "rod":
        rot = quat_to_matrix(shape_quat)
        local_point = (rot.T @ (pointcloud - shape_pos).T).T
        closest_local = local_point.copy()
        closest_local[:, 0] = 0.0
        closest_local[:, 1] = np.clip(closest_local[:, 1], -float(shape_size[1]), float(shape_size[1]))
        closest_local[:, 2] = 0.0
        closest = (rot @ closest_local.T).T + shape_pos
        vector = pointcloud - closest
        distance = np.linalg.norm(vector.astype(np.float32), axis=1, keepdims=True) - float(shape_size[0])
    else:
        return ground_dist, ground_vec

    norm = np.linalg.norm(vector.astype(np.float32), axis=1, keepdims=True)
    vector_norm = np.divide(vector, np.maximum(norm, 1.0e-8)).astype(np.float32)
    use_shape = distance <= ground_dist
    return np.where(use_shape, distance, ground_dist).astype(np.float32), np.where(use_shape, vector_norm, ground_vec).astype(np.float32)


def compute_edge_attr(
    pointcloud: np.ndarray,
    *,
    cloth_xdim: int,
    cloth_ydim: int,
    neighbor_radius: float,
    use_mesh_edge: bool,
) -> tuple[np.ndarray, np.ndarray]:
    distance_edges = radius_edges(pointcloud, neighbor_radius)
    mesh_edges = eight_neighbor_edges(cloth_xdim, cloth_ydim) if use_mesh_edge else np.empty((2, 0), dtype=np.int64)

    edges_blocks = []
    attr_blocks = []
    if distance_edges.shape[1] > 0:
        edges_blocks.append(distance_edges)
        attr_blocks.append(edge_features(pointcloud, distance_edges, edge_type=(1.0, 0.0)))
    if mesh_edges.shape[1] > 0:
        edges_blocks.append(mesh_edges)
        attr_blocks.append(edge_features(pointcloud, mesh_edges, edge_type=(0.0, 1.0)))
    if not edges_blocks:
        edges = np.asarray([[0, 1], [1, 0]], dtype=np.int64)
        return edges, edge_features(pointcloud, edges, edge_type=(0.0, 0.0))
    return np.concatenate(edges_blocks, axis=1).astype(np.int64), np.concatenate(attr_blocks, axis=0).astype(np.float32)


def radius_edges(pointcloud: np.ndarray, radius: float) -> np.ndarray:
    try:
        from scipy.spatial import cKDTree

        pairs = np.asarray(list(cKDTree(pointcloud).query_pairs(float(radius), p=2)), dtype=np.int64)
        if pairs.size == 0:
            return np.empty((2, 0), dtype=np.int64)
        undirected = pairs.T
    except Exception:
        diff = pointcloud[:, None, :] - pointcloud[None, :, :]
        dist = np.linalg.norm(diff, axis=-1)
        src, dst = np.where(np.triu((dist < radius) & (dist > 0), k=1))
        if len(src) == 0:
            return np.empty((2, 0), dtype=np.int64)
        undirected = np.stack([src, dst], axis=0).astype(np.int64)
    return np.concatenate([undirected, undirected[::-1]], axis=1).astype(np.int64)


def eight_neighbor_edges(cloth_xdim: int, cloth_ydim: int) -> np.ndarray:
    cloth_xdim, cloth_ydim = int(cloth_xdim), int(cloth_ydim)
    all_idx = np.arange(cloth_xdim * cloth_ydim).reshape((cloth_ydim, cloth_xdim))
    senders = []
    receivers = []
    pairs = [
        (all_idx[:, :-1], all_idx[:, :-1] + 1),
        (all_idx[:-1, :], all_idx[:-1, :] + cloth_xdim),
        (all_idx[:-1, :-1], all_idx[:-1, :-1] + 1 + cloth_xdim),
        (all_idx[1:, :-1], all_idx[1:, :-1] + 1 - cloth_xdim),
    ]
    for src, dst in pairs:
        senders.append(src.reshape(-1, 1))
        receivers.append(dst.reshape(-1, 1))
    senders_arr = np.concatenate(senders, axis=0)
    receivers_arr = np.concatenate(receivers, axis=0)
    return np.concatenate([np.concatenate([senders_arr, receivers_arr], axis=0), np.concatenate([receivers_arr, senders_arr], axis=0)], axis=1).T.astype(np.int64)


def edge_features(pointcloud: np.ndarray, edge_index: np.ndarray, edge_type: tuple[float, float]) -> np.ndarray:
    src, dst = edge_index
    disp = pointcloud[src] - pointcloud[dst]
    dist = np.linalg.norm(disp, axis=1, keepdims=True)
    edge_type_arr = np.tile(np.asarray(edge_type, dtype=np.float32), (edge_index.shape[1], 1))
    return np.concatenate([disp, dist, edge_type_arr], axis=1).astype(np.float32)


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    x, y, z, w = [float(v) for v in q]
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.asarray(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float32,
    )

