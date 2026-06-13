# SPDX-License-Identifier: BSD-3-Clause
"""Visibility and pointcloud-to-cloth matching utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CameraObservation:
    pointcloud: np.ndarray
    downsample_observable_idx: np.ndarray
    partial_pc_mapped_idx: np.ndarray
    rgb: np.ndarray | None = None
    depth: np.ndarray | None = None


def visible_observation_from_pointcloud(
    pointcloud: np.ndarray,
    downsampled_positions: np.ndarray,
    *,
    voxel_size: float,
    threshold: float | None = None,
    rgb: np.ndarray | None = None,
    depth: np.ndarray | None = None,
) -> CameraObservation:
    """Voxelize visible points and map them to downsampled cloth particles."""

    threshold = float(voxel_size if threshold is None else threshold)
    downsampled_positions = np.asarray(downsampled_positions, dtype=np.float32).reshape(-1, 3)
    vox_pc = voxelize_pointcloud(np.asarray(pointcloud, dtype=np.float32).reshape(-1, 3), voxel_size)
    if len(vox_pc) == 0 or len(downsampled_positions) == 0:
        return CameraObservation(
            pointcloud=np.empty((0, 3), dtype=np.float32),
            downsample_observable_idx=np.empty((0,), dtype=np.int64),
            partial_pc_mapped_idx=np.empty((0,), dtype=np.int64),
            rgb=rgb,
            depth=depth,
        )
    filtered_pc, mapped_idx = match_pointcloud_to_mesh(vox_pc, downsampled_positions, threshold=threshold)
    visible_idx = np.unique(mapped_idx).astype(np.int64)
    return CameraObservation(
        pointcloud=filtered_pc.astype(np.float32),
        downsample_observable_idx=visible_idx,
        partial_pc_mapped_idx=mapped_idx.astype(np.int64),
        rgb=rgb,
        depth=depth,
    )


def voxelize_pointcloud(points: np.ndarray, voxel_size: float) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    if len(points) == 0:
        return points
    voxel_size = max(float(voxel_size), 1.0e-8)
    keys = np.floor(points / voxel_size).astype(np.int64)
    _, unique_idx = np.unique(keys, axis=0, return_index=True)
    return points[np.sort(unique_idx)].astype(np.float32)


def match_pointcloud_to_mesh(pointcloud: np.ndarray, mesh: np.ndarray, *, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    """Mirror ManiFabric ``get_observable_particle_index_3`` semantics."""

    pointcloud = np.asarray(pointcloud, dtype=np.float32).reshape(-1, 3)
    mesh = np.asarray(mesh, dtype=np.float32).reshape(-1, 3)
    if len(pointcloud) == 0 or len(mesh) == 0:
        return np.empty((0, 3), dtype=np.float32), np.empty((0,), dtype=np.int64)
    distance = pairwise_dist(pointcloud, mesh)
    if len(pointcloud) > len(mesh):
        column_idx = np.argmin(distance, axis=1)
    else:
        try:
            from scipy.optimize import linear_sum_assignment

            clipped = distance.copy()
            clipped[clipped > threshold] = 1.0e10
            _, column_idx = linear_sum_assignment(clipped)
        except Exception:
            column_idx = np.argmin(distance, axis=1)
    if len(column_idx) != len(pointcloud):
        column_idx = np.asarray(column_idx, dtype=np.int64)
        row_idx = np.arange(len(column_idx), dtype=np.int64)
        mapped_dist = distance[row_idx, column_idx]
        keep = mapped_dist < threshold
        return pointcloud[row_idx[keep]].astype(np.float32), column_idx[keep].astype(np.int64)
    mapped_dist = distance[np.arange(len(pointcloud)), column_idx]
    keep = mapped_dist < threshold
    return pointcloud[keep].astype(np.float32), column_idx[keep].astype(np.int64)


def pairwise_dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    try:
        from scipy.spatial.distance import cdist

        return cdist(a, b).astype(np.float32)
    except Exception:
        diff = a[:, None, :] - b[None, :, :]
        return np.linalg.norm(diff, axis=-1).astype(np.float32)
