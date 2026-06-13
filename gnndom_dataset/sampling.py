# SPDX-License-Identifier: BSD-3-Clause
"""Sampling helpers mirroring the data fields used by ManiFabric datasets."""

from __future__ import annotations

import numpy as np

from gnndom_obs import CameraConfig, IsaacCameraSampler, visible_observation_from_pointcloud
from gnndom_obs.geometry_camera import depth_from_points


def downsample_indices(cloth_xdim: int, cloth_ydim: int, scale: int) -> tuple[np.ndarray, int, int]:
    scale = max(int(scale), 1)
    xs = list(range(0, cloth_xdim, scale))
    ys = list(range(0, cloth_ydim, scale))
    indices = [iy * cloth_xdim + ix for iy in ys for ix in xs]
    return np.asarray(indices, dtype=np.int64), len(xs), len(ys)


def observable_indices(downsample_idx: np.ndarray) -> np.ndarray:
    return np.arange(len(downsample_idx), dtype=np.int64)


def voxelize_pointcloud(points: np.ndarray, voxel_size: float) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    if points.size == 0:
        return points.reshape(0, 3)
    voxel_size = max(float(voxel_size), 1.0e-8)
    keys = np.floor(points / voxel_size).astype(np.int64)
    _, unique_idx = np.unique(keys, axis=0, return_index=True)
    return points[np.sort(unique_idx)].astype(np.float32)


def approximate_pointcloud(positions: np.ndarray, downsample_idx: np.ndarray, voxel_size: float) -> np.ndarray:
    return voxelize_pointcloud(np.asarray(positions, dtype=np.float32)[downsample_idx], voxel_size)


def full_observation(positions: np.ndarray, downsample_idx: np.ndarray, voxel_size: float) -> dict:
    return {
        "downsample_observable_idx": observable_indices(downsample_idx),
        "pointcloud": approximate_pointcloud(positions, downsample_idx, voxel_size),
    }


def geometry_camera_observation(
    positions: np.ndarray,
    downsample_idx: np.ndarray,
    *,
    camera_cfg: CameraConfig,
    visibility_threshold: float,
) -> dict:
    depth, visible_particle_idx = depth_from_points(positions, camera_cfg)
    downsample_idx = np.asarray(downsample_idx, dtype=np.int64)
    downsampled_positions = np.asarray(positions, dtype=np.float32)[downsample_idx]
    visible_full_mask = np.zeros(len(positions), dtype=bool)
    visible_full_mask[visible_particle_idx] = True
    downsample_visible_mask = visible_full_mask[downsample_idx]
    visible_downsample_pos = downsampled_positions[downsample_visible_mask]
    obs = visible_observation_from_pointcloud(
        visible_downsample_pos,
        downsampled_positions,
        voxel_size=camera_cfg.voxel_size,
        threshold=visibility_threshold,
        depth=depth,
    )
    return {
        "downsample_observable_idx": obs.downsample_observable_idx,
        "partial_pc_mapped_idx": obs.partial_pc_mapped_idx,
        "pointcloud": obs.pointcloud,
        "depth": obs.depth,
    }


def isaac_camera_observation(
    env,
    positions: np.ndarray,
    downsample_idx: np.ndarray,
    *,
    camera_cfg: CameraConfig,
    visibility_threshold: float,
    save_rgbd: bool,
) -> tuple[dict, dict]:
    frame = IsaacCameraSampler(camera_cfg).capture(env, save_rgbd=save_rgbd)
    downsampled_positions = np.asarray(positions, dtype=np.float32)[np.asarray(downsample_idx, dtype=np.int64)]
    obs = visible_observation_from_pointcloud(
        frame.pointcloud,
        downsampled_positions,
        voxel_size=camera_cfg.voxel_size,
        threshold=visibility_threshold,
        rgb=frame.rgb,
        depth=frame.depth,
    )
    data = {
        "downsample_observable_idx": obs.downsample_observable_idx,
        "partial_pc_mapped_idx": obs.partial_pc_mapped_idx,
        "pointcloud": obs.pointcloud,
    }
    if save_rgbd:
        if obs.rgb is not None:
            data["rgb"] = obs.rgb
        if obs.depth is not None:
            data["depth"] = obs.depth
    metadata = {
        "camera_intrinsics": frame.intrinsics,
        "camera_extrinsics": frame.camera_to_world,
    }
    return data, metadata
