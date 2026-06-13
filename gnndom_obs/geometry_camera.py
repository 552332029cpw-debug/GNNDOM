# SPDX-License-Identifier: BSD-3-Clause
"""Pure numpy camera geometry used for tests and debug fallback."""

from __future__ import annotations

import numpy as np

from .camera_config import CameraConfig


def project_points(points: np.ndarray, cfg: CameraConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    homog = np.concatenate([points, np.ones((len(points), 1), dtype=np.float32)], axis=1)
    cam = (cfg.world_to_camera() @ homog.T).T[:, :3]
    depth = cam[:, 2]
    k = cfg.intrinsics()
    u = cam[:, 0] * k[0, 0] / np.maximum(depth, 1.0e-8) + k[0, 2]
    v = cam[:, 1] * k[1, 1] / np.maximum(depth, 1.0e-8) + k[1, 2]
    return u.astype(np.float32), v.astype(np.float32), depth.astype(np.float32)


def depth_from_points(points: np.ndarray, cfg: CameraConfig) -> tuple[np.ndarray, np.ndarray]:
    """Rasterize particles to a depth image and return visible particle ids."""

    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    depth = np.zeros((int(cfg.height), int(cfg.width)), dtype=np.float32)
    owner = np.full((int(cfg.height), int(cfg.width)), -1, dtype=np.int64)
    u, v, z = project_points(points, cfg)
    ui = np.rint(u).astype(np.int64)
    vi = np.rint(v).astype(np.int64)
    valid = (z > cfg.near) & (z < cfg.far) & (ui >= 0) & (ui < cfg.width) & (vi >= 0) & (vi < cfg.height)
    for particle_idx in np.where(valid)[0]:
        px, py, pz = int(ui[particle_idx]), int(vi[particle_idx]), float(z[particle_idx])
        old = depth[py, px]
        if old <= 0.0 or pz < old:
            depth[py, px] = pz
            owner[py, px] = int(particle_idx)
    visible = np.unique(owner[owner >= 0]).astype(np.int64)
    return depth, visible


def pointcloud_from_depth(depth: np.ndarray, cfg: CameraConfig, *, rgb: np.ndarray | None = None) -> np.ndarray:
    """Back-project a depth image to z-up world coordinates."""

    del rgb
    depth = np.asarray(depth, dtype=np.float32)
    height, width = depth.shape
    k = cfg.intrinsics()
    ys, xs = np.where(depth > 0.0)
    if len(xs) == 0:
        return np.empty((0, 3), dtype=np.float32)
    z = depth[ys, xs]
    x = (xs.astype(np.float32) - k[0, 2]) * z / k[0, 0]
    y = (ys.astype(np.float32) - k[1, 2]) * z / k[1, 1]
    cam = np.stack([x, y, z, np.ones_like(z)], axis=1)
    world = (cfg.camera_to_world() @ cam.T).T[:, :3]
    return world.astype(np.float32)


def depth_to_pointcloud_with_intrinsics(depth: np.ndarray, intrinsics: np.ndarray, camera_to_world: np.ndarray) -> np.ndarray:
    depth = np.asarray(depth, dtype=np.float32)
    intrinsics = np.asarray(intrinsics, dtype=np.float32).reshape(3, 3)
    camera_to_world = np.asarray(camera_to_world, dtype=np.float32).reshape(4, 4)
    ys, xs = np.where(depth > 0.0)
    if len(xs) == 0:
        return np.empty((0, 3), dtype=np.float32)
    z = depth[ys, xs]
    x = (xs.astype(np.float32) - intrinsics[0, 2]) * z / intrinsics[0, 0]
    y = (ys.astype(np.float32) - intrinsics[1, 2]) * z / intrinsics[1, 1]
    cam = np.stack([x, y, z, np.ones_like(z)], axis=1)
    return (camera_to_world @ cam.T).T[:, :3].astype(np.float32)
