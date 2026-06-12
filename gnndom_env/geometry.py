# SPDX-License-Identifier: BSD-3-Clause
"""ManiFabric ClothDrop geometry helpers in GNNDOM z-up coordinates."""

from __future__ import annotations

import math

import numpy as np

from .config import ClothDropConfig, EnvShape, ObstacleConfig


BASE_TARGET_HEIGHT = 0.005


def yaw_quat_soft(rot_angle: float) -> tuple[float, float, float, float]:
    return (0.0, 0.0, math.sin(rot_angle / 2.0), math.cos(rot_angle / 2.0))


def rotate_z(point: np.ndarray, rot_angle: float) -> np.ndarray:
    rot = np.array(
        [
            [math.cos(rot_angle), -math.sin(rot_angle), 0.0],
            [math.sin(rot_angle), math.cos(rot_angle), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    return (rot @ np.asarray(point, dtype=np.float32).reshape(3, 1)).reshape(3)


def keypoint_indices(cloth_xdim: int, cloth_ydim: int) -> np.ndarray:
    return np.array([0, cloth_xdim * (cloth_ydim - 1), cloth_xdim * cloth_ydim - 1, cloth_xdim - 1], dtype=np.int64)


def drop_point_indices(cloth_xdim: int, cloth_ydim: int) -> np.ndarray:
    return keypoint_indices(cloth_xdim, cloth_ydim)[:2].copy()


def flat_positions(cfg: ClothDropConfig, *, fold: bool | None = None) -> np.ndarray:
    xdim, ydim = cfg.cloth_size
    particle_radius = cfg.cloth_particle_radius
    x = np.asarray([i * particle_radius for i in range(xdim)], dtype=np.float32)
    y = np.asarray([i * particle_radius for i in range(ydim)], dtype=np.float32)
    y = y - np.mean(y)
    x += np.float32(cfg.x_target)
    xx, yy = np.meshgrid(x, y)

    pos = np.zeros((xdim * ydim, 3), dtype=np.float32)
    pos[:, 0] = xx.flatten()
    pos[:, 1] = yy.flatten()
    pos[:, 2] = np.float32(cfg.target_height)

    should_fold = cfg.target_type == "fold" if fold is None else fold
    if should_fold:
        folded_x = xx.flatten()
        mean_x = np.mean(folded_x, dtype=np.float32)
        pos[folded_x < mean_x, 2] += np.float32(particle_radius)
        folded_x[folded_x > mean_x] = mean_x - (folded_x[folded_x > mean_x] - mean_x)
        pos[:, 0] = folded_x

    if abs(cfg.rot_angle) > 1.0e-8:
        rot = np.array(
            [
                [math.cos(cfg.rot_angle), -math.sin(cfg.rot_angle), 0.0],
                [math.sin(cfg.rot_angle), math.cos(cfg.rot_angle), 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        pos = (rot @ pos.T).T.astype(np.float32)
    return pos


def vertical_positions(cfg: ClothDropConfig) -> np.ndarray:
    xdim, ydim = cfg.cloth_size
    particle_radius = cfg.cloth_particle_radius
    x = np.asarray([i * particle_radius for i in range(xdim)], dtype=np.float32)
    x = np.asarray(list(reversed(x)), dtype=np.float32)
    y = np.asarray([i * particle_radius for i in range(ydim)], dtype=np.float32)
    y = y - np.mean(y)
    xx, yy = np.meshgrid(x, y)

    pos = np.zeros((xdim * ydim, 3), dtype=np.float32)
    pos[:, 0] = np.float32(cfg.vertical_x_low)
    pos[:, 1] = yy.flatten()
    pos[:, 2] = xx.flatten() - np.min(xx) + np.float32(cfg.vertical_height_low)
    return pos.astype(np.float32)


def target_picker_positions(cfg: ClothDropConfig) -> np.ndarray:
    return flat_positions(cfg)[drop_point_indices(cfg.cloth_xdim, cfg.cloth_ydim)].astype(np.float32)


def triangle_indices(cloth_xdim: int, cloth_ydim: int) -> np.ndarray:
    indices: list[int] = []
    for iy in range(cloth_ydim - 1):
        for ix in range(cloth_xdim - 1):
            i0 = iy * cloth_xdim + ix
            i1 = i0 + 1
            i2 = i0 + cloth_xdim
            i3 = i2 + 1
            indices.extend([i0, i1, i3, i0, i3, i2])
    return np.asarray(indices, dtype=np.int32).reshape(-1, 3)


def make_obstacle(env_shape: EnvShape | None, x_target: float, cloth_xdim: int, particle_radius: float, rot_angle: float) -> ObstacleConfig | None:
    if env_shape is None:
        return None

    center_x = x_target + cloth_xdim * particle_radius / 2.0
    if env_shape == "platform":
        size = (0.2, 0.2, 0.02)
        pos = rotate_z(np.array([center_x, 0.0, 0.0], dtype=np.float32), rot_angle)
        quat = yaw_quat_soft(rot_angle)
    elif env_shape == "sphere":
        size = (0.1, 0.1, 0.1)
        pos = rotate_z(np.array([center_x, 0.0, 0.0], dtype=np.float32), rot_angle)
        quat = (0.0, 0.0, 0.0, 1.0)
    elif env_shape == "rod":
        size = (0.01, 0.25)
        pos = rotate_z(np.array([center_x, 0.0, 0.1], dtype=np.float32), rot_angle)
        quat = (0.0, 0.0, math.sin((rot_angle + math.pi / 2.0) / 2.0), math.cos((rot_angle + math.pi / 2.0) / 2.0))
    elif env_shape == "table":
        size = (0.1, 0.1, 0.02)
        pos = rotate_z(np.array([center_x, 0.0, 0.1], dtype=np.float32), rot_angle)
        quat = yaw_quat_soft(rot_angle)
    else:
        raise ValueError(f"Unknown env_shape: {env_shape}")

    return ObstacleConfig(
        env_shape=env_shape,
        shape_size=tuple(float(v) for v in size),
        shape_pos=tuple(float(v) for v in pos),
        shape_quat=tuple(float(v) for v in quat),
        rot_angle=float(rot_angle),
    )


def target_height_offset(obstacle: ObstacleConfig | None) -> float:
    if obstacle is None:
        return 0.0
    if obstacle.env_shape in {"platform", "sphere"}:
        return float(obstacle.shape_size[2])
    if obstacle.env_shape == "rod":
        return float(obstacle.shape_size[0] + obstacle.shape_pos[2])
    if obstacle.env_shape == "table":
        return float(obstacle.shape_size[2] + obstacle.shape_pos[2])
    return 0.0
