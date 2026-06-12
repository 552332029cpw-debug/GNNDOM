# SPDX-License-Identifier: BSD-3-Clause
"""Sampling helpers mirroring the data fields used by ManiFabric datasets."""

from __future__ import annotations

import numpy as np


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

