# SPDX-License-Identifier: BSD-3-Clause
"""Camera configuration for z-up GNNDOM observations."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class CameraConfig:
    """Pinhole camera config in the public GNNDOM z-up frame.

    The default view is an oblique front camera aimed at the vertical cloth face.
    ``x`` and ``y`` span the horizontal plane; ``z`` is height.
    """

    camera_pos: tuple[float, float, float] = (1.2, 0.0, 0.7)
    camera_look_at: tuple[float, float, float] = (0.0, 0.0, 0.2)
    camera_up: tuple[float, float, float] = (0.0, 0.0, 1.0)
    width: int = 360
    height: int = 360
    fov: float = 45.0
    near: float = 0.01
    far: float = 5.0
    voxel_size: float = 0.0216

    def intrinsics(self) -> np.ndarray:
        width, height = float(self.width), float(self.height)
        hfov = np.deg2rad(float(self.fov))
        fx = width / (2.0 * np.tan(hfov / 2.0))
        vfov = 2.0 * np.arctan(np.tan(hfov / 2.0) * height / width)
        fy = height / (2.0 * np.tan(vfov / 2.0))
        return np.asarray(
            [
                [fx, 0.0, width / 2.0],
                [0.0, fy, height / 2.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

    def world_to_camera(self) -> np.ndarray:
        eye = np.asarray(self.camera_pos, dtype=np.float32)
        target = np.asarray(self.camera_look_at, dtype=np.float32)
        up = np.asarray(self.camera_up, dtype=np.float32)
        forward = target - eye
        forward = forward / np.maximum(np.linalg.norm(forward), 1.0e-8)
        right = np.cross(forward, up)
        if np.linalg.norm(right) < 1.0e-8:
            raise ValueError("camera_up is parallel to the camera view direction")
        right = right / np.linalg.norm(right)
        cam_up = np.cross(right, forward)
        rotation = np.stack([right, cam_up, forward], axis=0)
        translation = -rotation @ eye
        matrix = np.eye(4, dtype=np.float32)
        matrix[:3, :3] = rotation
        matrix[:3, 3] = translation
        return matrix

    def camera_to_world(self) -> np.ndarray:
        return np.linalg.inv(self.world_to_camera()).astype(np.float32)

    def to_dict(self) -> dict:
        out = asdict(self)
        out["coordinate_system"] = "z_up_xy_horizontal"
        return out
