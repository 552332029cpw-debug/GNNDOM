# SPDX-License-Identifier: BSD-3-Clause
"""Isaac camera adapter for GNNDOM observations.

The adapter intentionally imports Isaac modules lazily. In remote Isaac runs,
callers can either expose ``env.get_rgbd()`` / ``env.get_depth()`` style methods
or provide an Isaac camera object through ``env.isaac_camera``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .camera_config import CameraConfig
from .geometry_camera import depth_to_pointcloud_with_intrinsics


@dataclass
class IsaacCameraFrame:
    pointcloud: np.ndarray
    intrinsics: np.ndarray
    camera_to_world: np.ndarray
    rgb: np.ndarray | None = None
    depth: np.ndarray | None = None


class IsaacCameraSampler:
    def __init__(self, cfg: CameraConfig):
        self.cfg = cfg

    def capture(self, env: Any, *, save_rgbd: bool = False) -> IsaacCameraFrame:
        """Capture an RGBD/depth frame from an Isaac-capable environment."""

        frame = self._capture_from_env_methods(env, save_rgbd=save_rgbd)
        if frame is not None:
            return frame
        frame = self._capture_from_isaac_camera_attr(env, save_rgbd=save_rgbd)
        if frame is not None:
            return frame
        raise RuntimeError(
            "observation_mode='isaac_camera' requires an Isaac environment exposing "
            "get_rgbd()/get_depth() or an env.isaac_camera object. The current environment "
            "does not provide a renderer camera interface."
        )

    def metadata(self) -> dict:
        return {
            "camera_config": self.cfg.to_dict(),
            "camera_intrinsics": self.cfg.intrinsics(),
            "camera_extrinsics": self.cfg.camera_to_world(),
        }

    def _capture_from_env_methods(self, env: Any, *, save_rgbd: bool) -> IsaacCameraFrame | None:
        rgb = None
        depth = None
        if hasattr(env, "get_rgbd"):
            rgbd = np.asarray(env.get_rgbd(), dtype=np.float32)
            if rgbd.ndim == 3 and rgbd.shape[-1] >= 4:
                rgb = rgbd[..., :3]
                depth = rgbd[..., 3]
        elif hasattr(env, "get_depth"):
            depth = np.asarray(env.get_depth(), dtype=np.float32)
            if save_rgbd and hasattr(env, "get_rgb"):
                rgb = np.asarray(env.get_rgb())
        if depth is None:
            return None
        intrinsics = self._env_intrinsics(env)
        camera_to_world = self._env_camera_to_world(env)
        pointcloud = depth_to_pointcloud_with_intrinsics(depth, intrinsics, camera_to_world)
        return IsaacCameraFrame(
            pointcloud=pointcloud,
            intrinsics=intrinsics,
            camera_to_world=camera_to_world,
            rgb=rgb if save_rgbd else None,
            depth=depth if save_rgbd else None,
        )

    def _capture_from_isaac_camera_attr(self, env: Any, *, save_rgbd: bool) -> IsaacCameraFrame | None:
        camera = getattr(env, "isaac_camera", None)
        if camera is None:
            return None
        depth = None
        rgb = None
        for method_name in ("get_depth", "get_depth_data"):
            if hasattr(camera, method_name):
                depth = np.asarray(getattr(camera, method_name)(), dtype=np.float32)
                break
        if save_rgbd:
            for method_name in ("get_rgb", "get_rgba", "get_rgb_data"):
                if hasattr(camera, method_name):
                    rgb = np.asarray(getattr(camera, method_name)())
                    if rgb.ndim == 3 and rgb.shape[-1] == 4:
                        rgb = rgb[..., :3]
                    break
        if depth is None:
            return None
        intrinsics = self._camera_intrinsics(camera)
        camera_to_world = self._camera_to_world(camera)
        pointcloud = depth_to_pointcloud_with_intrinsics(depth, intrinsics, camera_to_world)
        return IsaacCameraFrame(
            pointcloud=pointcloud,
            intrinsics=intrinsics,
            camera_to_world=camera_to_world,
            rgb=rgb if save_rgbd else None,
            depth=depth if save_rgbd else None,
        )

    def _env_intrinsics(self, env: Any) -> np.ndarray:
        if hasattr(env, "get_camera_intrinsics"):
            return np.asarray(env.get_camera_intrinsics(), dtype=np.float32).reshape(3, 3)
        return self.cfg.intrinsics()

    def _env_camera_to_world(self, env: Any) -> np.ndarray:
        if hasattr(env, "get_camera_to_world"):
            return np.asarray(env.get_camera_to_world(), dtype=np.float32).reshape(4, 4)
        if hasattr(env, "get_camera_params"):
            params = env.get_camera_params()
            if isinstance(params, dict) and "camera_to_world" in params:
                return np.asarray(params["camera_to_world"], dtype=np.float32).reshape(4, 4)
        return self.cfg.camera_to_world()

    def _camera_intrinsics(self, camera: Any) -> np.ndarray:
        for attr in ("get_intrinsics_matrix", "get_intrinsics", "intrinsics"):
            if hasattr(camera, attr):
                value = getattr(camera, attr)
                value = value() if callable(value) else value
                arr = np.asarray(value, dtype=np.float32)
                if arr.size >= 9:
                    return arr.reshape(3, 3)
        return self.cfg.intrinsics()

    def _camera_to_world(self, camera: Any) -> np.ndarray:
        for attr in ("get_world_pose", "get_local_pose"):
            if hasattr(camera, attr):
                value = getattr(camera, attr)()
                matrix = pose_to_matrix(value)
                if matrix is not None:
                    return matrix
        for attr in ("camera_to_world", "world_transform"):
            if hasattr(camera, attr):
                value = getattr(camera, attr)
                value = value() if callable(value) else value
                arr = np.asarray(value, dtype=np.float32)
                if arr.size >= 16:
                    return arr.reshape(4, 4)
        return self.cfg.camera_to_world()


def pose_to_matrix(value: Any) -> np.ndarray | None:
    if not isinstance(value, tuple) or len(value) < 2:
        return None
    pos = np.asarray(value[0], dtype=np.float32).reshape(3)
    quat = np.asarray(value[1], dtype=np.float32).reshape(4)
    rot = quat_to_matrix_xyzw(quat)
    matrix = np.eye(4, dtype=np.float32)
    matrix[:3, :3] = rot
    matrix[:3, 3] = pos
    return matrix


def quat_to_matrix_xyzw(q: np.ndarray) -> np.ndarray:
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
