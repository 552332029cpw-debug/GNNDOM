# SPDX-License-Identifier: BSD-3-Clause
"""Online visible graph construction for GNNDOM planning."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from gnndom_graph.features import apply_picker_action, compute_node_attr, compute_visible_edge_attr
from gnndom_obs import CameraConfig, IsaacCameraSampler, visible_observation_from_pointcloud
from gnndom_obs.geometry_camera import depth_from_points

from .config import PlanConfig


@dataclass
class OnlineGraphState:
    pointcloud: np.ndarray
    vel_his: np.ndarray
    picker_position: np.ndarray
    scene_params: np.ndarray
    partial_pc_mapped_idx: np.ndarray
    downsample_idx: np.ndarray
    target_pos: np.ndarray
    target_picker_pos: np.ndarray
    rollout_info: dict


class OnlineVisibleGraphBuilder:
    def __init__(self, cfg: PlanConfig):
        self.cfg = cfg
        self.edge_predictor = None
        if cfg.edge_model_path is not None:
            from gnndom_edge.infer import EdgeGNNMeshPredictor

            self.edge_predictor = EdgeGNNMeshPredictor(
                checkpoint_path=cfg.edge_model_path,
                device=cfg.device,
                neighbor_radius=cfg.neighbor_radius,
            )

    def camera_config(self) -> CameraConfig:
        cfg = self.cfg
        return CameraConfig(
            camera_pos=tuple(cfg.camera_pos),
            camera_look_at=tuple(cfg.camera_look_at),
            width=int(cfg.camera_width),
            height=int(cfg.camera_height),
            fov=float(cfg.camera_fov),
            near=float(cfg.camera_near),
            far=float(cfg.camera_far),
            voxel_size=float(cfg.voxel_size),
        )

    def make_rollout_info(
        self,
        scene_cfg: Any,
        *,
        config_id: int,
        downsample_idx: np.ndarray,
        downsample_x_dim: int,
        downsample_y_dim: int,
    ) -> dict:
        target_pos = flat_positions_zup(scene_cfg).astype(np.float32)
        info = {
            "scene_params": np.asarray(
                [scene_cfg.cloth_particle_radius, downsample_x_dim, downsample_y_dim, config_id],
                dtype=np.float32,
            ),
            "downsample_idx": np.asarray(downsample_idx, dtype=np.int64),
            "target_pos": target_pos,
            "target_picker_pos": target_picker_positions_zup(scene_cfg).astype(np.float32),
            "drop_point_idx": drop_point_indices_zup(scene_cfg.cloth_xdim, scene_cfg.cloth_ydim),
            "ClothSize": np.asarray([scene_cfg.cloth_xdim, scene_cfg.cloth_ydim], dtype=np.int64),
            "ClothStiff": np.asarray(scene_cfg.cloth_stiff, dtype=np.float32),
            "mass": np.asarray(scene_cfg.mass, dtype=np.float32),
            "x_target": np.asarray(scene_cfg.x_target, dtype=np.float32),
            "rot_angle": np.asarray(scene_cfg.rot_angle, dtype=np.float32),
            "env_shape": scene_cfg.env_shape,
            "observation_mode": self.cfg.observation_mode,
            "camera_config": self.camera_config().to_dict(),
            "camera_coordinate_system": "z_up_xy_horizontal",
            "coordinate_system": "z_up_xy_horizontal",
        }
        if scene_cfg.obstacle is not None:
            info.update(
                {
                    "shape_size": np.asarray(scene_cfg.obstacle.shape_size, dtype=np.float32),
                    "shape_pos": np.asarray(scene_cfg.obstacle.shape_pos, dtype=np.float32),
                    "shape_quat": np.asarray(scene_cfg.obstacle.shape_quat, dtype=np.float32),
                }
            )
        return info

    def observe(
        self,
        env,
        scene_cfg: Any,
        *,
        config_id: int,
        downsample_history: list[np.ndarray],
    ) -> OnlineGraphState:
        positions = env.current_positions().astype(np.float32)
        picker_position = env.current_picker_positions().astype(np.float32)
        downsample_idx, downsample_x_dim, downsample_y_dim = downsample_indices(
            scene_cfg.cloth_xdim,
            scene_cfg.cloth_ydim,
            self.cfg.down_sample_scale,
        )
        rollout_info = self.make_rollout_info(
            scene_cfg,
            config_id=config_id,
            downsample_idx=downsample_idx,
            downsample_x_dim=downsample_x_dim,
            downsample_y_dim=downsample_y_dim,
        )
        camera_cfg = self.camera_config()
        if self.cfg.observation_mode == "geometry_camera":
            depth, visible_particle_idx = depth_from_points(positions, camera_cfg)
            visible_pointcloud = positions[np.asarray(visible_particle_idx, dtype=np.int64)]
            obs = visible_observation_from_pointcloud(
                visible_pointcloud,
                positions[downsample_idx],
                voxel_size=camera_cfg.voxel_size,
                threshold=self.cfg.visibility_threshold,
                depth=depth,
            )
            observation = {
                "pointcloud": obs.pointcloud,
                "partial_pc_mapped_idx": obs.partial_pc_mapped_idx,
                "downsample_observable_idx": obs.downsample_observable_idx,
                "depth": obs.depth,
            }
        else:
            try:
                frame = IsaacCameraSampler(camera_cfg).capture(env, save_rgbd=self.cfg.save_rgbd)
                obs = visible_observation_from_pointcloud(
                    frame.pointcloud,
                    positions[downsample_idx],
                    voxel_size=camera_cfg.voxel_size,
                    threshold=self.cfg.visibility_threshold,
                    rgb=frame.rgb,
                    depth=frame.depth,
                )
                observation = {
                    "pointcloud": obs.pointcloud,
                    "partial_pc_mapped_idx": obs.partial_pc_mapped_idx,
                    "downsample_observable_idx": obs.downsample_observable_idx,
                }
                metadata = {
                    "camera_intrinsics": frame.intrinsics,
                    "camera_extrinsics": frame.camera_to_world,
                }
                rollout_info.update(metadata)
            except RuntimeError as exc:
                raise RuntimeError(
                    f"{exc} Use --observation-mode geometry_camera for local smoke planning, "
                    "or attach an Isaac camera interface to the environment."
                ) from exc

        pointcloud = np.asarray(observation["pointcloud"], dtype=np.float32).reshape(-1, 3)
        partial_map = np.asarray(observation["partial_pc_mapped_idx"], dtype=np.int64).reshape(-1)
        if len(pointcloud) == 0 or len(partial_map) == 0:
            raise RuntimeError("Visible camera observation is empty; adjust camera pose, fov, or visibility threshold.")
        if len(pointcloud) != len(partial_map):
            min_len = min(len(pointcloud), len(partial_map))
            pointcloud = pointcloud[:min_len]
            partial_map = partial_map[:min_len]

        downsampled = positions[downsample_idx].astype(np.float32)
        if not downsample_history:
            downsample_history.append(downsampled.copy())
        vel_his_full = build_online_velocity_history(
            downsample_history,
            n_his=self.cfg.n_his,
            dt=self.cfg.dt,
            pred_time_interval=self.cfg.pred_time_interval,
        )
        vel_his = vel_his_full[partial_map].astype(np.float32)

        return OnlineGraphState(
            pointcloud=pointcloud,
            vel_his=vel_his,
            picker_position=picker_position,
            scene_params=np.asarray(rollout_info["scene_params"], dtype=np.float32),
            partial_pc_mapped_idx=partial_map,
            downsample_idx=np.asarray(downsample_idx, dtype=np.int64),
            target_pos=np.asarray(rollout_info["target_pos"], dtype=np.float32),
            target_picker_pos=np.asarray(rollout_info["target_picker_pos"], dtype=np.float32),
            rollout_info=rollout_info,
        )

    def build_model_graph(self, state: OnlineGraphState, action: np.ndarray) -> dict:
        pointcloud_input, velocity_his_input, picked_particles = apply_picker_action(
            state.pointcloud,
            state.vel_his,
            state.picker_position,
            np.asarray(action, dtype=np.float32),
            dt=self.cfg.dt,
            pred_time_interval=self.cfg.pred_time_interval,
        )
        mesh_edges = None
        if self.edge_predictor is not None and self.cfg.use_mesh_edge:
            mesh_edges = self.edge_predictor.predict_mesh_edges(pointcloud_input, threshold=self.cfg.edge_threshold)
        edge_index, edge_attr = compute_visible_edge_attr(
            pointcloud_input,
            cloth_xdim=int(state.scene_params[1]),
            cloth_ydim=int(state.scene_params[2]),
            visible_downsample_idx=state.partial_pc_mapped_idx,
            neighbor_radius=self.cfg.neighbor_radius,
            use_mesh_edge=self.cfg.use_mesh_edge,
            mesh_edges=mesh_edges,
        )
        node_attr = compute_node_attr(
            pointcloud_input,
            velocity_his_input,
            picked_particles,
            rollout_info=state.rollout_info,
            use_es=self.cfg.use_es,
        )
        expected_state_dim = int(self.cfg.n_his) * 3 + 6
        if node_attr.shape[1] != expected_state_dim:
            raise RuntimeError(
                f"Online graph node_dim={node_attr.shape[1]} does not match n_his-derived "
                f"state_dim={expected_state_dim}; check --n-his and checkpoint config."
            )
        return {
            "x": node_attr.astype(np.float32),
            "edge_index": edge_index.astype(np.int64),
            "edge_attr": edge_attr.astype(np.float32),
            "positions": pointcloud_input.astype(np.float32),
            "raw_positions": state.pointcloud.astype(np.float32),
            "target_pos": state.target_pos[state.downsample_idx][state.partial_pc_mapped_idx].astype(np.float32),
            "picker_position": state.picker_position.astype(np.float32),
            "action": np.asarray(action, dtype=np.float32),
            "picked_particles": picked_particles.astype(np.int64),
            "partial_pc_mapped_idx": state.partial_pc_mapped_idx.astype(np.int64),
            "vel_his": velocity_his_input.astype(np.float32),
        }


def build_online_velocity_history(
    downsample_history: list[np.ndarray],
    *,
    n_his: int,
    dt: float,
    pred_time_interval: int,
) -> np.ndarray:
    latest = np.asarray(downsample_history[-1], dtype=np.float32)
    if len(downsample_history) < 2:
        return np.zeros((len(latest), int(n_his) * 3), dtype=np.float32)
    velocities = []
    denom = np.float32(dt * pred_time_interval)
    for i in range(1, len(downsample_history)):
        velocities.append((downsample_history[i] - downsample_history[i - 1]) / denom)
    if not velocities:
        velocities = [np.zeros_like(latest)]
    selected = velocities[-int(n_his) :]
    while len(selected) < int(n_his):
        selected.insert(0, np.zeros_like(latest))
    return np.concatenate(selected, axis=1).astype(np.float32)


def downsample_indices(cloth_xdim: int, cloth_ydim: int, scale: int) -> tuple[np.ndarray, int, int]:
    scale = max(int(scale), 1)
    xs = list(range(0, int(cloth_xdim), scale))
    ys = list(range(0, int(cloth_ydim), scale))
    indices = [iy * int(cloth_xdim) + ix for iy in ys for ix in xs]
    return np.asarray(indices, dtype=np.int64), len(xs), len(ys)


def keypoint_indices_zup(cloth_xdim: int, cloth_ydim: int) -> np.ndarray:
    return np.array([0, cloth_xdim * (cloth_ydim - 1), cloth_xdim * cloth_ydim - 1, cloth_xdim - 1], dtype=np.int64)


def drop_point_indices_zup(cloth_xdim: int, cloth_ydim: int) -> np.ndarray:
    return keypoint_indices_zup(cloth_xdim, cloth_ydim)[:2].copy()


def target_picker_positions_zup(scene_cfg: Any) -> np.ndarray:
    return flat_positions_zup(scene_cfg)[drop_point_indices_zup(scene_cfg.cloth_xdim, scene_cfg.cloth_ydim)].astype(np.float32)


def flat_positions_zup(scene_cfg: Any) -> np.ndarray:
    xdim, ydim = scene_cfg.cloth_size
    particle_radius = float(scene_cfg.cloth_particle_radius)
    x = np.asarray([i * particle_radius for i in range(xdim)], dtype=np.float32)
    y = np.asarray([i * particle_radius for i in range(ydim)], dtype=np.float32)
    y = y - np.mean(y)
    x += np.float32(scene_cfg.x_target)
    xx, yy = np.meshgrid(x, y)
    pos = np.zeros((xdim * ydim, 3), dtype=np.float32)
    pos[:, 0] = xx.flatten()
    pos[:, 1] = yy.flatten()
    pos[:, 2] = np.float32(scene_cfg.target_height)
    if scene_cfg.target_type == "fold":
        folded_x = xx.flatten().copy()
        mean_x = np.mean(folded_x, dtype=np.float32)
        pos[folded_x < mean_x, 2] += np.float32(particle_radius)
        folded_x[folded_x > mean_x] = mean_x - (folded_x[folded_x > mean_x] - mean_x)
        pos[:, 0] = folded_x
    if abs(float(scene_cfg.rot_angle)) > 1.0e-8:
        rot_angle = float(scene_cfg.rot_angle)
        rot = np.asarray(
            [
                [math.cos(rot_angle), -math.sin(rot_angle), 0.0],
                [math.sin(rot_angle), math.cos(rot_angle), 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        pos = (rot @ pos.T).T.astype(np.float32)
    return pos
