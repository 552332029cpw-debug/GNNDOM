# SPDX-License-Identifier: BSD-3-Clause
"""ManiFabric-style rollout dataset generation on the GNNDOM Newton environment."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from gnndom_env import ClothDropConfig, ClothDropRuntimeConfig, ManiFabricClothDropSampler, NewtonClothDropEnv
from gnndom_env.geometry import drop_point_indices, flat_positions, target_picker_positions

from .sampling import approximate_pointcloud, downsample_indices, observable_indices
from .storage import save_rollout_info, save_step
from .trajectory import collect_trajectory


@dataclass(frozen=True)
class DatasetGenerationConfig:
    dataf: Path
    n_rollout: int = 2
    train_valid_ratio: float = 0.9
    dt: float = 1.0 / 60.0
    down_sample_scale: int = 3
    voxel_size: float = 0.0216
    swing_acc: float = 2.0
    pull_acc: float = 1.0
    drop_steps: int = 30
    seed: int = 43


class DataCollector:
    def __init__(
        self,
        cfg: DatasetGenerationConfig,
        *,
        phase: str,
        sampler: ManiFabricClothDropSampler,
        runtime: ClothDropRuntimeConfig,
        config_id_start: int = 0,
    ):
        if phase not in {"train", "valid"}:
            raise ValueError("phase must be train or valid.")
        self.cfg = cfg
        self.phase = phase
        self.sampler = sampler
        self.runtime = runtime
        self.config_id_start = int(config_id_start)
        ratio = float(cfg.train_valid_ratio)
        if phase == "train":
            self.n_rollout = int(cfg.n_rollout * ratio)
        else:
            self.n_rollout = int(cfg.n_rollout - int(cfg.n_rollout * ratio))
        self.data_dir = cfg.dataf / phase
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def gen_dataset(self) -> list[Path]:
        saved_dirs: list[Path] = []
        print(f"Generating dataset for {self.phase.upper()} phase ...", flush=True)
        for rollout_idx in range(self.n_rollout):
            config_id = self.config_id_start + rollout_idx
            print(f"{rollout_idx} / {self.n_rollout}", flush=True)
            rollout_dir = self.data_dir / str(rollout_idx)
            rollout_dir.mkdir(parents=True, exist_ok=True)

            scene_cfg = self.sampler.sample(config_id)
            env = NewtonClothDropEnv(scene_cfg, self.runtime)
            env.setup(initial="vertical")
            env.step_until_stable()
            prev_data, rollout_info = self.get_curr_env_data(env, scene_cfg, config_id=config_id, env_info=True)
            save_rollout_info(rollout_dir, rollout_info)

            rng = np.random.default_rng(self.cfg.seed + config_id)
            actions = collect_trajectory(
                prev_data["picker_position"],
                target_picker_positions(scene_cfg),
                dt=self.cfg.dt,
                rng=rng,
                swing_acc=self.cfg.swing_acc,
                pull_acc=self.cfg.pull_acc,
                drop_steps=self.cfg.drop_steps,
            )

            for step_idx, action in enumerate(actions):
                current_picker = env.current_picker_positions()
                next_picker = current_picker.copy()
                next_picker[0] += action[:3]
                next_picker[1] += action[4:7]
                grasp = action[[3, 7]].astype(np.int32)
                env.set_picker_positions(next_picker, np.stack([action[:3], action[4:7]], axis=0) / np.float32(self.cfg.dt), grasp_flags=grasp)
                env.step()

                curr_data = self.get_curr_env_data(env, scene_cfg, config_id=config_id)
                prev_data["velocities"] = ((curr_data["positions"] - prev_data["positions"]) / np.float32(self.cfg.dt)).astype(np.float32)
                prev_data["action"] = action.astype(np.float32)
                save_step(rollout_dir, step_idx, prev_data)
                prev_data = curr_data

            prev_data["action"] = np.zeros(8, dtype=np.float32)
            prev_data["velocities"] = np.zeros_like(prev_data["positions"], dtype=np.float32)
            save_step(rollout_dir, len(actions), prev_data)
            saved_dirs.append(rollout_dir)
            print(f"[INFO] wrote rollout {rollout_dir}", flush=True)
        return saved_dirs

    def get_curr_env_data(
        self,
        env: NewtonClothDropEnv,
        scene_cfg: ClothDropConfig,
        *,
        config_id: int,
        env_info: bool = False,
    ):
        positions = env.current_positions().astype(np.float32)
        picker_position = env.current_picker_positions().astype(np.float32)
        downsample_idx, downsample_x_dim, downsample_y_dim = downsample_indices(
            scene_cfg.cloth_xdim,
            scene_cfg.cloth_ydim,
            self.cfg.down_sample_scale,
        )
        downsample_observable_idx = observable_indices(downsample_idx)
        pointcloud = approximate_pointcloud(positions, downsample_idx, self.cfg.voxel_size)
        step_data = {
            "positions": positions,
            "picker_position": picker_position,
            "downsample_observable_idx": downsample_observable_idx,
            "pointcloud": pointcloud,
        }
        if not env_info:
            return step_data

        target_pos = flat_positions(scene_cfg).astype(np.float32)
        target_picker_pos = target_picker_positions(scene_cfg).astype(np.float32)
        rollout_info = {
            "scene_params": np.asarray([scene_cfg.cloth_particle_radius, downsample_x_dim, downsample_y_dim, config_id], dtype=np.float32),
            "downsample_idx": downsample_idx,
            "target_pos": target_pos,
            "target_picker_pos": target_picker_pos,
            "drop_point_idx": drop_point_indices(scene_cfg.cloth_xdim, scene_cfg.cloth_ydim),
            "ClothSize": np.asarray([scene_cfg.cloth_xdim, scene_cfg.cloth_ydim], dtype=np.int64),
            "ClothStiff": np.asarray(scene_cfg.cloth_stiff, dtype=np.float32),
            "mass": np.asarray(scene_cfg.mass, dtype=np.float32),
            "x_target": np.asarray(scene_cfg.x_target, dtype=np.float32),
            "rot_angle": np.asarray(scene_cfg.rot_angle, dtype=np.float32),
            "env_shape": scene_cfg.env_shape,
            "coordinate_system": "z_up_xy_horizontal",
        }
        if scene_cfg.obstacle is not None:
            rollout_info.update(
                {
                    "shape_size": np.asarray(scene_cfg.obstacle.shape_size, dtype=np.float32),
                    "shape_pos": np.asarray(scene_cfg.obstacle.shape_pos, dtype=np.float32),
                    "shape_quat": np.asarray(scene_cfg.obstacle.shape_quat, dtype=np.float32),
                }
            )
        return step_data, rollout_info

