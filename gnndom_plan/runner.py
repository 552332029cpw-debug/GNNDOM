# SPDX-License-Identifier: BSD-3-Clause
"""Plan runner that closes the observe-plan-execute loop."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from gnndom_env import ClothDropConfig, ClothDropRuntimeConfig, ManiFabricClothDropSampler, NewtonClothDropEnv

from .actions import split_action
from .config import PlanConfig
from .graph import OnlineVisibleGraphBuilder
from .planner import MPCPlanner
from .rollout import DynamicsRollout


class PlanRunner:
    def __init__(self, cfg: PlanConfig):
        cfg.validate()
        self.cfg = cfg
        self.cfg.log_dir.mkdir(parents=True, exist_ok=True)
        write_json(self.cfg.log_dir / "plan_config.json", cfg.to_json_dict())
        self.graph_builder = OnlineVisibleGraphBuilder(cfg)
        self.dynamics = DynamicsRollout(cfg, self.graph_builder)

    def run(self) -> dict:
        episode_summaries = []
        sampler = self._make_sampler()
        for episode_idx in range(self.cfg.configurations):
            scene_cfg = sampler.sample(episode_idx)
            summary = self.run_episode(episode_idx, scene_cfg)
            episode_summaries.append(summary)
        result = {"episodes": episode_summaries, "average_final_reward": float(np.mean([x["final_reward"] for x in episode_summaries]))}
        write_json(self.cfg.log_dir / "summary.json", result)
        return result

    def run_episode(self, episode_idx: int, scene_cfg: ClothDropConfig) -> dict:
        episode_dir = self.cfg.log_dir / str(episode_idx)
        episode_dir.mkdir(parents=True, exist_ok=True)
        env = NewtonClothDropEnv(scene_cfg, self._make_runtime())
        env.setup(initial="vertical")
        env.step_until_stable(
            max_steps=self.cfg.settle_steps,
            velocity_threshold=self.cfg.velocity_threshold,
            min_steps=self.cfg.min_stable_steps,
        )

        downsample_history: list[np.ndarray] = []
        init_state = self.graph_builder.observe(
            env,
            scene_cfg,
            config_id=episode_idx,
            downsample_history=downsample_history,
        )
        planner = MPCPlanner(self.cfg, self.dynamics)
        init_result = planner.init_traj(init_state)

        actions_executed = []
        predicted_returns = []
        visible_rewards = []
        control_seq_idx = 0
        max_controls = self.cfg.control_steps if self.cfg.control_steps is not None else len(planner.actions)
        while control_seq_idx < max_controls and control_seq_idx < len(planner.actions):
            state = self.graph_builder.observe(
                env,
                scene_cfg,
                config_id=episode_idx,
                downsample_history=downsample_history,
            )
            action_seq, rollout_result = planner.get_action(state, control_seq_idx=control_seq_idx)
            if len(action_seq) == 0:
                break
            action = action_seq[0]
            small_actions = split_action(action, self.cfg.pred_time_interval)
            for small_action in small_actions:
                execute_action(env, small_action, self.cfg.dt)
                actions_executed.append(small_action.astype(np.float32))
            positions = env.current_positions().astype(np.float32)
            downsample_history.append(positions[state.downsample_idx].copy())
            predicted_returns.append(float(rollout_result.final_ret))
            visible_rewards.append(float(full_target_reward(positions[state.downsample_idx], state.target_pos[state.downsample_idx])))
            control_seq_idx += 1

        final_positions = env.current_positions().astype(np.float32)
        target_positions = init_state.target_pos[init_state.downsample_idx]
        final_reward = full_target_reward(final_positions[init_state.downsample_idx], target_positions)
        action_arr = np.asarray(actions_executed, dtype=np.float32).reshape(-1, 8) if actions_executed else np.zeros((0, 8), dtype=np.float32)
        np.save(episode_dir / "actions.npy", action_arr)
        np.save(episode_dir / "final_positions.npy", final_positions)
        metrics = {
            "episode": int(episode_idx),
            "final_reward": float(final_reward),
            "num_executed_actions": int(len(action_arr)),
            "num_control_steps": int(control_seq_idx),
            "init_predicted_reward": float(init_result.final_ret),
            "predicted_returns": predicted_returns,
            "visible_rewards": visible_rewards,
            "cloth_size": [scene_cfg.cloth_xdim, scene_cfg.cloth_ydim],
            "env_shape": scene_cfg.env_shape,
            "target_source": init_state.rollout_info.get("target_source"),
            "geometric_target_source": init_state.rollout_info.get("geometric_target_source"),
            "target_release_grasp": int(np.asarray(init_state.rollout_info.get("target_release_grasp", -1))),
            "target_settle_steps": int(np.asarray(init_state.rollout_info.get("target_settle_steps", -1))),
            "drop_steps": int(self.cfg.drop_steps),
        }
        write_json(episode_dir / "metrics.json", metrics)
        return metrics

    def _make_sampler(self) -> ManiFabricClothDropSampler:
        base_target_type = "flat" if self.cfg.target_type == "random" else self.cfg.target_type
        base = ClothDropConfig(
            cloth_particle_radius=self.cfg.cloth_particle_radius,
            cloth_size=(self.cfg.cloth_xdim, self.cfg.cloth_ydim),
            cloth_stiff=tuple(self.cfg.cloth_stiffness),
            mass=self.cfg.cloth_mass,
            target_type=base_target_type,  # type: ignore[arg-type]
        )
        return ManiFabricClothDropSampler(
            seed=self.cfg.seed,
            base_cfg=base,
            target_type=self.cfg.target_type,  # type: ignore[arg-type]
            vary_cloth_size=self.cfg.vary_cloth_size,
            vary_stiffness=self.cfg.vary_stiffness,
            vary_mass=self.cfg.vary_mass,
            vary_orientation=self.cfg.vary_orientation,
            env_shape=self.cfg.env_shape,  # type: ignore[arg-type]
        )

    def _make_runtime(self) -> ClothDropRuntimeConfig:
        return ClothDropRuntimeConfig(
            device=self.cfg.runtime_device or self.cfg.device,
            fps=self.cfg.fps,
            substeps=self.cfg.substeps,
            iterations=self.cfg.iterations,
            air_drag=self.cfg.air_drag,
            settle_steps=self.cfg.settle_steps,
            velocity_threshold=self.cfg.velocity_threshold,
            min_stable_steps=self.cfg.min_stable_steps,
        )


def execute_action(env: NewtonClothDropEnv, action: np.ndarray, dt: float) -> None:
    action = np.asarray(action, dtype=np.float32).reshape(8)
    current_picker = env.current_picker_positions()
    next_picker = current_picker.copy()
    next_picker[0] += action[:3]
    next_picker[1] += action[4:7]
    grasp = action[[3, 7]].astype(np.int32)
    velocity = np.stack([action[:3], action[4:7]], axis=0) / np.float32(dt)
    env.set_picker_positions(next_picker, velocity, grasp_flags=grasp)
    env.step()


def full_target_reward(positions: np.ndarray, target_pos: np.ndarray) -> float:
    positions = np.asarray(positions, dtype=np.float32).reshape(-1, 3)
    target_pos = np.asarray(target_pos, dtype=np.float32).reshape(-1, 3)
    if len(positions) == 0 or len(positions) != len(target_pos):
        return -float("inf")
    return -float(np.mean(np.linalg.norm(positions - target_pos, axis=1)))


def write_json(path: Path, data: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(to_jsonable(data), f, indent=2)


def to_jsonable(value):
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value
