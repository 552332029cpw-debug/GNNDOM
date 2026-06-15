# SPDX-License-Identifier: BSD-3-Clause
"""Random-shooting MPC planner for GNNDOM z-up ClothDrop."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .config import PlanConfig
from .graph import OnlineGraphState

if TYPE_CHECKING:
    from .rollout import DynamicsRollout, RolloutResult


class MPCPlanner:
    def __init__(self, cfg: PlanConfig, dynamics: DynamicsRollout):
        self.cfg = cfg
        self.dynamics = dynamics
        self.rng = np.random.default_rng(cfg.seed)
        self.actions = np.zeros((0, 8), dtype=np.float32)
        self.step_mid = 0
        self.delta_actions = default_delta_actions(cfg.shooting_number)

    def init_traj(self, state: OnlineGraphState) -> RolloutResult:
        candidates = []
        mids = []
        returns = []
        results = []
        for _ in range(self.cfg.sampling_num):
            actions, step_mid = self._collect_trajectory(state.picker_position, state.target_picker_pos)
            candidates.append(actions)
            mids.append(step_mid)
            result = self.dynamics.rollout(state, actions)
            results.append(result)
            returns.append(result.final_ret)
        best = int(np.argmax(returns))
        self.actions = candidates[best]
        self.step_mid = int(mids[best])
        return results[best]

    def get_action(self, state: OnlineGraphState, *, control_seq_idx: int) -> tuple[np.ndarray, RolloutResult]:
        candidates = self._candidate_sequences(state, control_seq_idx)
        returns = []
        results = []
        for actions in candidates:
            result = self.dynamics.rollout(state, actions)
            results.append(result)
            returns.append(result.final_ret)
        best = int(np.argmax(returns))
        action_seq = candidates[best]
        self._update_traj(action_seq, control_seq_idx)
        return action_seq, results[best]

    def _candidate_sequences(self, state: OnlineGraphState, control_seq_idx: int) -> list[np.ndarray]:
        if control_seq_idx >= len(self.actions):
            return [np.zeros((1, 8), dtype=np.float32)]
        if control_seq_idx >= self.step_mid:
            return [self.actions[control_seq_idx:].astype(np.float32)]

        actions_swing = self.actions[control_seq_idx : self.step_mid].astype(np.float32)
        expanded = np.expand_dims(actions_swing, 0).repeat(len(self.delta_actions), axis=0)
        for i, delta in enumerate(self.delta_actions):
            expanded[i, :, :3] += actions_swing[:, :3] * delta
            expanded[i, :, 4:7] += actions_swing[:, 4:7] * delta

        assumed_1 = state.picker_position[0] + np.sum(expanded[:, :, :3], axis=1)
        assumed_2 = state.picker_position[1] + np.sum(expanded[:, :, 4:7], axis=1)
        assumed_mid = np.stack((assumed_1, assumed_2), axis=1)
        candidates = []
        for i, mid_pos in enumerate(assumed_mid):
            pull = self._generate_pull_actions(mid_pos, state.target_picker_pos)
            tail = np.zeros((self.cfg.drop_steps, 8), dtype=np.float32)
            candidates.append(np.concatenate((expanded[i], pull, tail), axis=0).astype(np.float32))
        return candidates

    def _update_traj(self, actions: np.ndarray, control_seq_idx: int) -> None:
        prefix = self.actions[:control_seq_idx] if control_seq_idx > 0 else np.zeros((0, 8), dtype=np.float32)
        self.actions = np.concatenate((prefix, actions.astype(np.float32)), axis=0)

    def _collect_trajectory(self, current_picker_position: np.ndarray, target_picker_position: np.ndarray) -> tuple[np.ndarray, int]:
        xy_trans = float(self.rng.uniform(0.1, 0.5))
        z_ratio = float(self.rng.uniform(0.1, 0.6))
        target_xy = target_picker_position[:, [0, 1]]
        segment = target_xy[1] - target_xy[0]
        norm = np.linalg.norm(segment)
        if norm < 1.0e-8:
            norm_direction = np.asarray([1.0, 0.0], dtype=np.float32)
        else:
            norm_direction = np.asarray([segment[1], -segment[0]], dtype=np.float32) / np.float32(norm)

        middle = target_picker_position.copy()
        middle[:, [0, 1]] = target_xy + xy_trans * norm_direction
        middle[:, 2] = current_picker_position[:, 2] + z_ratio * (target_picker_position[:, 2] - current_picker_position[:, 2])
        start_to_middle = generate_trajectory_zup(
            current_picker_position,
            middle,
            acc_max=self.cfg.swing_acc,
            dt=self.cfg.dt * self.cfg.pred_time_interval,
        )
        step_mid = len(start_to_middle) - 1
        middle_to_target = generate_trajectory_zup(
            middle,
            target_picker_position,
            acc_max=self.cfg.pull_acc,
            dt=self.cfg.dt * self.cfg.pred_time_interval,
        )
        trajectory = np.concatenate((start_to_middle, middle_to_target[1:]), axis=0)
        actions = actions_from_picker_trajectory(trajectory)
        tail = np.zeros((self.cfg.drop_steps, 8), dtype=np.float32)
        return np.concatenate((actions, tail), axis=0).astype(np.float32), step_mid

    def _generate_pull_actions(self, current_picker_position: np.ndarray, target_picker_position: np.ndarray) -> np.ndarray:
        trajectory = generate_trajectory_zup(
            current_picker_position,
            target_picker_position,
            acc_max=self.cfg.pull_acc,
            dt=self.cfg.dt * self.cfg.pred_time_interval,
        )
        return actions_from_picker_trajectory(trajectory)


def actions_from_picker_trajectory(trajectory: np.ndarray) -> np.ndarray:
    trajectory = np.asarray(trajectory, dtype=np.float32)
    actions = []
    for idx in range(1, len(trajectory)):
        action = np.ones(8, dtype=np.float32)
        action[:3] = trajectory[idx, 0] - trajectory[idx - 1, 0]
        action[4:7] = trajectory[idx, 1] - trajectory[idx - 1, 1]
        action[[3, 7]] = 1.0
        actions.append(action)
    if not actions:
        return np.zeros((0, 8), dtype=np.float32)
    return np.asarray(actions, dtype=np.float32)


def generate_trajectory_zup(current_picker_position: np.ndarray, target_picker_position: np.ndarray, *, acc_max: float, dt: float) -> np.ndarray:
    current_picker_position = np.asarray(current_picker_position, dtype=np.float32)
    target_picker_position = np.asarray(target_picker_position, dtype=np.float32)
    initial_xy = current_picker_position[:, [0, 1]]
    final_xy = target_picker_position[:, [0, 1]]
    angle = np.arctan2(final_xy[1, 1] - final_xy[0, 1], final_xy[1, 0] - final_xy[0, 0]) - np.arctan2(
        initial_xy[1, 1] - initial_xy[0, 1],
        initial_xy[1, 0] - initial_xy[0, 0],
    )
    translation = target_picker_position.mean(axis=0) - current_picker_position.mean(axis=0)
    acc_max = max(float(acc_max), 1.0e-6)
    dt = max(float(dt), 1.0e-8)
    time_steps = np.sqrt(4.0 * np.abs(translation) / acc_max) / dt
    steps = max(int(np.ceil(np.max(time_steps))), 1)
    rot_steps = float(angle) / float(steps)

    accel_steps = max(steps // 2, 1)
    decel_steps = max(steps - accel_steps, 1)
    v_max = translation * 2.0 / (steps * dt)
    accelerate = v_max / (accel_steps * dt)
    decelerate = -v_max / (decel_steps * dt)
    incremental_translation = np.zeros(3, dtype=np.float32)
    positions = [current_picker_position.astype(np.float32)]

    for i in range(steps):
        if i < accel_steps:
            incremental_translation = (incremental_translation / dt + accelerate * dt) * dt
        else:
            incremental_translation = (incremental_translation / dt + decelerate * dt) * dt
        vertices = positions[-1] + incremental_translation
        rotation_matrix = np.asarray(
            [
                [np.cos(rot_steps), -np.sin(rot_steps), 0.0],
                [np.sin(rot_steps), np.cos(rot_steps), 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        center = vertices.mean(axis=0)
        vertices = (rotation_matrix @ (vertices - center).T).T + center
        positions.append(vertices.astype(np.float32))

    return np.asarray(positions, dtype=np.float32)


def default_delta_actions(shooting_number: int) -> np.ndarray:
    base = np.asarray(
        [
            [0.1, 0.1, 0.1],
            [0.1, 0.0, 0.1],
            [0.1, -0.1, 0.1],
            [0.0, 0.1, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, -0.1, 0.0],
            [-0.1, 0.1, -0.1],
            [-0.1, 0.0, -0.1],
            [-0.1, -0.1, -0.1],
        ],
        dtype=np.float32,
    )
    shooting_number = int(shooting_number)
    if shooting_number <= len(base):
        return base[:shooting_number]
    rng = np.random.default_rng(0)
    extra = rng.uniform(-0.1, 0.1, size=(shooting_number - len(base), 3)).astype(np.float32)
    return np.concatenate((base, extra), axis=0)
