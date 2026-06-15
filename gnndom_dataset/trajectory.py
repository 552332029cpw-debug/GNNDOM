# SPDX-License-Identifier: BSD-3-Clause
"""ManiFabric-style two-picker fling trajectory generation in z-up coordinates."""

from __future__ import annotations

import numpy as np


def collect_trajectory(
    current_picker_position: np.ndarray,
    target_picker_position: np.ndarray,
    *,
    dt: float,
    rng: np.random.Generator,
    swing_acc: float = 2.0,
    pull_acc: float = 1.0,
    drop_steps: int = 30,
) -> np.ndarray:
    """Return grasped fling actions followed by zero-grasp release steps."""
    xy_trans = float(rng.uniform(0.1, 0.5))
    z_ratio = float(rng.uniform(0.1, 0.5))

    target_xy = target_picker_position[:, [0, 1]]
    segment = target_xy[1] - target_xy[0]
    norm = np.linalg.norm(segment)
    if norm < 1.0e-8:
        norm_direction = np.asarray([1.0, 0.0], dtype=np.float32)
    else:
        norm_direction = np.asarray([segment[1], -segment[0]], dtype=np.float32) / np.float32(norm)

    middle_state = target_picker_position.copy()
    middle_state[:, [0, 1]] = target_xy + xy_trans * norm_direction
    middle_state[:, 2] = current_picker_position[:, 2] + z_ratio * (target_picker_position[:, 2] - current_picker_position[:, 2])

    start_to_middle = generate_trajectory(current_picker_position, middle_state, acc_max=swing_acc, dt=dt)
    middle_to_target = generate_trajectory(middle_state, target_picker_position, acc_max=pull_acc, dt=dt)
    trajectory = np.concatenate((start_to_middle, middle_to_target[1:]), axis=0)
    trajectory_flat = trajectory.reshape(trajectory.shape[0], -1)

    action_list = []
    for step in range(1, trajectory_flat.shape[0]):
        action = np.zeros(8, dtype=np.float32)
        action[:3] = trajectory_flat[step, :3] - trajectory_flat[step - 1, :3]
        action[4:7] = trajectory_flat[step, 3:6] - trajectory_flat[step - 1, 3:6]
        action[[3, 7]] = 1.0
        action_list.append(action)

    if action_list:
        actions = np.asarray(action_list, dtype=np.float32)
    else:
        actions = np.zeros((0, 8), dtype=np.float32)
    action_drop = np.zeros((int(drop_steps), 8), dtype=np.float32)
    return np.concatenate((actions, action_drop), axis=0)


def generate_trajectory(current_picker_position: np.ndarray, target_picker_position: np.ndarray, *, acc_max: float, dt: float) -> np.ndarray:
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
