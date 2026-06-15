# SPDX-License-Identifier: BSD-3-Clause
"""Lightweight tests for GNNDOM planning utilities."""

from __future__ import annotations

import unittest

import numpy as np

from gnndom_plan.actions import split_action
from gnndom_plan.config import PlanConfig
from gnndom_plan.graph import OnlineVisibleGraphBuilder, build_online_velocity_history, downsample_indices
from gnndom_plan.planner import MPCPlanner, actions_from_picker_trajectory, generate_trajectory_zup


class PlanUtilityTests(unittest.TestCase):
    def test_split_action_preserves_grasp_flags(self) -> None:
        action = np.asarray([1.0, 2.0, 3.0, 1.0, 4.0, 5.0, 6.0, 1.0], dtype=np.float32)
        split = split_action(action, 2)
        self.assertEqual(split.shape, (2, 8))
        np.testing.assert_allclose(split[:, [3, 7]], np.ones((2, 2), dtype=np.float32))
        np.testing.assert_allclose(split[:, [0, 1, 2, 4, 5, 6]], np.asarray([[0.5, 1.0, 1.5, 2.0, 2.5, 3.0]] * 2))

    def test_actions_from_picker_trajectory_uses_z_up_axes(self) -> None:
        trajectory = np.asarray(
            [
                [[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]],
                [[0.0, 1.0, 2.0], [1.0, 1.0, 2.0]],
            ],
            dtype=np.float32,
        )
        actions = actions_from_picker_trajectory(trajectory)
        self.assertEqual(actions.shape, (1, 8))
        np.testing.assert_allclose(actions[0, :3], [0.0, 1.0, 1.0])
        np.testing.assert_allclose(actions[0, 4:7], [0.0, 1.0, 1.0])
        np.testing.assert_allclose(actions[0, [3, 7]], [1.0, 1.0])

    def test_generate_trajectory_zup_moves_height_on_z(self) -> None:
        current = np.asarray([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]], dtype=np.float32)
        target = np.asarray([[0.0, 1.0, 2.0], [1.0, 1.0, 2.0]], dtype=np.float32)
        trajectory = generate_trajectory_zup(current, target, acc_max=2.0, dt=0.5)
        self.assertEqual(trajectory.shape[1:], (2, 3))
        np.testing.assert_allclose(trajectory[0], current)
        self.assertGreater(float(trajectory[-1, :, 2].mean()), float(trajectory[0, :, 2].mean()))

    def test_planner_trajectory_releases_grasp_during_drop_tail(self) -> None:
        cfg = PlanConfig(dyn_path=__file__, drop_steps=3)
        planner = MPCPlanner(cfg, dynamics=None)  # type: ignore[arg-type]
        current = np.asarray([[0.0, 0.0, 0.1], [0.1, 0.0, 0.1]], dtype=np.float32)
        target = np.asarray([[0.1, 0.1, 0.2], [0.2, 0.1, 0.2]], dtype=np.float32)
        actions, _ = planner._collect_trajectory(current, target)

        self.assertGreater(len(actions), cfg.drop_steps)
        np.testing.assert_allclose(actions[:-cfg.drop_steps, [3, 7]], 1.0)
        np.testing.assert_allclose(actions[-cfg.drop_steps:], np.zeros((cfg.drop_steps, 8), dtype=np.float32))

    def test_downsample_indices_row_major(self) -> None:
        indices, xdim, ydim = downsample_indices(4, 4, 2)
        np.testing.assert_array_equal(indices, np.asarray([0, 2, 8, 10], dtype=np.int64))
        self.assertEqual((xdim, ydim), (2, 2))

    def test_online_velocity_history_pads_old_steps(self) -> None:
        first = np.zeros((2, 3), dtype=np.float32)
        second = np.ones((2, 3), dtype=np.float32)
        history = build_online_velocity_history([first, second], n_his=3, dt=0.5, pred_time_interval=2)
        self.assertEqual(history.shape, (2, 9))
        np.testing.assert_allclose(history[:, :6], np.zeros((2, 6), dtype=np.float32))
        np.testing.assert_allclose(history[:, 6:], np.ones((2, 3), dtype=np.float32))

    def test_rollout_info_uses_env_physical_target_state(self) -> None:
        class FakeEnv:
            def target_state(self):
                return {
                    "target_pos": np.full((4, 3), 7.0, dtype=np.float32),
                    "geometric_target_pos": np.zeros((4, 3), dtype=np.float32),
                    "target_picker_pos": np.ones((2, 3), dtype=np.float32),
                    "target_settle_steps": np.asarray(12, dtype=np.int32),
                    "target_source": "physical_settled",
                    "geometric_target_source": "flat_fold_pre_settle",
                    "target_release_grasp": np.asarray(0, dtype=np.int32),
                }

        class FakeScene:
            cloth_particle_radius = 0.01
            cloth_xdim = 2
            cloth_ydim = 2
            cloth_size = (2, 2)
            cloth_stiff = (0.9, 1.0, 0.9)
            mass = 0.1
            x_target = 0.1
            rot_angle = 0.0
            env_shape = "sphere"
            obstacle = None

        cfg = PlanConfig(dyn_path=__file__)
        info = OnlineVisibleGraphBuilder(cfg).make_rollout_info(
            FakeEnv(),
            FakeScene(),
            config_id=0,
            downsample_idx=np.asarray([0, 1, 2, 3], dtype=np.int64),
            downsample_x_dim=2,
            downsample_y_dim=2,
        )
        np.testing.assert_allclose(info["target_pos"], np.full((4, 3), 7.0, dtype=np.float32))
        np.testing.assert_allclose(info["geometric_target_pos"], np.zeros((4, 3), dtype=np.float32))
        self.assertEqual(int(info["target_settle_steps"]), 12)
        self.assertEqual(info["target_source"], "physical_settled")
        self.assertEqual(info["geometric_target_source"], "flat_fold_pre_settle")
        self.assertEqual(int(info["target_release_grasp"]), 0)


if __name__ == "__main__":
    unittest.main()
