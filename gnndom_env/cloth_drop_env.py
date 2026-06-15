# SPDX-License-Identifier: BSD-3-Clause
"""Unified Newton-backed ManiFabric ClothDrop environment."""

from __future__ import annotations

from dataclasses import asdict
import math

import numpy as np

from .config import ClothDropConfig, ClothDropRuntimeConfig
from .geometry import drop_point_indices, geometric_target_positions, target_picker_positions, vertical_positions
from .newton_backend import (
    SIM_SCALE,
    build_model_from_config,
    configure_device,
    make_vbd_solver,
    newton,
    newton_to_soft_positions,
    soft_to_newton_positions,
    wp,
)


@wp.kernel
def enforce_point_picker_kernel(
    particle_q: wp.array[wp.vec3],
    particle_qd: wp.array[wp.vec3],
    particle_ids: wp.array[wp.int32],
    picker_pos: wp.array[wp.vec3],
    picker_vel: wp.array[wp.vec3],
    grasp_flags: wp.array[wp.int32],
):
    tid = wp.tid()
    if grasp_flags[tid] <= 0:
        return
    particle_id = particle_ids[tid]
    particle_q[particle_id] = picker_pos[tid]
    particle_qd[particle_id] = picker_vel[tid]


@wp.kernel
def damp_particle_velocity_kernel(
    particle_qd: wp.array[wp.vec3],
    damping_factor: float,
):
    tid = wp.tid()
    particle_qd[tid] = particle_qd[tid] * damping_factor


class NewtonClothDropEnv:
    """A Newton implementation of ManiFabric's ClothDrop scene lifecycle."""

    def __init__(self, cfg: ClothDropConfig, runtime: ClothDropRuntimeConfig | None = None):
        self.cfg = cfg
        self.runtime = runtime or ClothDropRuntimeConfig()
        self.frame_dt = 1.0 / float(self.runtime.fps)
        self.sim_dt = self.frame_dt / float(self.runtime.substeps)
        self.model = None
        self.state_0 = None
        self.state_1 = None
        self.control = None
        self.contacts = None
        self.solver = None
        self.picker_ids_np = drop_point_indices(cfg.cloth_xdim, cfg.cloth_ydim).astype(np.int32)
        self.picker_ids = None
        self.picker_pos = None
        self.picker_vel = None
        self.picker_grasp_np = np.ones(2, dtype=np.int32)
        self.picker_grasp = None
        self._prev_positions: np.ndarray | None = None
        self._settled_target_pos: np.ndarray | None = None
        self._settled_target_steps: int | None = None

    def setup(self, *, initial: str = "vertical") -> None:
        configure_device(self.runtime.device)
        if initial == "target":
            initial_positions = geometric_target_positions(self.cfg)
        elif initial == "vertical":
            initial_positions = None
        else:
            raise ValueError("initial must be vertical or target.")
        self._build(initial_positions=initial_positions)
        if initial == "target":
            self.set_picker_positions(target_picker_positions(self.cfg))
        else:
            self.enable_fixed_pickers()

    def reset_to_vertical(self) -> None:
        self._build(initial_positions=None)
        self.enable_fixed_pickers()

    def reset_to_target(self, *, grasp: bool = True) -> None:
        self._build(initial_positions=geometric_target_positions(self.cfg))
        grasp_flags = np.ones(2, dtype=np.int32) if grasp else np.zeros(2, dtype=np.int32)
        self.set_picker_positions(target_picker_positions(self.cfg), grasp_flags=grasp_flags)

    def settled_target_positions(self, *, force: bool = False) -> np.ndarray:
        """Return the physical target after settling the geometric target in-scene."""
        if self._settled_target_pos is not None and not force:
            return self._settled_target_pos.copy()

        target_env = NewtonClothDropEnv(self.cfg, self.runtime)
        target_env.reset_to_target(grasp=False)
        steps = target_env.step_until_stable()
        self._settled_target_pos = target_env.current_positions().astype(np.float32)
        self._settled_target_steps = int(steps)
        return self._settled_target_pos.copy()

    def target_state(self, *, force: bool = False) -> dict:
        target_pos = self.settled_target_positions(force=force)
        geometric_target_pos = geometric_target_positions(self.cfg).astype(np.float32)
        return {
            "target_pos": target_pos,
            "geometric_target_pos": geometric_target_pos,
            "target_picker_pos": target_picker_positions(self.cfg).astype(np.float32),
            "target_settle_steps": np.asarray(self._settled_target_steps if self._settled_target_steps is not None else -1, dtype=np.int32),
            "target_source": "physical_settled",
            "geometric_target_source": "flat_fold_pre_settle",
            "target_release_grasp": np.asarray(0, dtype=np.int32),
        }

    def step(self) -> None:
        assert self.model is not None and self.state_0 is not None and self.state_1 is not None
        assert self.solver is not None and self.contacts is not None and self.control is not None
        for _ in range(self.runtime.substeps):
            self.state_0.clear_forces()
            self._apply_picker_arrays(self.state_0)
            self.model.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self._apply_picker_arrays(self.state_1)
            self.state_0, self.state_1 = self.state_1, self.state_0
            self._apply_air_drag(self.state_0)

    def step_until_stable(self, max_steps: int | None = None, velocity_threshold: float | None = None, min_steps: int | None = None) -> int:
        max_steps = self.runtime.settle_steps if max_steps is None else max_steps
        velocity_threshold = self.runtime.velocity_threshold if velocity_threshold is None else velocity_threshold
        min_steps = self.runtime.min_stable_steps if min_steps is None else min_steps
        steps_taken = 0
        for step_idx in range(max_steps):
            self.step()
            steps_taken = step_idx + 1
            if steps_taken < min_steps:
                continue
            speeds = np.linalg.norm(self.current_velocities(), axis=1)
            if float(np.max(speeds)) < velocity_threshold:
                break
        return steps_taken

    def get_current_config(self) -> dict:
        config = self.cfg.to_manifabric_dict()
        config.update(self.target_state())
        config["drop_point_idx"] = self.picker_ids_np.astype(np.int64)
        config["runtime"] = asdict(self.runtime)
        return config

    def current_positions(self) -> np.ndarray:
        assert self.state_0 is not None
        return newton_to_soft_positions(self.state_0.particle_q.numpy().copy()) / np.float32(SIM_SCALE)

    def current_velocities(self) -> np.ndarray:
        current = self.current_positions()
        if self._prev_positions is None:
            self._prev_positions = current.copy()
            return np.zeros_like(current, dtype=np.float32)
        velocities = (current - self._prev_positions) / np.float32(self.frame_dt)
        self._prev_positions = current.copy()
        return velocities.astype(np.float32)

    def current_picker_positions(self) -> np.ndarray:
        assert self.picker_pos is not None
        return newton_to_soft_positions(self.picker_pos.numpy().copy()) / np.float32(SIM_SCALE)

    def set_picker_positions(self, pos_soft: np.ndarray, vel_soft: np.ndarray | None = None, grasp_flags: np.ndarray | None = None) -> None:
        assert self.model is not None and self.state_0 is not None
        if self.picker_ids is None:
            self.picker_ids = wp.array(self.picker_ids_np, dtype=wp.int32, device=self.model.device)
        if vel_soft is None:
            vel_soft = np.zeros_like(pos_soft)
        if grasp_flags is not None:
            self.set_picker_grasp(grasp_flags)
        elif self.picker_grasp is None:
            self.set_picker_grasp(self.picker_grasp_np)
        self.picker_pos = wp.array(soft_to_newton_positions(np.asarray(pos_soft, dtype=np.float32)) * SIM_SCALE, dtype=wp.vec3, device=self.model.device)
        self.picker_vel = wp.array(soft_to_newton_positions(np.asarray(vel_soft, dtype=np.float32)) * SIM_SCALE, dtype=wp.vec3, device=self.model.device)
        self._apply_picker_arrays(self.state_0)

    def set_picker_grasp(self, grasp_flags: np.ndarray) -> None:
        assert self.model is not None
        grasp = np.asarray(grasp_flags, dtype=np.int32).reshape(2)
        self.picker_grasp_np = (grasp > 0).astype(np.int32)
        self.picker_grasp = wp.array(self.picker_grasp_np, dtype=wp.int32, device=self.model.device)
        flags = self.model.particle_flags.numpy()
        active_bit = int(newton.ParticleFlags.ACTIVE)
        for local_idx, particle_id in enumerate(self.picker_ids_np):
            if self.picker_grasp_np[local_idx] > 0:
                flags[particle_id] &= ~active_bit
            else:
                flags[particle_id] |= active_bit
        self.model.particle_flags = wp.array(flags, dtype=self.model.particle_flags.dtype, device=self.model.device)

    def enable_fixed_pickers(self) -> None:
        assert self.model is not None and self.state_0 is not None
        self.picker_ids = wp.array(self.picker_ids_np, dtype=wp.int32, device=self.model.device)
        self.set_picker_grasp(np.ones(2, dtype=np.int32))
        start_soft = vertical_positions(self.cfg)[self.picker_ids_np]
        self.set_picker_positions(start_soft, np.zeros_like(start_soft))

    def _build(self, *, initial_positions: np.ndarray | None) -> None:
        self.model = build_model_from_config(
            self.cfg,
            initial_positions=initial_positions,
            self_contact=self.runtime.self_contact,
            contact_radius_scale=self.runtime.contact_radius_scale,
        )
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()
        self.solver = make_vbd_solver(
            self.model,
            self.cfg,
            iterations=self.runtime.iterations,
            self_contact=self.runtime.self_contact,
        )
        self.picker_ids = wp.array(self.picker_ids_np, dtype=wp.int32, device=self.model.device)
        self.picker_pos = None
        self.picker_vel = None
        self.picker_grasp = None
        self.picker_grasp_np = np.ones(2, dtype=np.int32)
        self._prev_positions = None

    def _apply_picker_arrays(self, state) -> None:
        if self.picker_ids is None or self.picker_pos is None or self.picker_vel is None or self.picker_grasp is None:
            return
        wp.launch(
            enforce_point_picker_kernel,
            dim=2,
            inputs=[state.particle_q, state.particle_qd, self.picker_ids, self.picker_pos, self.picker_vel, self.picker_grasp],
            device=self.model.device,
        )

    def _apply_air_drag(self, state) -> None:
        if self.runtime.air_drag <= 0.0:
            return
        damping_factor = float(math.exp(-float(self.runtime.air_drag) * float(self.sim_dt)))
        wp.launch(
            damp_particle_velocity_kernel,
            dim=self.model.particle_count,
            inputs=[state.particle_qd, damping_factor],
            device=self.model.device,
        )
