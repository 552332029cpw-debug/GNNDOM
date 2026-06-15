# SPDX-License-Identifier: BSD-3-Clause
"""Newton backend helpers for GNNDOM z-up ClothDrop scenes."""

from __future__ import annotations

import math
import os
from pathlib import Path
import sys

import numpy as np

ACTIVE_ROOT = Path(__file__).resolve().parents[2]
NEWTON_REPO = ACTIVE_ROOT / "newton"
os.environ.setdefault("WARP_CACHE_PATH", str(ACTIVE_ROOT / ".warp_cache"))
if str(NEWTON_REPO) not in sys.path:
    sys.path.insert(0, str(NEWTON_REPO))

import warp as wp  # noqa: E402

import newton  # noqa: E402
from newton.solvers import SolverVBD  # noqa: E402

from .config import ClothDropConfig
from .geometry import triangle_indices, vertical_positions

SIM_SCALE = 100.0


def configure_device(device: str | None) -> None:
    if device:
        wp.set_device(device)


def soft_to_newton_positions(positions: np.ndarray) -> np.ndarray:
    return np.asarray(positions, dtype=np.float32).copy()


def newton_to_soft_positions(positions: np.ndarray) -> np.ndarray:
    return np.asarray(positions, dtype=np.float32).copy()


def soft_vec_to_newton(vec: tuple[float, ...] | np.ndarray, scale: float = 1.0) -> wp.vec3:
    arr = np.asarray(vec, dtype=np.float32)
    return wp.vec3(float(arr[0] * scale), float(arr[1] * scale), float(arr[2] * scale))


def soft_yaw_to_newton_quat(rot_angle: float) -> wp.quat:
    return wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), float(rot_angle))


def horizontal_rod_quat(rot_angle: float) -> wp.quat:
    yaw = soft_yaw_to_newton_quat(rot_angle)
    z_to_y = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), -math.pi * 0.5)
    return yaw * z_to_y


def build_model_from_config(
    cfg: ClothDropConfig,
    *,
    initial_positions: np.ndarray | None = None,
    self_contact: bool = False,
    contact_radius_scale: float = 0.5,
):
    builder = newton.ModelBuilder(gravity=-9.81 * SIM_SCALE)
    add_ground(builder, cfg)
    add_obstacle(builder, cfg)

    if initial_positions is None:
        add_vertical_cloth_grid(builder, cfg, contact_radius_scale=contact_radius_scale)
    else:
        add_cloth_mesh(builder, cfg, initial_positions, contact_radius_scale=contact_radius_scale)

    builder.color(include_bending=True)
    model = builder.finalize(requires_grad=False)
    model.soft_contact_ke = cfg.contact_ke
    model.soft_contact_kd = cfg.contact_kd
    model.soft_contact_mu = cfg.contact_mu
    return model


def make_vbd_solver(model, cfg: ClothDropConfig, *, iterations: int, self_contact: bool):
    spacing = cfg.cloth_particle_radius * SIM_SCALE
    return SolverVBD(
        model,
        iterations=iterations,
        particle_enable_self_contact=self_contact,
        particle_self_contact_radius=spacing * 0.75,
        particle_self_contact_margin=spacing * 0.75,
        particle_topological_contact_filter_threshold=1,
        particle_rest_shape_contact_exclusion_radius=spacing * 2.0,
        particle_vertex_contact_buffer_size=16,
        particle_edge_contact_buffer_size=20,
        particle_collision_detection_interval=-1,
        rigid_contact_k_start=cfg.contact_ke,
    )


def shape_contact_config(builder, cfg: ClothDropConfig):
    return builder.ShapeConfig(
        ke=cfg.shape_contact_ke if cfg.shape_contact_ke is not None else cfg.contact_ke,
        kd=cfg.shape_contact_kd if cfg.shape_contact_kd is not None else cfg.contact_kd,
        mu=cfg.shape_contact_mu if cfg.shape_contact_mu is not None else cfg.contact_mu,
    )


def add_vertical_cloth_grid(builder, cfg: ClothDropConfig, *, contact_radius_scale: float) -> None:
    spacing = cfg.cloth_particle_radius * SIM_SCALE
    builder.add_cloth_grid(
        pos=wp.vec3(
            cfg.vertical_x_low * SIM_SCALE,
            -0.5 * cfg.cloth_span_y * SIM_SCALE,
            (cfg.vertical_height_low + cfg.cloth_span_x) * SIM_SCALE,
        ),
        rot=wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), math.pi * 0.5),
        vel=wp.vec3(0.0, 0.0, 0.0),
        dim_x=cfg.cloth_xdim - 1,
        dim_y=cfg.cloth_ydim - 1,
        cell_x=spacing,
        cell_y=spacing,
        mass=cfg.mass / float(cfg.cloth_xdim * cfg.cloth_ydim),
        tri_ke=5.0e4 * cfg.cloth_stiff[0],
        tri_ka=5.0e4 * cfg.cloth_stiff[2],
        tri_kd=cfg.cloth_tri_damping,
        edge_ke=7.5e2 * cfg.cloth_stiff[1],
        edge_kd=cfg.cloth_edge_damping,
        particle_radius=spacing * contact_radius_scale,
    )


def add_cloth_mesh(builder, cfg: ClothDropConfig, positions_soft: np.ndarray, *, contact_radius_scale: float) -> None:
    vertices = soft_to_newton_positions(positions_soft) * np.float32(SIM_SCALE)
    builder.add_cloth_mesh(
        pos=wp.vec3(0.0, 0.0, 0.0),
        rot=wp.quat_identity(),
        scale=1.0,
        vel=wp.vec3(0.0, 0.0, 0.0),
        vertices=[wp.vec3(*p) for p in vertices],
        indices=triangle_indices(cfg.cloth_xdim, cfg.cloth_ydim).reshape(-1).tolist(),
        density=cfg.mass / float(cfg.cloth_xdim * cfg.cloth_ydim),
        tri_ke=5.0e4 * cfg.cloth_stiff[0],
        tri_ka=5.0e4 * cfg.cloth_stiff[2],
        tri_kd=cfg.cloth_tri_damping,
        edge_ke=7.5e2 * cfg.cloth_stiff[1],
        edge_kd=cfg.cloth_edge_damping,
        particle_radius=cfg.cloth_particle_radius * SIM_SCALE * contact_radius_scale,
    )


def add_ground(builder, cfg: ClothDropConfig) -> None:
    plane_half = 0.5 * cfg.plane_size * SIM_SCALE
    plane_half_z = 0.5 * cfg.plane_thickness * SIM_SCALE
    shape_cfg = shape_contact_config(builder, cfg)
    builder.add_shape_box(
        -1,
        wp.transform(wp.vec3(0.0, 0.0, -plane_half_z), wp.quat_identity()),
        hx=plane_half,
        hy=plane_half,
        hz=plane_half_z,
        cfg=shape_cfg,
    )


def add_obstacle(builder, cfg: ClothDropConfig) -> None:
    obstacle = cfg.obstacle
    if obstacle is None:
        return
    pos = soft_vec_to_newton(obstacle.shape_pos, scale=SIM_SCALE)
    size = tuple(float(v) * SIM_SCALE for v in obstacle.shape_size)
    quat = soft_yaw_to_newton_quat(obstacle.rot_angle)

    if obstacle.env_shape == "platform":
        add_box(builder, cfg, pos, size, quat)
        return
    if obstacle.env_shape == "sphere":
        builder.add_shape_sphere(-1, wp.transform(pos, wp.quat_identity()), radius=size[0], cfg=shape_contact_config(builder, cfg))
        return
    if obstacle.env_shape == "rod":
        rod_quat = horizontal_rod_quat(obstacle.rot_angle)
        builder.add_shape_capsule(-1, wp.transform(pos, rod_quat), radius=size[0], half_height=size[1], cfg=shape_contact_config(builder, cfg))
        return
    if obstacle.env_shape == "table":
        add_box(builder, cfg, pos, size, quat)
        add_table_legs(builder, cfg, obstacle.shape_pos, obstacle.rot_angle)
        return
    raise ValueError(f"Unknown env_shape: {obstacle.env_shape}")


def add_box(builder, cfg: ClothDropConfig, pos: wp.vec3, size: tuple[float, ...], quat: wp.quat) -> None:
    hx = size[0]
    hy = size[1] if len(size) > 1 else size[0]
    hz = size[2] if len(size) > 2 else size[0]
    builder.add_shape_box(-1, wp.transform(pos, quat), hx=hx, hy=hy, hz=hz, cfg=shape_contact_config(builder, cfg))


def add_table_legs(builder, cfg: ClothDropConfig, table_pos_soft: tuple[float, float, float], rot_angle: float) -> None:
    leg_radius = 0.01 * SIM_SCALE
    leg_half_height = max(0.01 * SIM_SCALE, table_pos_soft[2] * SIM_SCALE * 0.5)
    for sx in (-0.08, 0.08):
        for sy in (-0.08, 0.08):
            leg_soft = np.asarray(table_pos_soft, dtype=np.float32) + np.array([sx, sy, -table_pos_soft[2] * 0.5], dtype=np.float32)
            leg_pos = soft_vec_to_newton(leg_soft, scale=SIM_SCALE)
            builder.add_shape_capsule(
                -1,
                wp.transform(leg_pos, wp.quat_identity()),
                radius=leg_radius,
                half_height=leg_half_height,
                cfg=shape_contact_config(builder, cfg),
            )


def vertical_positions_for_config(cfg: ClothDropConfig) -> np.ndarray:
    return vertical_positions(cfg)
