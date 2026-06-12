# SPDX-License-Identifier: BSD-3-Clause
"""ClothDrop environment configuration for GNNDOM."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal


EnvShape = Literal["platform", "sphere", "rod", "table"]
TargetType = Literal["flat", "fold"]


@dataclass(frozen=True)
class ObstacleConfig:
    """Obstacle parameters in GNNDOM coordinates.

    Coordinates are stored as x/y horizontal and z height. Quaternions use xyzw
    order. Box shape sizes are stored as x/y/z half extents.
    """

    env_shape: EnvShape
    shape_size: tuple[float, ...]
    shape_pos: tuple[float, float, float]
    shape_quat: tuple[float, float, float, float]
    rot_angle: float


@dataclass(frozen=True)
class ClothDropConfig:
    """A generated ClothDrop scene config mirroring ManiFabric's config dict."""

    cloth_particle_radius: float = 0.00625
    cloth_size: tuple[int, int] = (48, 48)
    cloth_stiff: tuple[float, float, float] = (0.9, 1.0, 0.9)
    mass: float = 0.1
    target_type: TargetType = "flat"
    x_target: float = 0.1
    rot_angle: float = 0.0
    obstacle: ObstacleConfig | None = None
    target_height: float = 0.005
    vertical_x_low: float = 0.0
    vertical_height_low: float = 0.125
    plane_size: float = 2.0
    plane_thickness: float = 0.02
    picker_radius: float = 0.05
    contact_mu: float = 0.45
    cloth_tri_damping: float = 1.0e-3
    cloth_edge_damping: float = 5.0e-2

    def __post_init__(self) -> None:
        xdim, ydim = self.cloth_size
        if xdim < 2 or ydim < 2:
            raise ValueError("cloth_size dimensions must both be at least 2.")
        if len(self.cloth_stiff) != 3:
            raise ValueError("cloth_stiff must contain stretch, bend, and shear.")
        if self.target_type not in {"flat", "fold"}:
            raise ValueError("target_type must be flat or fold.")

    @property
    def cloth_xdim(self) -> int:
        return int(self.cloth_size[0])

    @property
    def cloth_ydim(self) -> int:
        return int(self.cloth_size[1])

    @property
    def env_shape(self) -> str | None:
        return None if self.obstacle is None else self.obstacle.env_shape

    @property
    def shape_size(self) -> tuple[float, ...] | None:
        return None if self.obstacle is None else self.obstacle.shape_size

    @property
    def shape_pos(self) -> tuple[float, float, float] | None:
        return None if self.obstacle is None else self.obstacle.shape_pos

    @property
    def shape_quat(self) -> tuple[float, float, float, float] | None:
        return None if self.obstacle is None else self.obstacle.shape_quat

    @property
    def cloth_span_x(self) -> float:
        return (self.cloth_xdim - 1) * self.cloth_particle_radius

    @property
    def cloth_span_y(self) -> float:
        return (self.cloth_ydim - 1) * self.cloth_particle_radius

    def with_target_type(self, target_type: TargetType) -> "ClothDropConfig":
        return replace(self, target_type=target_type)

    def to_manifabric_dict(self) -> dict:
        data = {
            "ClothPos": [0, 0, 0],
            "ClothSize": [self.cloth_xdim, self.cloth_ydim],
            "ClothStiff": list(self.cloth_stiff),
            "mass": self.mass,
            "x_target": self.x_target,
            "rot_angle": self.rot_angle,
            "env_shape": self.env_shape,
        }
        if self.obstacle is not None:
            data.update(
                {
                    "shape_size": self.obstacle.shape_size,
                    "shape_pos": self.obstacle.shape_pos,
                    "shape_quat": self.obstacle.shape_quat,
                }
            )
        return data


@dataclass(frozen=True)
class ClothDropRuntimeConfig:
    """Newton runtime options separate from the SoftGym-style scene config."""

    device: str | None = "cuda:0"
    fps: int = 60
    substeps: int = 8
    iterations: int = 8
    self_contact: bool = False
    contact_radius_scale: float = 0.5
    settle_steps: int = 420
    velocity_threshold: float = 0.03
    min_stable_steps: int = 100
