# SPDX-License-Identifier: BSD-3-Clause
"""ManiFabric ClothDrop domain-randomization sampler."""

from __future__ import annotations

from dataclasses import replace
import math
from typing import Literal

import numpy as np

from .config import ClothDropConfig, EnvShape, TargetType
from .geometry import BASE_TARGET_HEIGHT, make_obstacle, target_height_offset


MANIFABRIC_CLOTH_SIZE_RANGE = (32, 48)
MANIFABRIC_STIFFNESS_RANGE = (0.5, 2.0)
MANIFABRIC_MASS_RANGE = (0.02, 0.2)
MANIFABRIC_ROT_RANGE = (-math.pi / 6.0, math.pi / 6.0)
MANIFABRIC_TARGET_X_RANGE = (0.05, 0.15)
EnvShapeMode = Literal["None", "platform", "sphere", "rod", "table", "random", "all"] | None
TargetTypeMode = Literal["flat", "fold", "random"]


class ManiFabricClothDropSampler:
    """Sample configs with the same families used by ManiFabric ClothDropEnv."""

    def __init__(
        self,
        *,
        seed: int = 43,
        base_cfg: ClothDropConfig | None = None,
        target_type: TargetTypeMode = "flat",
        vary_cloth_size: bool = False,
        vary_stiffness: bool = False,
        vary_mass: bool = False,
        vary_orientation: bool = False,
        env_shape: EnvShapeMode = None,
    ):
        self.rng = np.random.default_rng(seed)
        self.base_cfg = base_cfg or ClothDropConfig()
        self.target_type = target_type
        self.vary_cloth_size = vary_cloth_size
        self.vary_stiffness = vary_stiffness
        self.vary_mass = vary_mass
        self.vary_orientation = vary_orientation
        self.env_shape = env_shape

    def sample(self, config_id: int) -> ClothDropConfig:
        cfg = self.base_cfg
        xdim, ydim = cfg.cloth_size
        if self.vary_cloth_size:
            xdim = int(self.rng.integers(*MANIFABRIC_CLOTH_SIZE_RANGE))
            ydim = int(self.rng.integers(*MANIFABRIC_CLOTH_SIZE_RANGE))

        mass = cfg.mass
        if self.vary_mass:
            mass = float(self.rng.uniform(*MANIFABRIC_MASS_RANGE))

        cloth_stiff = cfg.cloth_stiff
        if self.vary_stiffness:
            cloth_stiff = tuple(float(v) for v in self.rng.uniform(*MANIFABRIC_STIFFNESS_RANGE, size=3))

        target_type = self._sample_target_type()
        rot_angle = float(self.rng.uniform(*MANIFABRIC_ROT_RANGE)) if self.vary_orientation else cfg.rot_angle
        x_target = float(self.rng.uniform(*MANIFABRIC_TARGET_X_RANGE))
        env_shape = self._select_env_shape(config_id)
        obstacle = make_obstacle(env_shape, x_target, xdim, cfg.cloth_particle_radius, rot_angle)
        vertical_height_low = float(self.rng.uniform(0.1, 0.15))
        if env_shape == "rod":
            vertical_height_low += 0.1

        return replace(
            cfg,
            cloth_size=(xdim, ydim),
            cloth_stiff=cloth_stiff,
            mass=mass,
            target_type=target_type,
            x_target=x_target,
            rot_angle=rot_angle,
            obstacle=obstacle,
            target_height=BASE_TARGET_HEIGHT + target_height_offset(obstacle),
            vertical_height_low=vertical_height_low,
        )

    def _sample_target_type(self) -> TargetType:
        if self.target_type == "random":
            return str(self.rng.choice(np.asarray(["flat", "fold"], dtype=object)))  # type: ignore[return-value]
        return self.target_type

    def _select_env_shape(self, config_id: int) -> EnvShape | None:
        if self.env_shape in {None, "None"}:
            return None
        if self.env_shape == "random":
            return (None, "platform", "sphere", "rod")[config_id % 4]  # type: ignore[return-value]
        if self.env_shape == "all":
            return (None, "platform", "sphere", "rod", "table")[config_id % 5]  # type: ignore[return-value]
        return self.env_shape  # type: ignore[return-value]

