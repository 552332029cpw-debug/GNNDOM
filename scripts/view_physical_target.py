# SPDX-License-Identifier: BSD-3-Clause
"""View the physically settled target together with the obstacle scene."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from gnndom_env import ClothDropConfig, ClothDropRuntimeConfig, ManiFabricClothDropSampler, NewtonClothDropEnv
from gnndom_env.geometry import flat_positions, target_picker_positions, triangle_indices
from gnndom_env.newton_backend import SIM_SCALE, soft_to_newton_positions, wp

import newton.examples  # noqa: E402


class PhysicalTargetViewer:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.args = args
        base = ClothDropConfig(
            cloth_particle_radius=args.cloth_particle_radius,
            cloth_size=(args.cloth_xdim, args.cloth_ydim),
            cloth_stiff=tuple(args.cloth_stiffness),
            mass=args.cloth_mass,
            target_type="flat" if args.target_type == "random" else args.target_type,
        )
        sampler = ManiFabricClothDropSampler(
            seed=args.seed,
            base_cfg=base,
            target_type=args.target_type,
            vary_cloth_size=args.vary_cloth_size,
            vary_stiffness=args.vary_stiffness,
            vary_mass=args.vary_mass,
            vary_orientation=args.vary_orientation,
            env_shape=None if args.env_shape == "None" else args.env_shape,
        )
        self.cfg = sampler.sample(args.config_id)
        self.runtime = ClothDropRuntimeConfig(
            device=args.device,
            fps=args.fps,
            substeps=args.substeps,
            iterations=args.iterations,
            self_contact=args.self_contact,
            settle_steps=args.settle_steps,
            velocity_threshold=args.velocity_threshold,
            min_stable_steps=args.min_stable_steps,
        )

        self.env = NewtonClothDropEnv(self.cfg, self.runtime)
        self.env.reset_to_target(grasp=False)
        steps = self.env.step_until_stable()
        target_pos = self.env.current_positions().astype(np.float32)
        geometric_target_pos = flat_positions(self.cfg).astype(np.float32)

        self.geometric_points = wp.array(
            soft_to_newton_positions(geometric_target_pos) * np.float32(SIM_SCALE),
            dtype=wp.vec3,
            device=self.env.model.device,
        )
        self.target_indices = wp.array(
            triangle_indices(self.cfg.cloth_xdim, self.cfg.cloth_ydim).reshape(-1),
            dtype=wp.int32,
            device=self.env.model.device,
        )
        self.target_picker_points = wp.array(
            soft_to_newton_positions(target_picker_positions(self.cfg)) * np.float32(SIM_SCALE),
            dtype=wp.vec3,
            device=self.env.model.device,
        )
        self.target_picker_radii = wp.array([0.9, 0.9], dtype=wp.float32, device=self.env.model.device)
        self.target_picker_colors = wp.array(
            [wp.vec3(0.1, 1.0, 0.9), wp.vec3(0.1, 1.0, 0.9)],
            dtype=wp.vec3,
            device=self.env.model.device,
        )
        self.sim_time = 0.0

        displacement = np.linalg.norm(target_pos - geometric_target_pos, axis=1)
        print(
            "[INFO] physical target "
            f"env_shape={self.cfg.env_shape} target_type={self.cfg.target_type} "
            f"settle_steps={steps} "
            f"mean_geometric_to_physical={float(np.mean(displacement)):.6f} "
            f"max_geometric_to_physical={float(np.max(displacement)):.6f}"
        )
        if self.cfg.obstacle is not None:
            print(f"[INFO] obstacle size={self.cfg.obstacle.shape_size} pos={self.cfg.obstacle.shape_pos}")

        self.viewer.set_model(self.env.model)
        self.viewer.set_camera(wp.vec3(35.0, -85.0, 45.0), -68.0, -30.0)

    def step(self) -> None:
        self.sim_time += self.env.frame_dt

    def render(self) -> None:
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.env.state_0)
        if self.args.show_geometric_target:
            self.viewer.log_mesh(
                "/physical_target/geometric_target_before_settle",
                self.geometric_points,
                self.target_indices,
                color=(1.0, 0.72, 0.1),
                roughness=0.9,
                backface_culling=False,
            )
        self.viewer.log_points(
            "/physical_target/target_picker_positions",
            self.target_picker_points,
            self.target_picker_radii,
            self.target_picker_colors,
        )
        self.viewer.end_frame()


def main() -> None:
    parser = newton.examples.create_parser()
    parser.set_defaults(viewer="gl", start_paused=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--config-id", type=int, default=0)
    parser.add_argument("--cloth-xdim", type=int, default=48)
    parser.add_argument("--cloth-ydim", type=int, default=48)
    parser.add_argument("--cloth-particle-radius", type=float, default=0.00625)
    parser.add_argument("--cloth-mass", type=float, default=0.1)
    parser.add_argument("--cloth-stiffness", type=float, nargs=3, default=(0.9, 1.0, 0.9))
    parser.add_argument("--target-type", choices=("flat", "fold", "random"), default="flat")
    parser.add_argument("--env-shape", choices=("None", "platform", "sphere", "rod", "table", "random", "all"), default="sphere")
    parser.add_argument("--vary-cloth-size", action="store_true")
    parser.add_argument("--vary-stiffness", action="store_true")
    parser.add_argument("--vary-mass", action="store_true")
    parser.add_argument("--vary-orientation", action="store_true")
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--substeps", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--settle-steps", type=int, default=420)
    parser.add_argument("--min-stable-steps", type=int, default=100)
    parser.add_argument("--velocity-threshold", type=float, default=0.03)
    parser.add_argument("--self-contact", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--show-geometric-target", action=argparse.BooleanOptionalAction, default=True)
    viewer, args = newton.examples.init(parser)
    newton.examples.run(PhysicalTargetViewer(viewer, args), args)


if __name__ == "__main__":
    main()
