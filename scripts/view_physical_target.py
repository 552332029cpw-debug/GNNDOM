# SPDX-License-Identifier: BSD-3-Clause
"""View the target release-and-settle process together with the obstacle scene."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from gnndom_env import ClothDropConfig, ClothDropRuntimeConfig, ManiFabricClothDropSampler, NewtonClothDropEnv
from gnndom_env.geometry import geometric_target_positions, target_picker_positions, triangle_indices
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
            contact_ke=args.contact_ke,
            contact_kd=args.contact_kd,
            contact_mu=args.contact_mu,
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
            x_target=args.x_target,
            rot_angle=args.rot_angle,
            shape_size=None if args.shape_size is None else tuple(args.shape_size),
            shape_pos=None if args.shape_pos is None else tuple(args.shape_pos),
        )
        self.cfg = sampler.sample(args.config_id)
        self.runtime = ClothDropRuntimeConfig(
            device=args.device,
            fps=args.fps,
            substeps=args.substeps,
            iterations=args.iterations,
            self_contact=args.self_contact,
            air_drag=args.air_drag,
            settle_steps=args.settle_steps,
            velocity_threshold=args.velocity_threshold,
            min_stable_steps=args.min_stable_steps,
        )

        self.env = NewtonClothDropEnv(self.cfg, self.runtime)
        self.env.reset_to_target(grasp=True)
        geometric_target_pos = geometric_target_positions(self.cfg).astype(np.float32)
        self.release_requested = not args.start_paused
        self.released = False
        self.steps = 0
        self._reported_stable = False

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

        print(
            "[INFO] target release viewer "
            f"env_shape={self.cfg.env_shape} target_type={self.cfg.target_type} "
            f"contact_ke={self.cfg.contact_ke:.1f} contact_mu={self.cfg.contact_mu:.3f} "
            f"initial_grasp={int(np.max(self.env.picker_grasp_np))} "
            "press Space to release pickers and simulate settling"
        )
        if self.cfg.obstacle is not None:
            print(f"[INFO] obstacle size={self.cfg.obstacle.shape_size} pos={self.cfg.obstacle.shape_pos}")

        self.viewer.set_model(self.env.model)
        self.viewer.set_camera(wp.vec3(35.0, -85.0, 45.0), -68.0, -30.0)
        if hasattr(self.viewer, "renderer") and hasattr(self.viewer.renderer, "register_key_press"):
            self.viewer.renderer.register_key_press(self._on_key_press)

    def _on_key_press(self, symbol: int, modifiers: int) -> None:
        try:
            import pyglet
        except Exception:
            return
        if symbol == pyglet.window.key.SPACE:
            self.release_requested = True

    def step(self) -> None:
        if not self.release_requested:
            return
        if not self.released:
            self.env.set_picker_grasp(np.zeros(2, dtype=np.int32))
            self.released = True
            print(f"[INFO] released pickers release_grasp={int(np.max(self.env.picker_grasp_np))}")
        self.env.step()
        self.steps += 1
        if not self._reported_stable and self.steps >= self.runtime.min_stable_steps:
            speeds = np.linalg.norm(self.env.current_velocities(), axis=1)
            if float(np.max(speeds)) < float(self.runtime.velocity_threshold):
                target_pos = self.env.current_positions().astype(np.float32)
                geometric_target_pos = geometric_target_positions(self.cfg).astype(np.float32)
                displacement = np.linalg.norm(target_pos - geometric_target_pos, axis=1)
                print(
                    "[INFO] settled target "
                    f"settle_steps={self.steps} "
                    f"release_grasp={int(np.max(self.env.picker_grasp_np))} "
                    f"mean_geometric_to_physical={float(np.mean(displacement)):.6f} "
                    f"max_geometric_to_physical={float(np.max(displacement)):.6f}"
                )
                self._reported_stable = True
        self.sim_time += self.env.frame_dt

    def render(self) -> None:
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.env.state_0)
        if self.args.show_geometric_target:
            self.viewer.log_mesh(
                "/physical_target/geometric_target_before_settle",
                self.geometric_points,
                self.target_indices,
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
    parser.set_defaults(viewer="gl", device="cpu")
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--config-id", type=int, default=0)
    parser.add_argument("--cloth-xdim", type=int, default=48)
    parser.add_argument("--cloth-ydim", type=int, default=48)
    parser.add_argument("--cloth-particle-radius", type=float, default=0.00625)
    parser.add_argument("--cloth-mass", type=float, default=0.1)
    parser.add_argument("--cloth-stiffness", type=float, nargs=3, default=(0.9, 1.0, 0.9))
    parser.add_argument("--contact-ke", type=float, default=1.0e5)
    parser.add_argument("--contact-kd", type=float, default=1.0e-2)
    parser.add_argument("--contact-mu", type=float, default=2.0)
    parser.add_argument("--target-type", choices=("flat", "fold", "random"), default="flat")
    parser.add_argument("--x-target", type=float, default=None)
    parser.add_argument("--rot-angle", type=float, default=None)
    parser.add_argument("--env-shape", choices=("None", "platform", "sphere", "rod", "table", "random", "all"), default="sphere")
    parser.add_argument("--shape-size", type=float, nargs="+", default=None)
    parser.add_argument("--shape-pos", type=float, nargs=3, default=None)
    parser.add_argument("--vary-cloth-size", action="store_true")
    parser.add_argument("--vary-stiffness", action="store_true")
    parser.add_argument("--vary-mass", action="store_true")
    parser.add_argument("--vary-orientation", action="store_true")
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--substeps", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--air-drag", type=float, default=0.0)
    parser.add_argument("--settle-steps", type=int, default=420)
    parser.add_argument("--min-stable-steps", type=int, default=100)
    parser.add_argument("--velocity-threshold", type=float, default=0.03)
    parser.add_argument("--self-contact", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--show-geometric-target", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--start-paused", action=argparse.BooleanOptionalAction, default=True)
    viewer, args = newton.examples.init(parser)
    example = PhysicalTargetViewer(viewer, args)
    if args.start_paused and hasattr(viewer, "_paused"):
        viewer._paused = True
    newton.examples.run(example, args)


if __name__ == "__main__":
    main()
