# SPDX-License-Identifier: BSD-3-Clause
"""Command line entrypoint for GNNDOM online VSBL planning."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gnndom_plan import PlanConfig, PlanRunner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GNNDOM online VSBL MPC planning.")
    parser.add_argument("--dyn-path", type=Path, required=True)
    parser.add_argument("--edge-model-path", type=Path, default=None)
    parser.add_argument("--observation-mode", choices=("isaac_camera", "geometry_camera"), default="isaac_camera")
    parser.add_argument("--log-dir", type=Path, default=Path("runs/plan"))
    parser.add_argument("--configurations", type=int, default=1)
    parser.add_argument("--sampling-num", type=int, default=10)
    parser.add_argument("--shooting-number", type=int, default=9)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--runtime-device", default=None)
    parser.add_argument("--seed", type=int, default=43)

    parser.add_argument("--cloth-xdim", type=int, default=48)
    parser.add_argument("--cloth-ydim", type=int, default=48)
    parser.add_argument("--cloth-particle-radius", type=float, default=0.00625)
    parser.add_argument("--cloth-mass", type=float, default=0.1)
    parser.add_argument("--cloth-stiffness", type=float, nargs=3, default=(0.9, 1.0, 0.9))
    parser.add_argument("--env-shape", default=None, choices=("None", "platform", "sphere", "rod", "table", "random", "all"))
    parser.add_argument("--target-type", default="flat", choices=("flat", "fold", "random"))
    parser.add_argument("--vary-cloth-size", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--vary-stiffness", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--vary-mass", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--vary-orientation", action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument("--dt", type=float, default=1.0 / 60.0)
    parser.add_argument("--pred-time-interval", type=int, default=1)
    parser.add_argument("--n-his", type=int, default=5)
    parser.add_argument("--down-sample-scale", type=int, default=3)
    parser.add_argument("--neighbor-radius", type=float, default=0.045)
    parser.add_argument("--voxel-size", type=float, default=0.0216)
    parser.add_argument("--visibility-threshold", type=float, default=0.05)
    parser.add_argument("--use-mesh-edge", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-es", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--edge-threshold", type=float, default=0.5)

    parser.add_argument("--camera-width", type=int, default=480)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--camera-fov", type=float, default=100.0)
    parser.add_argument("--camera-pos", type=float, nargs=3, default=(1.45, -0.85, 0.95))
    parser.add_argument("--camera-look-at", type=float, nargs=3, default=(0.32, -0.08, 0.22))
    parser.add_argument("--camera-near", type=float, default=0.01)
    parser.add_argument("--camera-far", type=float, default=5.0)
    parser.add_argument("--save-rgbd", action="store_true")

    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--substeps", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--air-drag", type=float, default=0.0)
    parser.add_argument("--settle-steps", type=int, default=420)
    parser.add_argument("--min-stable-steps", type=int, default=100)
    parser.add_argument("--velocity-threshold", type=float, default=0.03)
    parser.add_argument("--swing-acc", type=float, default=2.0)
    parser.add_argument("--pull-acc", type=float, default=1.0)
    parser.add_argument("--drop-steps", type=int, default=3)
    parser.add_argument("--control-steps", type=int, default=None)
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> PlanConfig:
    env_shape = None if args.env_shape in {None, "None"} else args.env_shape
    return PlanConfig(
        dyn_path=args.dyn_path,
        log_dir=args.log_dir,
        edge_model_path=args.edge_model_path,
        observation_mode=args.observation_mode,
        configurations=args.configurations,
        sampling_num=args.sampling_num,
        shooting_number=args.shooting_number,
        seed=args.seed,
        device=args.device,
        cloth_xdim=args.cloth_xdim,
        cloth_ydim=args.cloth_ydim,
        cloth_particle_radius=args.cloth_particle_radius,
        cloth_mass=args.cloth_mass,
        cloth_stiffness=tuple(args.cloth_stiffness),
        env_shape=env_shape,
        target_type=args.target_type,
        vary_cloth_size=args.vary_cloth_size,
        vary_stiffness=args.vary_stiffness,
        vary_mass=args.vary_mass,
        vary_orientation=args.vary_orientation,
        dt=args.dt,
        pred_time_interval=args.pred_time_interval,
        n_his=args.n_his,
        down_sample_scale=args.down_sample_scale,
        neighbor_radius=args.neighbor_radius,
        voxel_size=args.voxel_size,
        visibility_threshold=args.visibility_threshold,
        use_mesh_edge=args.use_mesh_edge,
        use_es=args.use_es,
        edge_threshold=args.edge_threshold,
        camera_width=args.camera_width,
        camera_height=args.camera_height,
        camera_fov=args.camera_fov,
        camera_pos=tuple(args.camera_pos),
        camera_look_at=tuple(args.camera_look_at),
        camera_near=args.camera_near,
        camera_far=args.camera_far,
        save_rgbd=args.save_rgbd,
        runtime_device=args.runtime_device,
        fps=args.fps,
        substeps=args.substeps,
        iterations=args.iterations,
        air_drag=args.air_drag,
        settle_steps=args.settle_steps,
        min_stable_steps=args.min_stable_steps,
        velocity_threshold=args.velocity_threshold,
        swing_acc=args.swing_acc,
        pull_acc=args.pull_acc,
        drop_steps=args.drop_steps,
        control_steps=args.control_steps,
    )


def main() -> None:
    cfg = config_from_args(parse_args())
    result = PlanRunner(cfg).run()
    print(f"[INFO] average_final_reward={result['average_final_reward']:.6g}")


if __name__ == "__main__":
    main()
