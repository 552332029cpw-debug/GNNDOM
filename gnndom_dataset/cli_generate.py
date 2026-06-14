# SPDX-License-Identifier: BSD-3-Clause
"""Command line entrypoint for GNNDOM rollout dataset generation."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gnndom_env import ClothDropConfig, ClothDropRuntimeConfig, ManiFabricClothDropSampler
from gnndom_dataset.collector import DataCollector, DatasetGenerationConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate ManiFabric-style GNNDOM rollout dataset.")
    parser.add_argument("--dataf", type=Path, required=True)
    parser.add_argument("--n-rollout", type=int, default=2)
    parser.add_argument("--train-valid-ratio", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--config-id-start", type=int, default=0)
    parser.add_argument("--cloth-xdim", type=int, default=48)
    parser.add_argument("--cloth-ydim", type=int, default=48)
    parser.add_argument("--cloth-particle-radius", type=float, default=0.00625)
    parser.add_argument("--cloth-mass", type=float, default=0.1)
    parser.add_argument("--cloth-stiffness", type=float, nargs=3, default=(0.9, 1.0, 0.9))
    parser.add_argument("--target-type", choices=("flat", "fold", "random"), default="flat")
    parser.add_argument("--vary-cloth-size", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--vary-stiffness", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--vary-mass", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--vary-orientation", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--env-shape", choices=("None", "platform", "sphere", "rod", "table", "random", "all"), default="None")
    parser.add_argument("--down-sample-scale", type=int, default=3)
    parser.add_argument("--voxel-size", type=float, default=0.0216)
    parser.add_argument("--observation-mode", choices=("isaac_camera", "full", "geometry_camera"), default="isaac_camera")
    parser.add_argument("--save-rgbd", action="store_true")
    parser.add_argument("--camera-width", type=int, default=360)
    parser.add_argument("--camera-height", type=int, default=360)
    parser.add_argument("--camera-fov", type=float, default=45.0)
    parser.add_argument("--camera-pos", type=float, nargs=3, default=(1.2, 0.0, 0.7))
    parser.add_argument("--camera-look-at", type=float, nargs=3, default=(0.0, 0.0, 0.2))
    parser.add_argument("--camera-near", type=float, default=0.01)
    parser.add_argument("--camera-far", type=float, default=5.0)
    parser.add_argument("--min-visible-points", type=int, default=4)
    parser.add_argument("--visibility-threshold", type=float, default=0.0216)
    parser.add_argument("--swing-acc", type=float, default=2.0)
    parser.add_argument("--pull-acc", type=float, default=1.0)
    parser.add_argument("--drop-steps", type=int, default=30)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--substeps", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--settle-steps", type=int, default=2)
    parser.add_argument("--velocity-threshold", type=float, default=0.03)
    parser.add_argument("--self-contact", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_cfg = ClothDropConfig(
        cloth_particle_radius=args.cloth_particle_radius,
        cloth_size=(args.cloth_xdim, args.cloth_ydim),
        cloth_stiff=tuple(args.cloth_stiffness),
        mass=args.cloth_mass,
        target_type="flat" if args.target_type == "random" else args.target_type,
    )
    dataset_cfg = DatasetGenerationConfig(
        dataf=args.dataf,
        n_rollout=args.n_rollout,
        train_valid_ratio=args.train_valid_ratio,
        dt=1.0 / float(args.fps),
        down_sample_scale=args.down_sample_scale,
        voxel_size=args.voxel_size,
        swing_acc=args.swing_acc,
        pull_acc=args.pull_acc,
        drop_steps=args.drop_steps,
        seed=args.seed,
        observation_mode=args.observation_mode,
        save_rgbd=args.save_rgbd,
        camera_width=args.camera_width,
        camera_height=args.camera_height,
        camera_fov=args.camera_fov,
        camera_pos=tuple(args.camera_pos),
        camera_look_at=tuple(args.camera_look_at),
        camera_near=args.camera_near,
        camera_far=args.camera_far,
        min_visible_points=args.min_visible_points,
        visibility_threshold=args.visibility_threshold,
    )
    runtime = ClothDropRuntimeConfig(
        device=args.device,
        fps=args.fps,
        substeps=args.substeps,
        iterations=args.iterations,
        self_contact=args.self_contact,
        settle_steps=args.settle_steps,
        velocity_threshold=args.velocity_threshold,
        min_stable_steps=0,
    )

    train_count = int(args.n_rollout * args.train_valid_ratio)
    for phase, config_offset in (("train", args.config_id_start), ("valid", args.config_id_start + train_count)):
        sampler = ManiFabricClothDropSampler(
            seed=args.seed + config_offset,
            base_cfg=base_cfg,
            target_type=args.target_type,
            vary_cloth_size=args.vary_cloth_size,
            vary_stiffness=args.vary_stiffness,
            vary_mass=args.vary_mass,
            vary_orientation=args.vary_orientation,
            env_shape=None if args.env_shape == "None" else args.env_shape,
        )
        collector = DataCollector(dataset_cfg, phase=phase, sampler=sampler, runtime=runtime, config_id_start=config_offset)
        collector.gen_dataset()


if __name__ == "__main__":
    main()
