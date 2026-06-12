# SPDX-License-Identifier: BSD-3-Clause
"""Smoke check for the Newton-backed ManiFabric ClothDrop environment."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from gnndom_env import ClothDropConfig, ClothDropRuntimeConfig, ManiFabricClothDropSampler, NewtonClothDropEnv
from gnndom_env.geometry import flat_positions, vertical_positions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test the GNNDOM ClothDrop environment.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--config-id", type=int, default=0)
    parser.add_argument("--cloth-xdim", type=int, default=4)
    parser.add_argument("--cloth-ydim", type=int, default=4)
    parser.add_argument("--target-type", choices=("flat", "fold", "random"), default="flat")
    parser.add_argument("--env-shape", choices=("None", "platform", "sphere", "rod", "table", "random", "all"), default="None")
    parser.add_argument("--vary-cloth-size", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--vary-stiffness", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--vary-mass", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--vary-orientation", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--substeps", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--settle-steps", type=int, default=2)
    parser.add_argument("--velocity-threshold", type=float, default=0.03)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base = ClothDropConfig(cloth_size=(args.cloth_xdim, args.cloth_ydim), target_type="flat")
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
    cfg = sampler.sample(args.config_id)
    runtime = ClothDropRuntimeConfig(
        device=args.device,
        fps=args.fps,
        substeps=args.substeps,
        iterations=args.iterations,
        settle_steps=args.settle_steps,
        velocity_threshold=args.velocity_threshold,
        min_stable_steps=0,
    )
    env = NewtonClothDropEnv(cfg, runtime)
    env.setup(initial="vertical")
    steps = env.step_until_stable()
    positions = env.current_positions()
    pickers = env.current_picker_positions()
    current_config = env.get_current_config()

    expected_shape = (cfg.cloth_xdim * cfg.cloth_ydim, 3)
    if positions.shape != expected_shape:
        raise RuntimeError(f"positions shape {positions.shape} != {expected_shape}")
    if flat_positions(cfg).shape != expected_shape:
        raise RuntimeError("flat target shape mismatch")
    if vertical_positions(cfg).shape != expected_shape:
        raise RuntimeError("vertical target shape mismatch")
    if pickers.shape != (2, 3):
        raise RuntimeError(f"picker shape {pickers.shape} != (2, 3)")
    if "target_pos" not in current_config or "target_picker_pos" not in current_config:
        raise RuntimeError("current config is missing target fields")

    print(
        "[SMOKE] ok "
        f"cloth={cfg.cloth_xdim}x{cfg.cloth_ydim} "
        f"target_type={cfg.target_type} env_shape={cfg.env_shape} "
        f"mass={cfg.mass:.4f} stiff={tuple(round(v, 4) for v in cfg.cloth_stiff)} "
        f"steps={steps} positions={positions.shape} pickers={pickers.shape}"
    )
    print(f"[SMOKE] target_x={cfg.x_target:.6f} rot_angle={cfg.rot_angle:.6f}")
    if cfg.obstacle is not None:
        print(f"[SMOKE] obstacle size={cfg.obstacle.shape_size} pos={cfg.obstacle.shape_pos}")


if __name__ == "__main__":
    main()

