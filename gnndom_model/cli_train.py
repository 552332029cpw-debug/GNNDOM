# SPDX-License-Identifier: BSD-3-Clause
"""Command line entrypoint for training GNNDOM Dynamic GNN models."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gnndom_model.trainer import DynamicTrainConfig, DynamicTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train GNNDOM Dynamic GNN from graph transition data.")
    parser.add_argument("--graphf", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--train-mode", choices=("vsbl", "full", "graph_imit"), default="vsbl")
    parser.add_argument("--output-type", choices=("vel", "accel"), default="vel")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--vsbl-lr", type=float, default=None)
    parser.add_argument("--full-lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--state-dim", type=int, default=21)
    parser.add_argument("--relation-dim", type=int, default=6)
    parser.add_argument("--global-size", type=int, default=128)
    parser.add_argument("--proc-layer", type=int, default=10)
    parser.add_argument("--reward-w", type=float, default=1.0e5)
    parser.add_argument("--imit-w", type=float, default=5.0)
    parser.add_argument("--imit-w-lat", type=float, default=1.0)
    parser.add_argument("--use-reward", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tune-teach", action="store_true")
    parser.add_argument("--copy-teach", nargs="*", default=[], choices=("encoder", "decoder", "processor"))
    parser.add_argument("--full-dyn-path", type=Path, default=None)
    parser.add_argument("--dt", type=float, default=1.0 / 60.0)
    parser.add_argument("--pred-time-interval", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-train-graphs", type=int, default=None)
    parser.add_argument("--max-valid-graphs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--min-delta", type=float, default=1.0e-5)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = DynamicTrainConfig(
        graphf=args.graphf,
        out_dir=args.out_dir,
        train_mode=args.train_mode,
        output_type=args.output_type,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        vsbl_lr=args.vsbl_lr,
        full_lr=args.full_lr,
        weight_decay=args.weight_decay,
        device=args.device,
        state_dim=args.state_dim,
        relation_dim=args.relation_dim,
        global_size=args.global_size,
        proc_layer=args.proc_layer,
        reward_w=args.reward_w,
        imit_w=args.imit_w,
        imit_w_lat=args.imit_w_lat,
        use_reward=args.use_reward,
        tune_teach=args.tune_teach,
        copy_teach=args.copy_teach,
        full_dyn_path=args.full_dyn_path,
        dt=args.dt,
        pred_time_interval=args.pred_time_interval,
        num_workers=args.num_workers,
        max_train_graphs=args.max_train_graphs,
        max_valid_graphs=args.max_valid_graphs,
        patience=args.patience,
        min_delta=args.min_delta,
        seed=args.seed,
    )
    result = DynamicTrainer(cfg).train()
    print(f"[INFO] best_valid_loss={result['best_valid_loss']:.6g}")


if __name__ == "__main__":
    main()
