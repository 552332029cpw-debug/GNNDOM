# SPDX-License-Identifier: BSD-3-Clause
"""Command line entrypoint for training GNNDOM EdgeGNN."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gnndom_edge.trainer import EdgeTrainConfig, EdgeTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train EdgeGNN to classify cloth mesh edges from pointcloud radius edges.")
    parser.add_argument("--graphf", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--neighbor-radius", type=float, default=0.045)
    parser.add_argument("--state-dim", type=int, default=3)
    parser.add_argument("--relation-dim", type=int, default=4)
    parser.add_argument("--global-size", type=int, default=128)
    parser.add_argument("--proc-layer", type=int, default=10)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-train-graphs", type=int, default=None)
    parser.add_argument("--max-valid-graphs", type=int, default=None)
    parser.add_argument("--edge-model-path", type=Path, default=None)
    parser.add_argument("--load-optim", action="store_true")
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--min-delta", type=float, default=1.0e-5)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = EdgeTrainConfig(
        graphf=args.graphf,
        out_dir=args.out_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        beta1=args.beta1,
        device=args.device,
        neighbor_radius=args.neighbor_radius,
        state_dim=args.state_dim,
        relation_dim=args.relation_dim,
        global_size=args.global_size,
        proc_layer=args.proc_layer,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        max_train_graphs=args.max_train_graphs,
        max_valid_graphs=args.max_valid_graphs,
        edge_model_path=args.edge_model_path,
        load_optim=args.load_optim,
        patience=args.patience,
        min_delta=args.min_delta,
        seed=args.seed,
    )
    result = EdgeTrainer(cfg).train()
    print(f"[INFO] best_valid_loss={result['best_valid_loss']:.6g}")


if __name__ == "__main__":
    main()
