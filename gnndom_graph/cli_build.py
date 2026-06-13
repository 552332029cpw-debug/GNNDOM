# SPDX-License-Identifier: BSD-3-Clause
"""Command line entrypoint for building GNNDOM graph transition data."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gnndom_graph.builder import GraphBuildConfig, build_graphs_from_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build graph transitions from GNNDOM rollout dataset.")
    parser.add_argument("--dataf", type=Path, required=True)
    parser.add_argument("--graphf", type=Path, required=True)
    parser.add_argument("--n-his", type=int, default=5)
    parser.add_argument("--pred-time-interval", type=int, default=1)
    parser.add_argument("--dt", type=float, default=1.0 / 60.0)
    parser.add_argument("--neighbor-radius", type=float, default=0.045)
    parser.add_argument("--use-mesh-edge", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-es", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--graph-mode", choices=("full", "vsbl", "both"), default="full")
    parser.add_argument("--edge-model-path", type=Path, default=None, help="Optional EdgeGNN checkpoint for vsbl mesh-edge prediction.")
    parser.add_argument("--edge-threshold", type=float, default=0.5, help="Sigmoid threshold for EdgeGNN mesh-edge logits.")
    parser.add_argument("--edge-device", type=str, default="cpu", help="Device used for EdgeGNN inference.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = GraphBuildConfig(
        dataf=args.dataf,
        graphf=args.graphf,
        n_his=args.n_his,
        pred_time_interval=args.pred_time_interval,
        dt=args.dt,
        neighbor_radius=args.neighbor_radius,
        use_mesh_edge=args.use_mesh_edge,
        use_es=args.use_es,
        graph_mode=args.graph_mode,
        edge_model_path=args.edge_model_path,
        edge_threshold=args.edge_threshold,
        edge_device=args.edge_device,
    )
    saved = build_graphs_from_dataset(cfg)
    print(f"[INFO] saved {len(saved)} graphs to {args.graphf}")


if __name__ == "__main__":
    main()
