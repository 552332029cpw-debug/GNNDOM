# SPDX-License-Identifier: BSD-3-Clause
"""EdgeGNN training utilities for GNNDOM pointcloud mesh-edge prediction."""

from .dataset import EdgeBatch, EdgeGraphDataset, collate_edge_graphs
from .infer import EdgeGNNMeshPredictor
from .model import EdgeGNN
from .trainer import EdgeTrainConfig, EdgeTrainer

__all__ = [
    "EdgeBatch",
    "EdgeGNNMeshPredictor",
    "EdgeGNN",
    "EdgeGraphDataset",
    "EdgeTrainConfig",
    "EdgeTrainer",
    "collate_edge_graphs",
]
