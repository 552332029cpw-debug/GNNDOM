# SPDX-License-Identifier: BSD-3-Clause
"""EdgeGNN checkpoint inference for visible graph mesh-edge prediction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .dataset import radius_edges
from .model import EdgeGNN


@dataclass
class EdgeGNNMeshPredictor:
    """Loads an EdgeGNN checkpoint and predicts mesh edges on a pointcloud."""

    checkpoint_path: Path
    device: str = "cpu"
    neighbor_radius: float = 0.045

    def __post_init__(self) -> None:
        import torch

        self.torch = torch
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        cfg = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
        state_dim = int(cfg.get("state_dim", 3))
        relation_dim = int(cfg.get("relation_dim", 4))
        global_size = int(cfg.get("global_size", 128))
        proc_layer = int(cfg.get("proc_layer", 10))
        self.model = EdgeGNN(
            state_dim=state_dim,
            relation_dim=relation_dim,
            hidden_dim=global_size,
            proc_layer=proc_layer,
            global_size=global_size,
        ).to(self.device)
        state = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        self.model.load_state_dict(state)
        self.model.eval()

    def predict_mesh_edges(self, pointcloud: np.ndarray, *, threshold: float = 0.5) -> np.ndarray:
        pointcloud = np.asarray(pointcloud, dtype=np.float32)
        if len(pointcloud) == 0:
            return np.empty((2, 0), dtype=np.int64)
        normalized = pointcloud - pointcloud.mean(axis=0, keepdims=True)
        edge_index, edge_attr = radius_edges(normalized, self.neighbor_radius)
        if edge_index.shape[1] == 0:
            return np.empty((2, 0), dtype=np.int64)

        torch = self.torch
        with torch.no_grad():
            x = torch.as_tensor(normalized, dtype=torch.float32, device=self.device)
            edge_index_t = torch.as_tensor(edge_index, dtype=torch.long, device=self.device)
            edge_attr_t = torch.as_tensor(edge_attr, dtype=torch.float32, device=self.device)
            graph_index = torch.zeros((len(pointcloud),), dtype=torch.long, device=self.device)
            logits = self.model(x, edge_index_t, edge_attr_t, graph_index)["mesh_edge"]
            keep = (torch.sigmoid(logits).reshape(-1) >= float(threshold)).detach().cpu().numpy()
        if not np.any(keep):
            return np.empty((2, 0), dtype=np.int64)
        return edge_index[:, keep].astype(np.int64)
