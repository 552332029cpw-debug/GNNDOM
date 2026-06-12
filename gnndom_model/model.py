# SPDX-License-Identifier: BSD-3-Clause
"""Pure PyTorch Dynamic GNN matching the ManiFabric training interface."""

from __future__ import annotations

import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, *, num_layers: int = 2, layer_norm: bool = True):
        super().__init__()
        layers: list[nn.Module] = []
        last_dim = in_dim
        for _ in range(max(0, num_layers - 1)):
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(nn.ReLU())
            last_dim = hidden_dim
        layers.append(nn.Linear(last_dim, out_dim))
        if layer_norm:
            layers.append(nn.LayerNorm(out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GNBlock(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.edge_mlp = MLP(hidden_dim * 3, hidden_dim, hidden_dim)
        self.node_mlp = MLP(hidden_dim * 2, hidden_dim, hidden_dim)

    def forward(self, node_lat: torch.Tensor, edge_index: torch.Tensor, edge_lat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        src, dst = edge_index
        edge_input = torch.cat([node_lat[src], node_lat[dst], edge_lat], dim=-1)
        edge_update = self.edge_mlp(edge_input)
        edge_lat = edge_lat + edge_update

        agg = torch.zeros_like(node_lat)
        agg.index_add_(0, dst, edge_lat)
        count = torch.zeros((node_lat.shape[0], 1), dtype=node_lat.dtype, device=node_lat.device)
        ones = torch.ones((edge_lat.shape[0], 1), dtype=node_lat.dtype, device=node_lat.device)
        count.index_add_(0, dst, ones)
        agg = agg / count.clamp_min(1.0)

        node_update = self.node_mlp(torch.cat([node_lat, agg], dim=-1))
        node_lat = node_lat + node_update
        return node_lat, edge_lat


class RewardModel(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, graph_lat: torch.Tensor) -> torch.Tensor:
        return self.net(graph_lat).squeeze(-1)


class DynamicGNN(nn.Module):
    def __init__(
        self,
        *,
        state_dim: int = 21,
        relation_dim: int = 6,
        hidden_dim: int = 128,
        proc_layer: int = 10,
        decoder_output_dim: int = 3,
        use_reward: bool = False,
        name: str = "full",
    ):
        super().__init__()
        self.name = name
        self.use_reward = bool(use_reward)
        self.node_encoder = MLP(state_dim, hidden_dim, hidden_dim)
        self.edge_encoder = MLP(relation_dim, hidden_dim, hidden_dim)
        self.processor = nn.ModuleList([GNBlock(hidden_dim) for _ in range(proc_layer)])
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, decoder_output_dim),
        )
        self.reward_model = RewardModel(hidden_dim) if self.use_reward else None

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        graph_index: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | None]:
        node_lat = self.node_encoder(x)
        edge_lat = self.edge_encoder(edge_attr)
        for block in self.processor:
            node_lat, edge_lat = block(node_lat, edge_index, edge_lat)

        pred = self.decoder(node_lat)
        graph_lat = None
        reward = None
        if graph_index is not None:
            graph_lat = mean_pool(node_lat, graph_index)
            if self.reward_model is not None:
                reward = self.reward_model(graph_lat)

        return {
            "pred": pred,
            "reward": reward,
            "node_lat": node_lat,
            "graph_lat": graph_lat,
        }


def mean_pool(node_lat: torch.Tensor, graph_index: torch.Tensor) -> torch.Tensor:
    num_graphs = int(graph_index.max().item()) + 1 if graph_index.numel() else 0
    pooled = torch.zeros((num_graphs, node_lat.shape[1]), dtype=node_lat.dtype, device=node_lat.device)
    pooled.index_add_(0, graph_index, node_lat)
    count = torch.zeros((num_graphs, 1), dtype=node_lat.dtype, device=node_lat.device)
    count.index_add_(0, graph_index, torch.ones((node_lat.shape[0], 1), dtype=node_lat.dtype, device=node_lat.device))
    return pooled / count.clamp_min(1.0)
