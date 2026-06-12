# SPDX-License-Identifier: BSD-3-Clause
"""Pure PyTorch EdgeGNN for mesh-edge classification."""

from __future__ import annotations

import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, *, num_layers: int = 3, layer_norm: bool = False):
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


class EdgeGNBlock(nn.Module):
    def __init__(self, hidden_dim: int, global_size: int):
        super().__init__()
        self.global_size = int(global_size)
        self.edge_mlp = MLP(hidden_dim * 3 + self.global_size, hidden_dim, hidden_dim)
        self.node_mlp = MLP(hidden_dim * 2 + self.global_size, hidden_dim, hidden_dim)
        self.global_mlp = MLP(self.global_size + hidden_dim * 2, hidden_dim, self.global_size) if self.global_size > 0 else None

    def forward(
        self,
        node_lat: torch.Tensor,
        edge_index: torch.Tensor,
        edge_lat: torch.Tensor,
        global_lat: torch.Tensor,
        graph_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        src, dst = edge_index
        edge_graph = graph_index[src]
        edge_update = self.edge_mlp(torch.cat([node_lat[src], node_lat[dst], edge_lat, global_lat[edge_graph]], dim=-1))
        edge_lat = edge_lat + edge_update

        agg = torch.zeros_like(node_lat)
        agg.index_add_(0, dst, edge_lat)
        count = torch.zeros((node_lat.shape[0], 1), dtype=node_lat.dtype, device=node_lat.device)
        count.index_add_(0, dst, torch.ones((edge_lat.shape[0], 1), dtype=node_lat.dtype, device=node_lat.device))
        agg = agg / count.clamp_min(1.0)

        node_update = self.node_mlp(torch.cat([node_lat, agg, global_lat[graph_index]], dim=-1))
        node_lat = node_lat + node_update

        if self.global_mlp is not None:
            node_mean = mean_pool(node_lat, graph_index, global_lat.shape[0])
            edge_mean = mean_pool(edge_lat, edge_graph, global_lat.shape[0])
            global_lat = global_lat + self.global_mlp(torch.cat([global_lat, node_mean, edge_mean], dim=-1))
        return node_lat, edge_lat, global_lat


class EdgeGNN(nn.Module):
    def __init__(
        self,
        *,
        state_dim: int = 3,
        relation_dim: int = 4,
        hidden_dim: int = 128,
        proc_layer: int = 10,
        global_size: int = 128,
    ):
        super().__init__()
        self.global_size = int(global_size)
        self.node_encoder = MLP(state_dim, hidden_dim, hidden_dim)
        self.edge_encoder = MLP(relation_dim, hidden_dim, hidden_dim)
        self.processor = nn.ModuleList([EdgeGNBlock(hidden_dim, self.global_size) for _ in range(proc_layer)])
        self.decoder = MLP(hidden_dim, hidden_dim, 1, layer_norm=False)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor, graph_index: torch.Tensor) -> dict[str, torch.Tensor]:
        node_lat = self.node_encoder(x)
        edge_lat = self.edge_encoder(edge_attr)
        num_graphs = int(graph_index.max().item()) + 1 if graph_index.numel() else 1
        global_lat = torch.zeros((num_graphs, self.global_size), dtype=x.dtype, device=x.device)
        for block in self.processor:
            node_lat, edge_lat, global_lat = block(node_lat, edge_index, edge_lat, global_lat, graph_index)
        return {
            "mesh_edge": self.decoder(edge_lat),
            "node_lat": node_lat,
            "edge_lat": edge_lat,
            "lat_nxt": global_lat,
        }


def mean_pool(values: torch.Tensor, index: torch.Tensor, num_groups: int) -> torch.Tensor:
    pooled = torch.zeros((num_groups, values.shape[1]), dtype=values.dtype, device=values.device)
    pooled.index_add_(0, index, values)
    count = torch.zeros((num_groups, 1), dtype=values.dtype, device=values.device)
    count.index_add_(0, index, torch.ones((values.shape[0], 1), dtype=values.dtype, device=values.device))
    return pooled / count.clamp_min(1.0)
