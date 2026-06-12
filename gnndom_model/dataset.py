# SPDX-License-Identifier: BSD-3-Clause
"""Dataset and batching for GNNDOM graph transition files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class GraphBatch:
    x: torch.Tensor
    edge_index: torch.Tensor
    edge_attr: torch.Tensor
    gt_vel: torch.Tensor
    gt_accel: torch.Tensor
    positions: torch.Tensor
    raw_positions: torch.Tensor
    target_pos: torch.Tensor
    picker_position: torch.Tensor
    action: torch.Tensor
    picked_particles: torch.Tensor
    partial_pc_mapped_idx: torch.Tensor
    graph_index: torch.Tensor
    ptr: torch.Tensor
    path: list[str]

    def to(self, device: torch.device | str) -> "GraphBatch":
        tensor_fields = {
            name: value.to(device)
            for name, value in self.__dict__.items()
            if isinstance(value, torch.Tensor)
        }
        return GraphBatch(path=self.path, **tensor_fields)

    @property
    def num_graphs(self) -> int:
        return len(self.path)


class GraphDataset(Dataset):
    """Loads graph_XXXX.npz transitions produced by ``gnndom_graph``.

    The v1 graph builder emits one graph schema. ``input_type`` is still kept so
    callers can preserve ManiFabric's ``full``/``vsbl`` mode split; future
    partial-visible graph files can be paired here without changing trainer APIs.
    """

    def __init__(self, graphf: str | Path, phase: str, *, input_type: str = "full", max_graphs: int | None = None):
        self.graphf = Path(graphf)
        self.phase = phase
        self.input_type = input_type
        self.paths = _collect_graph_paths(self.graphf / phase)
        if max_graphs is not None:
            self.paths = self.paths[: int(max_graphs)]
        if not self.paths:
            raise FileNotFoundError(f"No graph_*.npz files found under {self.graphf / phase}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> dict:
        path = self.paths[idx]
        with np.load(path) as data:
            item = {key: data[key].copy() for key in data.files}
        node_count = int(item["x"].shape[0])
        item["partial_pc_mapped_idx"] = np.arange(node_count, dtype=np.int64)
        item["path"] = str(path)
        item["input_type"] = self.input_type
        return item

    @property
    def node_dim(self) -> int:
        return int(self[0]["x"].shape[1])

    @property
    def edge_dim(self) -> int:
        return int(self[0]["edge_attr"].shape[1])


def collate_graphs(items: Iterable[dict]) -> GraphBatch:
    items = list(items)
    x_blocks = []
    edge_attr_blocks = []
    edge_index_blocks = []
    gt_vel_blocks = []
    gt_accel_blocks = []
    positions_blocks = []
    raw_positions_blocks = []
    target_blocks = []
    graph_index_blocks = []
    partial_idx_blocks = []
    picker_blocks = []
    action_blocks = []
    picked_blocks = []
    ptr = [0]
    offset = 0
    paths = []

    for graph_id, item in enumerate(items):
        x = _float_tensor(item["x"])
        num_nodes = int(x.shape[0])
        edge_index = torch.as_tensor(item["edge_index"], dtype=torch.long) + offset

        x_blocks.append(x)
        edge_index_blocks.append(edge_index)
        edge_attr_blocks.append(_float_tensor(item["edge_attr"]))
        gt_vel_blocks.append(_float_tensor(item["gt_vel"]))
        gt_accel_blocks.append(_float_tensor(item["gt_accel"]))
        positions_blocks.append(_float_tensor(item["positions"]))
        raw_positions_blocks.append(_float_tensor(item.get("raw_positions", item["positions"])))
        target_blocks.append(_float_tensor(item["target_pos"]))
        graph_index_blocks.append(torch.full((num_nodes,), graph_id, dtype=torch.long))
        partial_idx_blocks.append(torch.as_tensor(item["partial_pc_mapped_idx"], dtype=torch.long) + offset)
        picker_blocks.append(_float_tensor(item["picker_position"]).reshape(1, 2, 3))
        action_blocks.append(_float_tensor(item["action"]).reshape(1, 8))
        picked = torch.as_tensor(item["picked_particles"], dtype=torch.long).reshape(1, 2)
        picked_blocks.append(torch.where(picked >= 0, picked + offset, picked))
        paths.append(str(item["path"]))

        offset += num_nodes
        ptr.append(offset)

    return GraphBatch(
        x=torch.cat(x_blocks, dim=0),
        edge_index=torch.cat(edge_index_blocks, dim=1),
        edge_attr=torch.cat(edge_attr_blocks, dim=0),
        gt_vel=torch.cat(gt_vel_blocks, dim=0),
        gt_accel=torch.cat(gt_accel_blocks, dim=0),
        positions=torch.cat(positions_blocks, dim=0),
        raw_positions=torch.cat(raw_positions_blocks, dim=0),
        target_pos=torch.cat(target_blocks, dim=0),
        picker_position=torch.cat(picker_blocks, dim=0),
        action=torch.cat(action_blocks, dim=0),
        picked_particles=torch.cat(picked_blocks, dim=0),
        partial_pc_mapped_idx=torch.cat(partial_idx_blocks, dim=0),
        graph_index=torch.cat(graph_index_blocks, dim=0),
        ptr=torch.as_tensor(ptr, dtype=torch.long),
        path=paths,
    )


def _collect_graph_paths(phase_dir: Path) -> list[Path]:
    if not phase_dir.exists():
        return []
    paths: list[Path] = []
    for rollout_dir in sorted([p for p in phase_dir.iterdir() if p.is_dir()], key=lambda p: int(p.name) if p.name.isdigit() else p.name):
        paths.extend(sorted(rollout_dir.glob("graph_*.npz"), key=lambda p: int(p.stem.split("_")[-1])))
    return paths


def _float_tensor(array: np.ndarray) -> torch.Tensor:
    return torch.as_tensor(array, dtype=torch.float32)
