# SPDX-License-Identifier: BSD-3-Clause
"""ManiFabric-style EdgeGNN dataset built from GNNDOM graph transitions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class EdgeBatch:
    x: torch.Tensor
    edge_index: torch.Tensor
    edge_attr: torch.Tensor
    gt_mesh_edge: torch.Tensor
    graph_index: torch.Tensor
    ptr: torch.Tensor
    path: list[str]

    def to(self, device: torch.device | str) -> "EdgeBatch":
        tensor_fields = {
            name: value.to(device)
            for name, value in self.__dict__.items()
            if isinstance(value, torch.Tensor)
        }
        return EdgeBatch(path=self.path, **tensor_fields)

    @property
    def num_graphs(self) -> int:
        return len(self.path)


class EdgeGraphDataset(Dataset):
    """Creates EdgeGNN samples from ``graph_XXXX.npz`` files.

    This mirrors ManiFabric's ``ClothDatasetPointCloudEdge``:
    node features are centered pointcloud coordinates, candidate edges are
    radius-neighbor edges, and labels mark cloth mesh adjacency.
    """

    def __init__(
        self,
        graphf: str | Path,
        phase: str,
        *,
        neighbor_radius: float = 0.045,
        use_raw_positions: bool = True,
        max_graphs: int | None = None,
    ):
        self.graphf = Path(graphf)
        self.phase = phase
        self.neighbor_radius = float(neighbor_radius)
        self.use_raw_positions = bool(use_raw_positions)
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
            raw = {key: data[key].copy() for key in data.files}

        pos_key = "raw_positions" if self.use_raw_positions and "raw_positions" in raw else "positions"
        pointcloud = raw[pos_key].astype(np.float32)
        normalized = pointcloud - pointcloud.mean(axis=0, keepdims=True)
        edge_index, edge_attr = radius_edges(normalized, self.neighbor_radius)
        scene_params = raw["scene_params"].astype(np.float32)
        gt_mesh_edge = mesh_edge_labels(edge_index, node_count=len(pointcloud), scene_params=scene_params)

        return {
            "x": normalized.astype(np.float32),
            "edge_index": edge_index.astype(np.int64),
            "edge_attr": edge_attr.astype(np.float32),
            "gt_mesh_edge": gt_mesh_edge.astype(np.float32),
            "path": str(path),
        }


def collate_edge_graphs(items: Iterable[dict]) -> EdgeBatch:
    items = list(items)
    x_blocks = []
    edge_index_blocks = []
    edge_attr_blocks = []
    label_blocks = []
    graph_index_blocks = []
    ptr = [0]
    offset = 0
    paths = []

    for graph_id, item in enumerate(items):
        x = torch.as_tensor(item["x"], dtype=torch.float32)
        num_nodes = int(x.shape[0])
        x_blocks.append(x)
        edge_index_blocks.append(torch.as_tensor(item["edge_index"], dtype=torch.long) + offset)
        edge_attr_blocks.append(torch.as_tensor(item["edge_attr"], dtype=torch.float32))
        label_blocks.append(torch.as_tensor(item["gt_mesh_edge"], dtype=torch.float32))
        graph_index_blocks.append(torch.full((num_nodes,), graph_id, dtype=torch.long))
        paths.append(str(item["path"]))
        offset += num_nodes
        ptr.append(offset)

    return EdgeBatch(
        x=torch.cat(x_blocks, dim=0),
        edge_index=torch.cat(edge_index_blocks, dim=1),
        edge_attr=torch.cat(edge_attr_blocks, dim=0),
        gt_mesh_edge=torch.cat(label_blocks, dim=0),
        graph_index=torch.cat(graph_index_blocks, dim=0),
        ptr=torch.as_tensor(ptr, dtype=torch.long),
        path=paths,
    )


def radius_edges(pointcloud: np.ndarray, radius: float) -> tuple[np.ndarray, np.ndarray]:
    pointcloud = np.asarray(pointcloud, dtype=np.float32)
    radius = float(radius)
    if len(pointcloud) < 2:
        edge_index = np.asarray([[0], [0]], dtype=np.int64)
        edge_attr = edge_features(pointcloud, edge_index)
        return edge_index, edge_attr

    try:
        from scipy.spatial import cKDTree

        pairs = np.asarray(list(cKDTree(pointcloud).query_pairs(radius, p=2)), dtype=np.int64)
        if pairs.size == 0:
            return fake_edges(pointcloud)
        undirected = pairs.T
    except Exception:
        diff = pointcloud[:, None, :] - pointcloud[None, :, :]
        dist = np.linalg.norm(diff, axis=-1)
        src, dst = np.where(np.triu((dist < radius) & (dist > 0.0), k=1))
        if len(src) == 0:
            return fake_edges(pointcloud)
        undirected = np.stack([src, dst], axis=0).astype(np.int64)

    edge_index = np.concatenate([undirected, undirected[::-1]], axis=1).astype(np.int64)
    return edge_index, edge_features(pointcloud, edge_index)


def fake_edges(pointcloud: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(pointcloud) >= 2:
        edge_index = np.asarray([[0, 1], [1, 0]], dtype=np.int64)
    else:
        edge_index = np.asarray([[0], [0]], dtype=np.int64)
    return edge_index, edge_features(pointcloud, edge_index)


def edge_features(pointcloud: np.ndarray, edge_index: np.ndarray) -> np.ndarray:
    src, dst = edge_index
    disp = pointcloud[src] - pointcloud[dst]
    dist = np.linalg.norm(disp, axis=1, keepdims=True)
    return np.concatenate([disp, dist], axis=1).astype(np.float32)


def mesh_edge_labels(edge_index: np.ndarray, *, node_count: int, scene_params: np.ndarray) -> np.ndarray:
    cloth_xdim = int(scene_params[1])
    cloth_ydim = int(scene_params[2])
    labels = np.zeros((edge_index.shape[1], 1), dtype=np.float32)
    if cloth_xdim <= 0 or cloth_ydim <= 0:
        return labels

    for edge_idx in range(edge_index.shape[1]):
        src = int(edge_index[0, edge_idx])
        dst = int(edge_index[1, edge_idx])
        if src < 0 or dst < 0 or src >= node_count or dst >= node_count:
            continue
        if _is_grid_neighbor(src, dst, cloth_xdim, cloth_ydim):
            labels[edge_idx, 0] = 1.0
    return labels


def _is_grid_neighbor(src: int, dst: int, cloth_xdim: int, cloth_ydim: int) -> bool:
    if src == dst:
        return True
    sx, sy = src % cloth_xdim, src // cloth_xdim
    dx, dy = dst % cloth_xdim, dst // cloth_xdim
    if sy >= cloth_ydim or dy >= cloth_ydim:
        return False
    return max(abs(sx - dx), abs(sy - dy)) == 1


def _collect_graph_paths(phase_dir: Path) -> list[Path]:
    if not phase_dir.exists():
        return []
    paths: list[Path] = []
    rollout_dirs = sorted([p for p in phase_dir.iterdir() if p.is_dir()], key=lambda p: int(p.name) if p.name.isdigit() else p.name)
    for rollout_dir in rollout_dirs:
        paths.extend(sorted(rollout_dir.glob("graph_*.npz"), key=lambda p: int(p.stem.split("_")[-1])))
    return paths
