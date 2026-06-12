# SPDX-License-Identifier: BSD-3-Clause
"""Build ManiFabric-style graph transitions from GNNDOM rollout datasets."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np

from gnndom_dataset.storage import load_step

from .features import apply_picker_action, build_velocity_history, compute_edge_attr, compute_node_attr


@dataclass(frozen=True)
class GraphBuildConfig:
    dataf: Path
    graphf: Path
    n_his: int = 5
    pred_time_interval: int = 1
    dt: float = 1.0 / 60.0
    neighbor_radius: float = 0.045
    use_mesh_edge: bool = True
    use_es: bool = True


def build_graphs_from_dataset(cfg: GraphBuildConfig) -> list[Path]:
    saved: list[Path] = []
    for phase in ("train", "valid"):
        phase_dir = cfg.dataf / phase
        if not phase_dir.exists():
            continue
        for rollout_dir in sorted([p for p in phase_dir.iterdir() if p.is_dir()], key=lambda p: int(p.name)):
            out_dir = cfg.graphf / phase / rollout_dir.name
            out_dir.mkdir(parents=True, exist_ok=True)
            saved.extend(build_rollout_graphs(rollout_dir, out_dir, cfg))
    return saved


def build_rollout_graphs(rollout_dir: Path, out_dir: Path, cfg: GraphBuildConfig) -> list[Path]:
    with (rollout_dir / "rollout_info.json").open("r", encoding="utf-8") as f:
        rollout_info = json.load(f)
    step_files = sorted([p for p in rollout_dir.glob("*.npz")], key=lambda p: int(p.stem))
    max_start = len(step_files) - cfg.pred_time_interval
    saved: list[Path] = []
    for timestep in range(max_start):
        graph = build_transition_graph(rollout_dir, rollout_info, timestep, cfg)
        path = out_dir / f"graph_{timestep:04d}.npz"
        np.savez_compressed(path, **graph)
        saved.append(path)

    graph_info = {
        "source_rollout": str(rollout_dir),
        "num_graphs": len(saved),
        "n_his": cfg.n_his,
        "pred_time_interval": cfg.pred_time_interval,
        "dt": cfg.dt,
        "neighbor_radius": cfg.neighbor_radius,
        "use_mesh_edge": cfg.use_mesh_edge,
        "use_es": cfg.use_es,
        "coordinate_system": "z_up_xy_horizontal",
        "builder_config": {**asdict(cfg), "dataf": str(cfg.dataf), "graphf": str(cfg.graphf)},
    }
    if saved:
        first = np.load(saved[0])
        graph_info.update(
            {
                "node_dim": int(first["x"].shape[1]),
                "edge_dim": int(first["edge_attr"].shape[1]),
                "target_dim": int(first["gt_accel"].shape[1]),
            }
        )
    with (out_dir / "graph_info.json").open("w", encoding="utf-8") as f:
        json.dump(graph_info, f, indent=2)
    return saved


def build_transition_graph(rollout_dir: Path, rollout_info: dict, timestep: int, cfg: GraphBuildConfig) -> dict:
    data_cur = load_step(rollout_dir, timestep)
    data_nxt = load_step(rollout_dir, timestep + cfg.pred_time_interval)
    downsample_idx = np.asarray(rollout_info["downsample_idx"], dtype=np.int64)
    scene_params = np.asarray(rollout_info["scene_params"], dtype=np.float32)
    cloth_xdim = int(scene_params[1])
    cloth_ydim = int(scene_params[2])

    full_pos_cur = data_cur["positions"].astype(np.float32)
    full_pos_nxt = data_nxt["positions"].astype(np.float32)
    pointcloud = full_pos_cur[downsample_idx].astype(np.float32)
    pointcloud_nxt = full_pos_nxt[downsample_idx].astype(np.float32)
    velocity_his = build_velocity_history(
        rollout_dir,
        timestep,
        n_his=cfg.n_his,
        pred_time_interval=cfg.pred_time_interval,
        downsample_idx=downsample_idx,
    )
    pointcloud_input, velocity_his_input, picked_particles = apply_picker_action(
        pointcloud,
        velocity_his,
        data_cur["picker_position"].astype(np.float32),
        data_cur["action"].astype(np.float32),
        dt=cfg.dt,
        pred_time_interval=cfg.pred_time_interval,
    )

    node_attr = compute_node_attr(pointcloud_input, velocity_his_input, picked_particles, rollout_info=rollout_info, use_es=cfg.use_es)
    edge_index, edge_attr = compute_edge_attr(
        pointcloud_input,
        cloth_xdim=cloth_xdim,
        cloth_ydim=cloth_ydim,
        neighbor_radius=cfg.neighbor_radius,
        use_mesh_edge=cfg.use_mesh_edge,
    )
    gt_vel = (pointcloud_nxt - pointcloud) / np.float32(cfg.dt * cfg.pred_time_interval)
    prev_vel = velocity_his[:, -3:]
    gt_accel = (gt_vel - prev_vel) / np.float32(cfg.dt * cfg.pred_time_interval)

    return {
        "x": node_attr.astype(np.float32),
        "edge_index": edge_index.astype(np.int64),
        "edge_attr": edge_attr.astype(np.float32),
        "gt_vel": gt_vel.astype(np.float32),
        "gt_accel": gt_accel.astype(np.float32),
        "positions": pointcloud_input.astype(np.float32),
        "raw_positions": pointcloud.astype(np.float32),
        "target_pos": np.asarray(rollout_info["target_pos"], dtype=np.float32)[downsample_idx],
        "picker_position": data_cur["picker_position"].astype(np.float32),
        "action": data_cur["action"].astype(np.float32),
        "picked_particles": picked_particles.astype(np.int64),
        "scene_params": scene_params.astype(np.float32),
        "downsample_idx": downsample_idx.astype(np.int64),
    }

