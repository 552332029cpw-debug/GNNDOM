# SPDX-License-Identifier: BSD-3-Clause
"""Build ManiFabric-style graph transitions from GNNDOM rollout datasets."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
import json
from pathlib import Path

import numpy as np

from gnndom_dataset.storage import load_step

from .features import apply_picker_action, build_velocity_history, compute_edge_attr, compute_node_attr, compute_visible_edge_attr


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
    graph_mode: str = "full"
    edge_model_path: Path | None = None
    edge_threshold: float = 0.5
    edge_device: str = "cpu"


def build_graphs_from_dataset(cfg: GraphBuildConfig) -> list[Path]:
    if cfg.graph_mode not in {"full", "vsbl", "both"}:
        raise ValueError("graph_mode must be one of: full, vsbl, both")
    saved: list[Path] = []
    modes = ("full", "vsbl") if cfg.graph_mode == "both" else (cfg.graph_mode,)
    for phase in ("train", "valid"):
        phase_dir = cfg.dataf / phase
        if not phase_dir.exists():
            continue
        for mode in modes:
            for rollout_dir in sorted([p for p in phase_dir.iterdir() if p.is_dir()], key=lambda p: int(p.name)):
                out_root = cfg.graphf / mode if cfg.graph_mode == "both" else cfg.graphf
                out_dir = out_root / phase / rollout_dir.name
                out_dir.mkdir(parents=True, exist_ok=True)
                saved.extend(build_rollout_graphs(rollout_dir, out_dir, cfg, graph_mode=mode))
    return saved


def build_rollout_graphs(rollout_dir: Path, out_dir: Path, cfg: GraphBuildConfig, *, graph_mode: str | None = None) -> list[Path]:
    graph_mode = cfg.graph_mode if graph_mode is None else graph_mode
    with (rollout_dir / "rollout_info.json").open("r", encoding="utf-8") as f:
        rollout_info = json.load(f)
    step_files = sorted([p for p in rollout_dir.glob("*.npz")], key=lambda p: int(p.stem))
    max_start = len(step_files) - cfg.pred_time_interval
    saved: list[Path] = []
    for timestep in range(max_start):
        graph = build_transition_graph(rollout_dir, rollout_info, timestep, cfg, graph_mode=graph_mode)
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
        "graph_mode": graph_mode,
        "edge_model_path": None if cfg.edge_model_path is None else str(cfg.edge_model_path),
        "edge_threshold": cfg.edge_threshold,
        "edge_device": cfg.edge_device,
        "coordinate_system": "z_up_xy_horizontal",
        "target_source": rollout_info.get("target_source", "legacy_unknown"),
        "geometric_target_source": rollout_info.get("geometric_target_source", "legacy_unknown"),
        "target_release_grasp": rollout_info.get("target_release_grasp", None),
        "target_settle_steps": rollout_info.get("target_settle_steps", None),
        "has_geometric_target_pos": "geometric_target_pos" in rollout_info,
        "builder_config": _json_config(cfg),
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


def build_transition_graph(rollout_dir: Path, rollout_info: dict, timestep: int, cfg: GraphBuildConfig, *, graph_mode: str | None = None) -> dict:
    graph_mode = cfg.graph_mode if graph_mode is None else graph_mode
    return build_transition_graph_for_mode(rollout_dir, rollout_info, timestep, cfg, graph_mode)


def build_transition_graph_for_mode(rollout_dir: Path, rollout_info: dict, timestep: int, cfg: GraphBuildConfig, graph_mode: str) -> dict:
    if graph_mode == "full":
        return build_full_transition_graph(rollout_dir, rollout_info, timestep, cfg)
    if graph_mode == "vsbl":
        return build_visible_transition_graph(rollout_dir, rollout_info, timestep, cfg)
    raise ValueError("graph_mode must be full or vsbl")


def build_full_transition_graph(rollout_dir: Path, rollout_info: dict, timestep: int, cfg: GraphBuildConfig) -> dict:
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
        "graph_mode": np.asarray("full"),
    }


def build_visible_transition_graph(rollout_dir: Path, rollout_info: dict, timestep: int, cfg: GraphBuildConfig) -> dict:
    data_cur = load_step(rollout_dir, timestep)
    data_nxt = load_step(rollout_dir, timestep + cfg.pred_time_interval)
    if "partial_pc_mapped_idx" not in data_cur:
        raise KeyError(f"{rollout_dir}/{timestep}.npz is missing partial_pc_mapped_idx; regenerate dataset with camera observation.")

    downsample_idx = np.asarray(rollout_info["downsample_idx"], dtype=np.int64)
    scene_params = np.asarray(rollout_info["scene_params"], dtype=np.float32)
    cloth_xdim = int(scene_params[1])
    cloth_ydim = int(scene_params[2])
    partial_map = np.asarray(data_cur["partial_pc_mapped_idx"], dtype=np.int64).reshape(-1)
    pointcloud = data_cur["pointcloud"].astype(np.float32)
    if len(pointcloud) != len(partial_map):
        min_len = min(len(pointcloud), len(partial_map))
        pointcloud = pointcloud[:min_len]
        partial_map = partial_map[:min_len]
    if len(pointcloud) == 0:
        raise RuntimeError(f"{rollout_dir}/{timestep}.npz has an empty visible pointcloud")

    full_pos_cur = data_cur["positions"].astype(np.float32)
    full_pos_nxt = data_nxt["positions"].astype(np.float32)
    down_pos_cur = full_pos_cur[downsample_idx]
    down_pos_nxt = full_pos_nxt[downsample_idx]
    velocity_his_full = build_velocity_history(
        rollout_dir,
        timestep,
        n_his=cfg.n_his,
        pred_time_interval=cfg.pred_time_interval,
        downsample_idx=downsample_idx,
    )
    velocity_his = velocity_his_full[partial_map].astype(np.float32)
    pointcloud_input, velocity_his_input, picked_particles = apply_picker_action(
        pointcloud,
        velocity_his,
        data_cur["picker_position"].astype(np.float32),
        data_cur["action"].astype(np.float32),
        dt=cfg.dt,
        pred_time_interval=cfg.pred_time_interval,
    )

    node_attr = compute_node_attr(pointcloud_input, velocity_his_input, picked_particles, rollout_info=rollout_info, use_es=cfg.use_es)
    mesh_edges = None
    mesh_edge_source = "none"
    if cfg.use_mesh_edge:
        if cfg.edge_model_path is not None:
            predictor = _get_edge_predictor(str(cfg.edge_model_path), cfg.edge_device, cfg.neighbor_radius)
            mesh_edges = predictor.predict_mesh_edges(pointcloud_input, threshold=cfg.edge_threshold)
            mesh_edge_source = "edgegnn"
        else:
            mesh_edge_source = "visible_ground_truth"
    edge_index, edge_attr = compute_visible_edge_attr(
        pointcloud_input,
        cloth_xdim=cloth_xdim,
        cloth_ydim=cloth_ydim,
        visible_downsample_idx=partial_map,
        neighbor_radius=cfg.neighbor_radius,
        use_mesh_edge=cfg.use_mesh_edge,
        mesh_edges=mesh_edges,
    )
    gt_vel = (down_pos_nxt[partial_map] - down_pos_cur[partial_map]) / np.float32(cfg.dt * cfg.pred_time_interval)
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
        "target_pos": np.asarray(rollout_info["target_pos"], dtype=np.float32)[downsample_idx][partial_map],
        "picker_position": data_cur["picker_position"].astype(np.float32),
        "action": data_cur["action"].astype(np.float32),
        "picked_particles": picked_particles.astype(np.int64),
        "scene_params": scene_params.astype(np.float32),
        "downsample_idx": downsample_idx.astype(np.int64),
        "partial_pc_mapped_idx": partial_map.astype(np.int64),
        "downsample_observable_idx": np.asarray(data_cur["downsample_observable_idx"], dtype=np.int64),
        "graph_mode": np.asarray("vsbl"),
        "mesh_edge_source": np.asarray(mesh_edge_source),
    }


def _json_config(cfg: GraphBuildConfig) -> dict:
    data = asdict(cfg)
    data["dataf"] = str(cfg.dataf)
    data["graphf"] = str(cfg.graphf)
    data["edge_model_path"] = None if cfg.edge_model_path is None else str(cfg.edge_model_path)
    return data


@lru_cache(maxsize=4)
def _get_edge_predictor(edge_model_path: str, edge_device: str, neighbor_radius: float):
    from gnndom_edge.infer import EdgeGNNMeshPredictor

    return EdgeGNNMeshPredictor(Path(edge_model_path), device=edge_device, neighbor_radius=float(neighbor_radius))
