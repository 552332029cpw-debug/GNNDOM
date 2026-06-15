# SPDX-License-Identifier: BSD-3-Clause
"""Dynamic GNN rollout used by online MPC planning."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from gnndom_model import DynamicGNN

from .config import PlanConfig
from .graph import OnlineGraphState, OnlineVisibleGraphBuilder


@dataclass
class RolloutResult:
    final_ret: float
    model_positions: np.ndarray
    shape_positions: np.ndarray
    pred_vel: list[np.ndarray]


class DynamicsRollout:
    def __init__(self, cfg: PlanConfig, graph_builder: OnlineVisibleGraphBuilder):
        self.cfg = cfg
        self.graph_builder = graph_builder
        self.device = torch.device(cfg.device)
        self.model, self.model_config = load_vsbl_model(cfg.dyn_path, self.device)
        self.output_type = str(self.model_config.get("output_type", "vel"))
        if self.output_type not in {"vel", "accel"}:
            raise ValueError(f"Unsupported checkpoint output_type={self.output_type!r}; expected vel or accel")
        self.model.eval()

    def rollout(self, init_state: OnlineGraphState, actions: np.ndarray) -> RolloutResult:
        actions = np.asarray(actions, dtype=np.float32).reshape(-1, 8)
        state = clone_graph_state(init_state)
        horizon = len(actions)
        model_positions = np.zeros((horizon, len(state.pointcloud), 3), dtype=np.float32)
        shape_positions = np.zeros((horizon, 2, 3), dtype=np.float32)
        pred_vels: list[np.ndarray] = []
        final_ret = 0.0

        for t, action in enumerate(actions):
            model_positions[t] = state.pointcloud
            shape_positions[t] = state.picker_position
            graph = self.graph_builder.build_model_graph(state, action)
            pred = self._predict(graph)
            next_pos, next_vel_his, picker_pos, pred_vel = self._integrate(state, graph, pred, action)
            state.pointcloud = next_pos
            state.vel_his = next_vel_his
            state.picker_position = picker_pos
            pred_vels.append(pred_vel)
            final_ret = visible_target_reward(state.pointcloud, graph["target_pos"])

        return RolloutResult(
            final_ret=float(final_ret),
            model_positions=model_positions,
            shape_positions=shape_positions,
            pred_vel=pred_vels,
        )

    def _predict(self, graph: dict) -> np.ndarray:
        expected_state_dim = int(self.model_config.get("state_dim", graph["x"].shape[1]))
        expected_relation_dim = int(self.model_config.get("relation_dim", graph["edge_attr"].shape[1]))
        if graph["x"].shape[1] != expected_state_dim:
            raise RuntimeError(
                f"Online graph node_dim={graph['x'].shape[1]} does not match checkpoint state_dim={expected_state_dim}; "
                "check --n-his and graph feature settings."
            )
        if graph["edge_attr"].shape[1] != expected_relation_dim:
            raise RuntimeError(
                f"Online graph edge_dim={graph['edge_attr'].shape[1]} does not match checkpoint relation_dim={expected_relation_dim}; "
                "check mesh-edge/rest-distance graph settings."
            )
        with torch.no_grad():
            x = torch.as_tensor(graph["x"], dtype=torch.float32, device=self.device)
            edge_index = torch.as_tensor(graph["edge_index"], dtype=torch.long, device=self.device)
            edge_attr = torch.as_tensor(graph["edge_attr"], dtype=torch.float32, device=self.device)
            out = self.model(x, edge_index, edge_attr)
        pred = out["pred"].detach().cpu().numpy().astype(np.float32)
        if not np.all(np.isfinite(pred)):
            raise RuntimeError("DynamicGNN rollout produced non-finite predictions")
        return pred

    def _integrate(self, state: OnlineGraphState, graph: dict, pred: np.ndarray, action: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        pointcloud = graph["positions"].astype(np.float32).copy()
        velocity_his = graph["vel_his"].astype(np.float32).copy()
        step_dt = np.float32(self.cfg.dt * self.cfg.pred_time_interval)
        if self.output_type == "accel":
            pred_vel = velocity_his[:, -3:] + pred * step_dt
        else:
            pred_vel = pred
        next_pos = pointcloud + pred_vel * step_dt
        next_pos[:, 2] = np.maximum(next_pos[:, 2], np.float32(self.cfg.cloth_particle_radius))
        pred_vel = (next_pos - pointcloud) / step_dt
        next_vel_his = np.hstack([velocity_his[:, 3:], pred_vel]).astype(np.float32)

        action_2 = np.asarray(action, dtype=np.float32).reshape(2, 4)
        picker_pos = state.picker_position.copy()
        picker_pos[0] += action_2[0, :3]
        picker_pos[1] += action_2[1, :3]
        picked_particles = graph["picked_particles"].reshape(2)
        for local_picker, picked in enumerate(picked_particles):
            if picked >= 0:
                next_pos[picked] = graph["positions"][picked]
                next_vel_his[picked] = graph["vel_his"][picked]
        return next_pos.astype(np.float32), next_vel_his.astype(np.float32), picker_pos.astype(np.float32), pred_vel.astype(np.float32)


def load_vsbl_model(path: Path, device: torch.device) -> tuple[DynamicGNN, dict]:
    checkpoint = torch.load(path, map_location=device)
    if not isinstance(checkpoint, dict):
        raise ValueError("Dynamic checkpoint must be a dict containing model_state_dict and config")
    config = checkpoint.get("config")
    if not isinstance(config, dict):
        raise ValueError("Dynamic checkpoint is missing a config dict")
    model = DynamicGNN(
        state_dim=int(config.get("state_dim", 21)),
        relation_dim=int(config.get("relation_dim", 6)),
        hidden_dim=int(config.get("global_size", 128)),
        proc_layer=int(config.get("proc_layer", 10)),
        use_reward=False,
        name="vsbl",
    ).to(device)
    state = checkpoint.get("model_state_dict")
    if state is None:
        raise ValueError("Dynamic checkpoint is missing model_state_dict")
    model.load_state_dict(state)
    return model, config


def clone_graph_state(state: OnlineGraphState) -> OnlineGraphState:
    return OnlineGraphState(
        pointcloud=state.pointcloud.copy(),
        vel_his=state.vel_his.copy(),
        picker_position=state.picker_position.copy(),
        scene_params=state.scene_params.copy(),
        partial_pc_mapped_idx=state.partial_pc_mapped_idx.copy(),
        downsample_idx=state.downsample_idx.copy(),
        target_pos=state.target_pos.copy(),
        target_picker_pos=state.target_picker_pos.copy(),
        rollout_info=state.rollout_info,
    )


def visible_target_reward(pointcloud: np.ndarray, target_pos: np.ndarray) -> float:
    pointcloud = np.asarray(pointcloud, dtype=np.float32).reshape(-1, 3)
    target_pos = np.asarray(target_pos, dtype=np.float32).reshape(-1, 3)
    if len(pointcloud) == 0 or len(pointcloud) != len(target_pos):
        return -float("inf")
    return -float(np.mean(np.linalg.norm(pointcloud - target_pos, axis=1)))
