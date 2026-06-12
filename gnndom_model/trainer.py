# SPDX-License-Identifier: BSD-3-Clause
"""Training loops for GNNDOM Dynamic GNN modes."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .dataset import GraphBatch, GraphDataset, collate_graphs
from .model import DynamicGNN


@dataclass
class DynamicTrainConfig:
    graphf: Path
    out_dir: Path
    train_mode: str = "vsbl"
    output_type: str = "vel"
    epochs: int = 100
    batch_size: int = 4
    lr: float = 1.0e-4
    vsbl_lr: float | None = None
    full_lr: float | None = None
    weight_decay: float = 0.0
    device: str = "cpu"
    state_dim: int = 21
    relation_dim: int = 6
    global_size: int = 128
    proc_layer: int = 10
    reward_w: float = 1.0e5
    imit_w: float = 5.0
    imit_w_lat: float = 1.0
    use_reward: bool = True
    tune_teach: bool = False
    copy_teach: list[str] = field(default_factory=list)
    full_dyn_path: Path | None = None
    dt: float = 1.0 / 60.0
    pred_time_interval: int = 1
    num_workers: int = 0
    max_train_graphs: int | None = None
    max_valid_graphs: int | None = None
    seed: int = 0


class DynamicTrainer:
    def __init__(self, cfg: DynamicTrainConfig):
        if cfg.train_mode not in {"vsbl", "full", "graph_imit"}:
            raise ValueError(f"Unsupported train_mode: {cfg.train_mode}")
        if cfg.output_type not in {"vel", "accel"}:
            raise ValueError(f"Unsupported output_type: {cfg.output_type}")
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        torch.manual_seed(cfg.seed)
        self.cfg.out_dir.mkdir(parents=True, exist_ok=True)

        self.train_loader = self._make_loader("train", shuffle=True, max_graphs=cfg.max_train_graphs)
        self.valid_loader = self._make_loader("valid", shuffle=False, max_graphs=cfg.max_valid_graphs)
        self.models = self._make_models()
        self.optimizers = self._make_optimizers()
        self.best_valid = float("inf")

        if cfg.train_mode == "graph_imit":
            self._prepare_teacher_student()

        self._save_config()

    def train(self) -> dict:
        history = []
        for epoch in range(1, self.cfg.epochs + 1):
            train_metrics = self._run_epoch(train=True)
            valid_metrics = self._run_epoch(train=False)
            row = {"epoch": epoch, **{f"train_{k}": v for k, v in train_metrics.items()}, **{f"valid_{k}": v for k, v in valid_metrics.items()}}
            history.append(row)
            print(
                f"[EPOCH {epoch:04d}] "
                f"train_loss={row['train_loss']:.6g} valid_loss={row['valid_loss']:.6g} "
                f"train_dyn={row['train_dyn_loss']:.6g} valid_dyn={row['valid_dyn_loss']:.6g}"
            )
            if row["valid_loss"] < self.best_valid:
                self.best_valid = row["valid_loss"]
                self._save_checkpoints(epoch, row)
            self._write_history(history)
        return {"best_valid_loss": self.best_valid, "history": history}

    def _make_loader(self, phase: str, *, shuffle: bool, max_graphs: int | None) -> DataLoader:
        dataset = GraphDataset(self.cfg.graphf, phase, input_type="full", max_graphs=max_graphs)
        return DataLoader(
            dataset,
            batch_size=self.cfg.batch_size,
            shuffle=shuffle,
            num_workers=self.cfg.num_workers,
            collate_fn=collate_graphs,
        )

    def _make_models(self) -> dict[str, DynamicGNN]:
        cfg = self.cfg
        models: dict[str, DynamicGNN] = {}
        if cfg.train_mode in {"full", "graph_imit"}:
            models["full"] = DynamicGNN(
                state_dim=cfg.state_dim,
                relation_dim=cfg.relation_dim,
                hidden_dim=cfg.global_size,
                proc_layer=cfg.proc_layer,
                use_reward=cfg.use_reward,
                name="full",
            ).to(self.device)
        if cfg.train_mode in {"vsbl", "graph_imit"}:
            models["vsbl"] = DynamicGNN(
                state_dim=cfg.state_dim,
                relation_dim=cfg.relation_dim,
                hidden_dim=cfg.global_size,
                proc_layer=cfg.proc_layer,
                use_reward=False if cfg.train_mode == "vsbl" else cfg.use_reward,
                name="vsbl",
            ).to(self.device)
        return models

    def _make_optimizers(self) -> dict[str, torch.optim.Optimizer]:
        cfg = self.cfg
        optimizers: dict[str, torch.optim.Optimizer] = {}
        if "full" in self.models:
            lr = cfg.full_lr if cfg.full_lr is not None else cfg.lr
            optimizers["full"] = torch.optim.Adam(self.models["full"].parameters(), lr=lr, weight_decay=cfg.weight_decay)
        if "vsbl" in self.models:
            lr = cfg.vsbl_lr if cfg.vsbl_lr is not None else cfg.lr
            optimizers["vsbl"] = torch.optim.Adam(self.models["vsbl"].parameters(), lr=lr, weight_decay=cfg.weight_decay)
        return optimizers

    def _prepare_teacher_student(self) -> None:
        cfg = self.cfg
        if cfg.full_dyn_path is not None:
            self._load_model(self.models["full"], cfg.full_dyn_path)
        elif not cfg.tune_teach:
            raise ValueError("graph_imit requires --full-dyn-path unless --tune-teach is set for scratch smoke training")

        if cfg.copy_teach:
            self._copy_teacher_modules(cfg.copy_teach)
        if not cfg.tune_teach:
            for param in self.models["full"].parameters():
                param.requires_grad = False
            self.optimizers.pop("full", None)

    def _run_epoch(self, *, train: bool) -> dict[str, float]:
        for model in self.models.values():
            model.train(train)
        loader = self.train_loader if train else self.valid_loader
        totals = {"loss": 0.0, "dyn_loss": 0.0, "reward_loss": 0.0, "imit_loss": 0.0}
        num_batches = 0

        for batch in loader:
            batch = batch.to(self.device)
            if train:
                for optim in self.optimizers.values():
                    optim.zero_grad(set_to_none=True)

            with torch.set_grad_enabled(train):
                losses = self._compute_losses(batch)
                if train:
                    losses["loss"].backward()
                    for optim in self.optimizers.values():
                        optim.step()

            for key in totals:
                totals[key] += float(losses[key].detach().cpu())
            num_batches += 1

        return {key: value / max(1, num_batches) for key, value in totals.items()}

    def _compute_losses(self, batch: GraphBatch) -> dict[str, torch.Tensor]:
        cfg = self.cfg
        target = batch.gt_vel if cfg.output_type == "vel" else batch.gt_accel
        zero = torch.zeros((), dtype=batch.x.dtype, device=batch.x.device)

        if cfg.train_mode == "vsbl":
            out = self.models["vsbl"](batch.x, batch.edge_index, batch.edge_attr, batch.graph_index)
            dyn_loss = F.mse_loss(out["pred"], target)
            return {"loss": dyn_loss, "dyn_loss": dyn_loss, "reward_loss": zero, "imit_loss": zero}

        if cfg.train_mode == "full":
            out = self.models["full"](batch.x, batch.edge_index, batch.edge_attr, batch.graph_index)
            dyn_loss = F.mse_loss(out["pred"], target)
            reward_loss = self._reward_loss(out, batch) if cfg.use_reward else zero
            loss = dyn_loss + cfg.reward_w * reward_loss
            return {"loss": loss, "dyn_loss": dyn_loss, "reward_loss": reward_loss, "imit_loss": zero}

        full_grad = cfg.tune_teach
        with torch.set_grad_enabled(self.models["full"].training and full_grad):
            full_out = self.models["full"](batch.x, batch.edge_index, batch.edge_attr, batch.graph_index)
        vsbl_out = self.models["vsbl"](batch.x, batch.edge_index, batch.edge_attr, batch.graph_index)
        dyn_loss = F.mse_loss(vsbl_out["pred"], target)
        reward_loss = self._reward_loss(vsbl_out, batch) if cfg.use_reward else zero
        if cfg.tune_teach:
            dyn_loss = dyn_loss + F.mse_loss(full_out["pred"], target)
            reward_loss = reward_loss + (self._reward_loss(full_out, batch) if cfg.use_reward else zero)
        imit_node_loss = F.mse_loss(vsbl_out["node_lat"], full_out["node_lat"].detach())
        imit_lat_loss = F.mse_loss(vsbl_out["graph_lat"], full_out["graph_lat"].detach())
        imit_loss = cfg.imit_w_lat * imit_lat_loss + imit_node_loss
        loss = dyn_loss + cfg.imit_w * imit_loss + cfg.reward_w * reward_loss
        return {"loss": loss, "dyn_loss": dyn_loss, "reward_loss": reward_loss, "imit_loss": imit_loss}

    def _reward_loss(self, out: dict[str, torch.Tensor | None], batch: GraphBatch) -> torch.Tensor:
        reward = out["reward"]
        if reward is None:
            return torch.zeros((), dtype=batch.x.dtype, device=batch.x.device)
        reward_target = self._reward_target(batch)
        return F.mse_loss(reward, reward_target)

    def _reward_target(self, batch: GraphBatch) -> torch.Tensor:
        step_dt = self.cfg.dt * self.cfg.pred_time_interval
        next_pos = batch.raw_positions + batch.gt_vel * step_dt
        node_reward = -torch.linalg.norm(next_pos - batch.target_pos, dim=-1)
        reward = torch.zeros((batch.num_graphs,), dtype=batch.x.dtype, device=batch.x.device)
        reward.index_add_(0, batch.graph_index, node_reward)
        count = torch.zeros((batch.num_graphs,), dtype=batch.x.dtype, device=batch.x.device)
        count.index_add_(0, batch.graph_index, torch.ones_like(node_reward))
        return reward / count.clamp_min(1.0)

    def _copy_teacher_modules(self, module_names: list[str]) -> None:
        full = self.models["full"]
        vsbl = self.models["vsbl"]
        mapping = {
            "encoder": ("node_encoder", "edge_encoder"),
            "decoder": ("decoder",),
            "processor": ("processor",),
        }
        for name in module_names:
            for attr in mapping.get(name, (name,)):
                if hasattr(full, attr) and hasattr(vsbl, attr):
                    getattr(vsbl, attr).load_state_dict(getattr(full, attr).state_dict())

    def _save_checkpoints(self, epoch: int, metrics: dict) -> None:
        for name, model in self.models.items():
            if self.cfg.train_mode == "graph_imit":
                filename = f"graph_imit_{name}_dyn_best.pth"
            else:
                filename = f"{name}_dyn_best.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "config": self._json_config(),
                    "metrics": metrics,
                },
                self.cfg.out_dir / filename,
            )

    def _load_model(self, model: DynamicGNN, path: Path) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        state = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state)

    def _save_config(self) -> None:
        with (self.cfg.out_dir / "train_config.json").open("w", encoding="utf-8") as f:
            json.dump(self._json_config(), f, indent=2)

    def _write_history(self, history: list[dict]) -> None:
        with (self.cfg.out_dir / "history.json").open("w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

    def _json_config(self) -> dict:
        cfg = asdict(self.cfg)
        cfg["graphf"] = str(self.cfg.graphf)
        cfg["out_dir"] = str(self.cfg.out_dir)
        cfg["full_dyn_path"] = None if self.cfg.full_dyn_path is None else str(self.cfg.full_dyn_path)
        return cfg
