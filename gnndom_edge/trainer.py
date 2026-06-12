# SPDX-License-Identifier: BSD-3-Clause
"""Training loop for GNNDOM EdgeGNN."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .dataset import EdgeBatch, EdgeGraphDataset, collate_edge_graphs
from .model import EdgeGNN


@dataclass
class EdgeTrainConfig:
    graphf: Path
    out_dir: Path
    epochs: int = 1000
    batch_size: int = 16
    lr: float = 1.0e-4
    beta1: float = 0.9
    device: str = "cpu"
    neighbor_radius: float = 0.045
    state_dim: int = 3
    relation_dim: int = 4
    global_size: int = 128
    proc_layer: int = 10
    weight_decay: float = 0.0
    num_workers: int = 0
    max_train_graphs: int | None = None
    max_valid_graphs: int | None = None
    edge_model_path: Path | None = None
    load_optim: bool = False
    seed: int = 0


class EdgeTrainer:
    def __init__(self, cfg: EdgeTrainConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        torch.manual_seed(cfg.seed)
        self.cfg.out_dir.mkdir(parents=True, exist_ok=True)
        self.train_loader = self._make_loader("train", shuffle=True, max_graphs=cfg.max_train_graphs)
        self.valid_loader = self._make_loader("valid", shuffle=False, max_graphs=cfg.max_valid_graphs)
        self.model = EdgeGNN(
            state_dim=cfg.state_dim,
            relation_dim=cfg.relation_dim,
            hidden_dim=cfg.global_size,
            proc_layer=cfg.proc_layer,
            global_size=cfg.global_size,
        ).to(self.device)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=cfg.lr, betas=(cfg.beta1, 0.999), weight_decay=cfg.weight_decay)
        self.best_valid = float("inf")
        self.start_epoch = 1
        if cfg.edge_model_path is not None:
            self._load_checkpoint(cfg.edge_model_path, load_optim=cfg.load_optim)
        self._write_config()

    def train(self) -> dict:
        history = []
        for epoch in range(self.start_epoch, self.cfg.epochs + 1):
            train_metrics = self._run_epoch(train=True)
            valid_metrics = self._run_epoch(train=False)
            row = {"epoch": epoch, **{f"train_{k}": v for k, v in train_metrics.items()}, **{f"valid_{k}": v for k, v in valid_metrics.items()}}
            history.append(row)
            print(
                f"[EDGE {epoch:04d}] "
                f"train_loss={row['train_loss']:.6g} valid_loss={row['valid_loss']:.6g} "
                f"valid_acc={row['valid_accuracy']:.4f} valid_precision={row['valid_precision']:.4f} valid_recall={row['valid_recall']:.4f}"
            )
            if row["valid_loss"] < self.best_valid:
                self.best_valid = row["valid_loss"]
                self._save_checkpoint(epoch, row)
            self._write_history(history)
        return {"best_valid_loss": self.best_valid, "history": history}

    def _make_loader(self, phase: str, *, shuffle: bool, max_graphs: int | None) -> DataLoader:
        dataset = EdgeGraphDataset(self.cfg.graphf, phase, neighbor_radius=self.cfg.neighbor_radius, max_graphs=max_graphs)
        return DataLoader(
            dataset,
            batch_size=self.cfg.batch_size,
            shuffle=shuffle,
            num_workers=self.cfg.num_workers,
            collate_fn=collate_edge_graphs,
        )

    def _run_epoch(self, *, train: bool) -> dict[str, float]:
        self.model.train(train)
        loader = self.train_loader if train else self.valid_loader
        totals = {"loss": 0.0, "accuracy": 0.0, "precision": 0.0, "recall": 0.0}
        num_batches = 0
        for batch in loader:
            batch = batch.to(self.device)
            if train:
                self.optim.zero_grad(set_to_none=True)
            with torch.set_grad_enabled(train):
                logits = self.model(batch.x, batch.edge_index, batch.edge_attr, batch.graph_index)["mesh_edge"]
                loss = F.binary_cross_entropy_with_logits(logits, batch.gt_mesh_edge)
                if train:
                    loss.backward()
                    self.optim.step()
            metrics = edge_metrics(logits.detach(), batch.gt_mesh_edge)
            totals["loss"] += float(loss.detach().cpu())
            for key, value in metrics.items():
                totals[key] += value
            num_batches += 1
        return {key: value / max(1, num_batches) for key, value in totals.items()}

    def _save_checkpoint(self, epoch: int, metrics: dict) -> None:
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optim.state_dict(),
                "config": self._json_config(),
                "metrics": metrics,
            },
            self.cfg.out_dir / "edge_gnn_best.pth",
        )

    def _load_checkpoint(self, path: Path, *, load_optim: bool) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        state = checkpoint.get("model_state_dict", checkpoint)
        self.model.load_state_dict(state)
        if load_optim and "optimizer_state_dict" in checkpoint:
            self.optim.load_state_dict(checkpoint["optimizer_state_dict"])
        if "epoch" in checkpoint:
            self.start_epoch = int(checkpoint["epoch"]) + 1

    def _write_config(self) -> None:
        with (self.cfg.out_dir / "edge_train_config.json").open("w", encoding="utf-8") as f:
            json.dump(self._json_config(), f, indent=2)

    def _write_history(self, history: list[dict]) -> None:
        with (self.cfg.out_dir / "edge_history.json").open("w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

    def _json_config(self) -> dict:
        cfg = asdict(self.cfg)
        cfg["graphf"] = str(self.cfg.graphf)
        cfg["out_dir"] = str(self.cfg.out_dir)
        cfg["edge_model_path"] = None if self.cfg.edge_model_path is None else str(self.cfg.edge_model_path)
        return cfg


def edge_metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    pred = (torch.sigmoid(logits) > 0.5).to(labels.dtype)
    labels = (labels > 0.5).to(labels.dtype)
    correct = (pred == labels).float().mean()
    true_pos = ((pred == 1) & (labels == 1)).float().sum()
    pred_pos = (pred == 1).float().sum()
    label_pos = (labels == 1).float().sum()
    precision = true_pos / pred_pos.clamp_min(1.0)
    recall = true_pos / label_pos.clamp_min(1.0)
    return {
        "accuracy": float(correct.cpu()),
        "precision": float(precision.cpu()),
        "recall": float(recall.cpu()),
    }
