# SPDX-License-Identifier: BSD-3-Clause
"""Rollout dataset storage helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def _jsonify(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(v) for v in value]
    return value


def save_rollout_info(rollout_dir: Path, info: dict) -> Path:
    rollout_dir.mkdir(parents=True, exist_ok=True)
    path = rollout_dir / "rollout_info.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(_jsonify(info), f, indent=2)
    return path


def save_step(rollout_dir: Path, timestep: int, step_data: dict) -> Path:
    rollout_dir.mkdir(parents=True, exist_ok=True)
    path = rollout_dir / f"{int(timestep)}.npz"
    arrays = {key: np.asarray(value) for key, value in step_data.items()}
    np.savez_compressed(path, **arrays)
    return path


def load_step(rollout_dir: Path, timestep: int) -> dict[str, np.ndarray]:
    path = rollout_dir / f"{int(timestep)}.npz"
    data = np.load(path)
    return {key: data[key] for key in data.files}

