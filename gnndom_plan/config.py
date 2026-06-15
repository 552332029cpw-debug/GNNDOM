# SPDX-License-Identifier: BSD-3-Clause
"""Configuration for online GNNDOM planning."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class PlanConfig:
    dyn_path: Path
    log_dir: Path = Path("runs/plan")
    edge_model_path: Path | None = None
    observation_mode: str = "isaac_camera"
    configurations: int = 1
    sampling_num: int = 10
    shooting_number: int = 9
    seed: int = 43
    device: str = "cpu"

    cloth_xdim: int = 48
    cloth_ydim: int = 48
    cloth_particle_radius: float = 0.00625
    cloth_mass: float = 0.1
    cloth_stiffness: tuple[float, float, float] = (0.9, 1.0, 0.9)
    env_shape: str | None = None
    target_type: str = "flat"
    vary_cloth_size: bool = False
    vary_stiffness: bool = False
    vary_mass: bool = False
    vary_orientation: bool = False

    dt: float = 1.0 / 60.0
    pred_time_interval: int = 1
    n_his: int = 5
    down_sample_scale: int = 3
    neighbor_radius: float = 0.045
    voxel_size: float = 0.0216
    visibility_threshold: float = 0.05
    use_mesh_edge: bool = True
    use_es: bool = True
    edge_threshold: float = 0.5

    camera_width: int = 480
    camera_height: int = 480
    camera_fov: float = 100.0
    camera_pos: tuple[float, float, float] = (1.45, -0.85, 0.95)
    camera_look_at: tuple[float, float, float] = (0.32, -0.08, 0.22)
    camera_near: float = 0.01
    camera_far: float = 5.0
    save_rgbd: bool = False

    runtime_device: str | None = None
    fps: int = 60
    substeps: int = 8
    iterations: int = 8
    settle_steps: int = 420
    min_stable_steps: int = 100
    velocity_threshold: float = 0.03

    swing_acc: float = 2.0
    pull_acc: float = 1.0
    drop_steps: int = 3
    control_steps: int | None = None

    def to_json_dict(self) -> dict:
        data = asdict(self)
        data["dyn_path"] = str(self.dyn_path)
        data["log_dir"] = str(self.log_dir)
        data["edge_model_path"] = None if self.edge_model_path is None else str(self.edge_model_path)
        return data

    def validate(self) -> None:
        if self.observation_mode not in {"isaac_camera", "geometry_camera"}:
            raise ValueError("observation_mode must be isaac_camera or geometry_camera")
        if not self.dyn_path.exists():
            raise FileNotFoundError(f"Dynamic checkpoint not found: {self.dyn_path}")
        if self.edge_model_path is not None and not self.edge_model_path.exists():
            raise FileNotFoundError(f"EdgeGNN checkpoint not found: {self.edge_model_path}")
        if self.pred_time_interval < 1:
            raise ValueError("pred_time_interval must be >= 1")
        if self.configurations < 1:
            raise ValueError("configurations must be >= 1")
        if self.sampling_num < 1:
            raise ValueError("sampling_num must be >= 1")
        if self.shooting_number < 1:
            raise ValueError("shooting_number must be >= 1")
