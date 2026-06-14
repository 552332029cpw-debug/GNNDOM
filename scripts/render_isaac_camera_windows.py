# SPDX-License-Identifier: BSD-3-Clause
r"""Offline Windows Isaac Sim camera sampler for existing GNNDOM rollouts."""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path
from typing import Iterable

np = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataf", type=Path, required=True)
    parser.add_argument("--phases", nargs="+", default=("train", "valid"))
    parser.add_argument("--rollouts", nargs="*", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--save-rgbd", action="store_true")
    parser.add_argument("--camera-width", type=int, default=480)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--camera-fov", type=float, default=100.0)
    parser.add_argument("--camera-pos", nargs=3, type=float, default=(1.45, -0.85, 0.95))
    parser.add_argument("--camera-look-at", nargs=3, type=float, default=(0.32, -0.08, 0.22))
    parser.add_argument("--voxel-size", type=float, default=0.0216)
    parser.add_argument("--visibility-threshold", type=float, default=0.05)
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--renderer", default="RaytracedLighting")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    global np
    import numpy as _np

    np = _np
    log_path = args.dataf / "_isaac_camera_render_log.txt"
    status(log_path, f"start dataf={args.dataf} exists={args.dataf.exists()}")

    from isaacsim import SimulationApp

    app = None
    try:
        app = SimulationApp(
            {
                "headless": bool(args.headless),
                "renderer": str(args.renderer),
                "width": int(args.camera_width),
                "height": int(args.camera_height),
            }
        )
        status(log_path, "SimulationApp ready")
        renderer = Renderer(args, log_path)
        total = 0
        for rollout_dir in iter_rollouts(args.dataf, args.phases, args.rollouts):
            total += renderer.process_rollout(rollout_dir)
        status(log_path, f"updated {total} step files")
    except Exception:
        text = traceback.format_exc()
        status(log_path, "ERROR:\n" + text)
        print(text, flush=True)
        return 1
    finally:
        if app is not None:
            app.close()
    return 0


def status(path: Path, msg: str) -> None:
    print(f"[ISAAC_CAMERA] {msg}", flush=True)
    try:
        with path.open("a", encoding="utf-8") as f:
            f.write(msg.rstrip() + "\n")
    except Exception:
        pass


def iter_rollouts(dataf: Path, phases: Iterable[str], rollouts: list[int] | None):
    selected = None if rollouts is None else {int(v) for v in rollouts}
    for phase in phases:
        phase_dir = dataf / phase
        if not phase_dir.exists():
            continue
        for rollout_dir in sorted([p for p in phase_dir.iterdir() if p.is_dir() and p.name.isdigit()], key=lambda p: int(p.name)):
            if selected is None or int(rollout_dir.name) in selected:
                yield rollout_dir


class Renderer:
    def __init__(self, args: argparse.Namespace, log_path: Path):
        self.args = args
        self.log_path = log_path
        self.mesh = None
        self.topology = None
        self._setup_scene()

    def _setup_scene(self) -> None:
        import omni.replicator.core as rep
        from pxr import UsdGeom, UsdLux

        stage = current_stage()
        UsdGeom.Xform.Define(stage, "/World/GNNDOM")
        light = UsdLux.DomeLight.Define(stage, "/World/GNNDOM/Light")
        light.CreateIntensityAttr(2500.0)
        camera = rep.create.camera(
            position=tuple(float(v) for v in self.args.camera_pos),
            look_at=tuple(float(v) for v in self.args.camera_look_at),
        )
        self.render_product = rep.create.render_product(camera, (int(self.args.camera_width), int(self.args.camera_height)))
        self.pointcloud_annotator = make_pointcloud_annotator(rep)
        self.depth_annotator = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane", device="cpu")
        self.rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
        self.pointcloud_annotator.attach(self.render_product)
        self.depth_annotator.attach(self.render_product)
        self.rgb_annotator.attach(self.render_product)
        for _ in range(3):
            step_kit()

    def process_rollout(self, rollout_dir: Path) -> int:
        info_path = rollout_dir / "rollout_info.json"
        with info_path.open("r", encoding="utf-8") as f:
            info = json.load(f)
        cloth_size = info.get("ClothSize")
        if cloth_size is None:
            raise RuntimeError(f"{info_path} missing ClothSize")
        self.ensure_mesh(int(cloth_size[0]), int(cloth_size[1]))
        downsample_idx = np.asarray(info["downsample_idx"], dtype=np.int64)
        updated = 0
        for step_path in sorted(rollout_dir.glob("*.npz"), key=lambda p: int(p.stem)):
            if self.process_step(step_path, downsample_idx):
                updated += 1
        status(self.log_path, f"{rollout_dir}: updated {updated} steps")
        return updated

    def ensure_mesh(self, cloth_xdim: int, cloth_ydim: int) -> None:
        if self.topology == (cloth_xdim, cloth_ydim):
            return
        from pxr import Gf, Sdf, UsdGeom

        faces = triangle_indices(cloth_xdim, cloth_ydim)
        self.mesh = UsdGeom.Mesh.Define(current_stage(), "/World/GNNDOM/Cloth")
        self.mesh.GetPointsAttr().Set([Gf.Vec3f(0.0, 0.0, 0.0)] * (cloth_xdim * cloth_ydim))
        self.mesh.GetFaceVertexCountsAttr().Set([3] * len(faces))
        self.mesh.GetFaceVertexIndicesAttr().Set(faces.reshape(-1).tolist())
        self.mesh.CreateDoubleSidedAttr(True)
        self.mesh.GetPrim().CreateAttribute("primvars:displayColor", Sdf.ValueTypeNames.Color3fArray).Set([Gf.Vec3f(0.35, 0.42, 0.95)])
        self.mesh.GetPrim().CreateAttribute("primvars:displayColor:interpolation", Sdf.ValueTypeNames.Token).Set("constant")
        self.topology = (cloth_xdim, cloth_ydim)
        step_kit()

    def process_step(self, step_path: Path, downsample_idx) -> bool:
        data = load_npz(step_path)
        if not self.args.overwrite and {"pointcloud", "partial_pc_mapped_idx", "downsample_observable_idx"}.issubset(data):
            return False
        positions = np.asarray(data["positions"], dtype=np.float32).reshape(-1, 3)
        self.update_mesh(positions)
        pointcloud, rgb, depth = self.capture()
        visible = visible_observation(pointcloud, positions[downsample_idx], self.args.voxel_size, self.args.visibility_threshold)
        data["pointcloud"] = visible["pointcloud"]
        data["partial_pc_mapped_idx"] = visible["partial_pc_mapped_idx"]
        data["downsample_observable_idx"] = visible["downsample_observable_idx"]
        data["camera_depth_positive_pixel_count"] = np.asarray(depth_positive_pixel_count(depth), dtype=np.int64)
        data["camera_raw_pointcloud_count"] = np.asarray(len(pointcloud), dtype=np.int64)
        data["camera_voxel_pointcloud_count"] = np.asarray(visible["voxel_pointcloud_count"], dtype=np.int64)
        data["camera_matched_pointcloud_count"] = np.asarray(len(visible["pointcloud"]), dtype=np.int64)
        if self.args.save_rgbd:
            data["rgb"] = rgb
            data["depth"] = depth
        save_npz(step_path, data)
        status(
            self.log_path,
            (
                f"step {step_path}: depth_pos={int(data['camera_depth_positive_pixel_count'])} "
                f"raw_pc={int(data['camera_raw_pointcloud_count'])} "
                f"voxel_pc={int(data['camera_voxel_pointcloud_count'])} "
                f"matched_pc={int(data['camera_matched_pointcloud_count'])} "
                f"visible={len(data['downsample_observable_idx'])}"
            ),
        )
        return True

    def update_mesh(self, positions) -> None:
        from pxr import Gf

        self.mesh.GetPointsAttr().Set([Gf.Vec3f(float(x), float(y), float(z)) for x, y, z in positions])
        step_kit()

    def capture(self):
        import omni.replicator.core as rep

        rep.orchestrator.step()
        step_kit()
        pc = np.asarray(annotator_array(self.pointcloud_annotator.get_data()), dtype=np.float32).reshape(-1, 3)
        if len(pc):
            pc = pc[np.all(np.isfinite(pc), axis=1)]
        rgb = np.asarray(annotator_array(self.rgb_annotator.get_data()))
        depth = np.asarray(annotator_array(self.depth_annotator.get_data()), dtype=np.float32)
        return pc.astype(np.float32), rgb, depth


def current_stage():
    import omni.usd

    return omni.usd.get_context().get_stage()


def step_kit() -> None:
    import omni.kit.app

    omni.kit.app.get_app().update()


def make_pointcloud_annotator(rep):
    for kwargs in (
        {"device": "cpu", "init_params": {"includeUnlabelled": True}},
        {"device": "cpu", "init_params": {"include_unlabelled": True}},
        {"device": "cpu"},
    ):
        try:
            return rep.AnnotatorRegistry.get_annotator("pointcloud", **kwargs)
        except Exception:
            pass
    raise RuntimeError("Could not create Isaac pointcloud annotator")


def annotator_array(value):
    if isinstance(value, dict):
        return value.get("data", next(iter(value.values())))
    return value


def load_npz(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k].copy() for k in data.files}


def save_npz(path: Path, data: dict) -> None:
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **{k: np.asarray(v) for k, v in data.items()})
    tmp.replace(path)


def visible_observation(pointcloud, downsampled_positions, voxel_size: float, threshold: float) -> dict:
    vox = voxelize(pointcloud, voxel_size)
    if len(vox) == 0:
        return empty_visible(0)
    downsampled_positions = np.asarray(downsampled_positions, dtype=np.float32).reshape(-1, 3)
    dist = np.linalg.norm(vox[:, None, :] - downsampled_positions[None, :, :], axis=-1)
    mapped = np.argmin(dist, axis=1).astype(np.int64)
    keep = dist[np.arange(len(vox)), mapped] < float(threshold)
    mapped = mapped[keep]
    filtered = vox[keep]
    if len(filtered) == 0:
        return empty_visible(len(vox))
    return {
        "pointcloud": filtered.astype(np.float32),
        "partial_pc_mapped_idx": mapped.astype(np.int64),
        "downsample_observable_idx": np.unique(mapped).astype(np.int64),
        "voxel_pointcloud_count": int(len(vox)),
    }


def empty_visible(voxel_count: int = 0) -> dict:
    return {
        "pointcloud": np.empty((0, 3), dtype=np.float32),
        "partial_pc_mapped_idx": np.empty((0,), dtype=np.int64),
        "downsample_observable_idx": np.empty((0,), dtype=np.int64),
        "voxel_pointcloud_count": int(voxel_count),
    }


def voxelize(points, voxel_size: float):
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    if len(points) == 0:
        return points
    keys = np.floor(points / max(float(voxel_size), 1e-8)).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    return points[np.sort(idx)].astype(np.float32)


def depth_positive_pixel_count(depth) -> int:
    depth = np.asarray(depth, dtype=np.float32)
    if depth.size == 0:
        return 0
    return int(np.count_nonzero(np.isfinite(depth) & (depth > 0.0)))


def triangle_indices(cloth_xdim: int, cloth_ydim: int):
    faces = []
    for y in range(cloth_ydim - 1):
        for x in range(cloth_xdim - 1):
            i0 = y * cloth_xdim + x
            i1 = i0 + 1
            i2 = i0 + cloth_xdim
            i3 = i2 + 1
            faces.append((i0, i1, i2))
            faces.append((i1, i3, i2))
    return np.asarray(faces, dtype=np.int64)


if __name__ == "__main__":
    raise SystemExit(main())
