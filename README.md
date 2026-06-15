# GNNDOM

GNNDOM is a ManiFabric-style cloth manipulation reproduction focused on the
software pipeline before real-robot experiments. The current implementation
uses a z-up coordinate convention:

```text
x: horizontal axis
y: horizontal axis
z: height above ground
```

The code in this repository is organized so a remote GPU machine can generate
rollout data, build graph transitions, and train the Dynamic GNN / EdgeGNN
models.

## Modules

```text
gnndom_env/       Newton-backed ClothDrop environment and ManiFabric sampling rules
gnndom_dataset/   rollout dataset generation in ManiFabric-style directories
gnndom_graph/     rollout-to-graph transition builder
gnndom_model/     Dynamic GNN training: vsbl, full, graph_imit
gnndom_edge/      EdgeGNN mesh-edge classifier
scripts/          small smoke checks
```

Generated data and training outputs are intentionally ignored by git:

```text
data/
runs/
```

## Remote Setup

Clone the repository on the remote machine:

```bash
git clone https://github.com/552332029cpw-debug/GNNDOM.git
cd GNNDOM
```

Create a training environment:

```bash
python3 -m venv .venv-train
source .venv-train/bin/activate
pip install --upgrade pip
pip install torch torchvision torchaudio numpy scipy
```

Check that PyTorch can see the GPU:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu only')"
```

The environment and dataset generation modules also require the Newton backend.
Place or install the project Newton backend so it is available at:

```text
../newton
```

or adjust `gnndom_env/newton_backend.py` to point at the local Newton checkout.

## Environment Smoke Test

```bash
python scripts/smoke_env.py \
  --cloth-xdim 4 \
  --cloth-ydim 4 \
  --env-shape sphere \
  --device cpu
```

Expected output includes a line like:

```text
[SMOKE] ok ...
```

## Target Semantics

`target_pos` is the physically settled cloth state, not the raw geometric
flat/fold placement. The environment first builds the obstacle scene, places
the cloth in `geometric_target_pos`, releases both pickers
(`target_release_grasp = 0`), and simulates until the cloth settles. This
matches ManiFabric's target-generation path and keeps obstacle scenes reachable:
for example, a sphere target becomes cloth draped on the sphere rather than a
flat sheet floating through it.

The dataset and online planner preserve both states:

```text
target_pos             physical release-and-settle target used for rewards/training
geometric_target_pos   pre-settle flat/fold seed, saved only for debug/visualization
target_picker_pos      geometric picker goal used to generate the fling path
target_settle_steps    number of simulation steps used by target settling
target_source          physical_settled
```

Because this changes the supervision target, regenerate datasets, graph files,
and checkpoints made before this alignment change.

To inspect one sampled scene numerically:

```bash
python scripts/check_physical_target.py \
  --env-shape sphere \
  --device cpu
```

To view the obstacle and target release-and-settle process:

```bash
python scripts/view_physical_target.py \
  --env-shape sphere \
  --target-type flat \
  --device cuda \
  --viewer gl
```

The viewer starts paused with the cloth held at `geometric_target_pos` by the
target pickers. Press Space to release both pickers and simulate the fall into
the physical `target_pos`. It shows only one cloth by default; pass
`--show-geometric-target` only when you also want to overlay the pre-settle
geometric debug mesh. For non-interactive smoke runs, pass `--no-start-paused`.

Contact tuning matters for obstacle targets. The Newton VBD particle-rigid
contact path averages cloth-side and shape-side material properties, then caps
the warm-start contact penalty. GNNDOM therefore sets both soft and rigid shape
contact material from:

```bash
--contact-ke 1e5 --contact-kd 1e-2 --contact-mu 2.0
```

Raising only `--contact-mu` is usually not enough if the normal contact penalty
is too low, because friction is limited by the normal load.

Target and obstacle geometry can be overridden from the CLI. For example:

```bash
python scripts/view_physical_target.py \
  --target-type fold \
  --env-shape sphere \
  --x-target 0.08 \
  --rot-angle 0.2 \
  --shape-size 0.12 0.12 0.12 \
  --device cuda \
  --viewer gl
```

Use `--target-type flat|fold|random`, `--env-shape None|platform|sphere|rod|table|random|all`,
`--shape-size ...`, and `--shape-pos x y z` to inspect specific scenes before
regenerating data with the same flags.

For `rod`, `--shape-size radius half_length` controls the horizontal capsule
rod. The default is `0.005 0.25`.

Rod scenes often need explicit velocity damping because a hanging cloth behaves
like a pendulum. Start with:

```bash
python scripts/view_physical_target.py \
  --env-shape rod \
  --shape-size 0.006 0.30 \
  --self-contact \
  --substeps 16 \
  --iterations 16 \
  --contact-ke 5e4 \
  --contact-kd 5e-2 \
  --contact-mu 1.5 \
  --air-drag 4.0 \
  --device cuda \
  --viewer gl
```

## Generate Rollout Dataset

By default, dataset generation expects an Isaac camera interface and uses
`--observation-mode isaac_camera`. For full-graph smoke runs without camera
observation, pass `--observation-mode full`.

The camera observation path follows ManiFabric's partial-observation flow:

```text
Isaac RGBD/depth camera
  -> back-project depth to z-up world pointcloud
  -> voxelize pointcloud
  -> match visible pointcloud to downsampled cloth particles
  -> save pointcloud, downsample_observable_idx, partial_pc_mapped_idx
```

This corresponds to ManiFabric's `get_rgbd() -> get_world_coords() ->
get_observable_particle_index_old()/get_observable_particle_index_3()` path.
The minimum Isaac camera interface is:

```python
get_depth()              # (H, W) float depth
get_camera_intrinsics()  # (3, 3)
get_camera_to_world()    # (4, 4)
```

or a combined:

```python
get_rgbd()               # (H, W, 4), rgb in [:3], depth in [3]
```

```bash
python gnndom_dataset/cli_generate.py \
  --dataf data/smoke \
  --n-rollout 2 \
  --train-valid-ratio 0.5 \
  --cloth-xdim 4 \
  --cloth-ydim 4 \
  --device cpu \
  --observation-mode full \
  --substeps 1 \
  --iterations 1 \
  --settle-steps 420 \
  --min-stable-steps 100
```

This creates:

```text
data/smoke/train/0/rollout_info.json
data/smoke/train/0/0.npz
data/smoke/valid/0/...
```

Isaac camera dataset generation:

```bash
python gnndom_dataset/cli_generate.py \
  --dataf data/camera_smoke \
  --n-rollout 2 \
  --train-valid-ratio 0.5 \
  --cloth-xdim 4 \
  --cloth-ydim 4 \
  --device cuda \
  --observation-mode isaac_camera \
  --save-rgbd
```

The camera path stores `pointcloud` and `downsample_observable_idx` in every
step file. It also stores `partial_pc_mapped_idx` for later visible graph and
EdgeGNN training. With `--save-rgbd`, it additionally stores `rgb` and `depth`.

## Build Graph Transitions

Full graph mode is the default and uses the known downsampled cloth topology:

```bash
python gnndom_graph/cli_build.py \
  --dataf data/smoke \
  --graphf data/smoke_graphs \
  --graph-mode full \
  --n-his 5 \
  --pred-time-interval 1 \
  --neighbor-radius 0.045
```

This creates graph transition files under:

```text
data/smoke_graphs/train/
data/smoke_graphs/valid/
```

Visible graph mode follows ManiFabric's `vsbl` branch. It consumes the camera
fields saved in each step file:

```text
pointcloud
downsample_observable_idx
partial_pc_mapped_idx
```

and builds graph nodes on the visible pointcloud while supervising labels from
the matched downsampled cloth particles:

```bash
python gnndom_graph/cli_build.py \
  --dataf data/camera_smoke \
  --graphf data/camera_smoke_vsbl_graphs \
  --graph-mode vsbl \
  --n-his 5 \
  --pred-time-interval 1 \
  --neighbor-radius 0.045
```

By default, `vsbl` graph construction uses the simulator-visible particle
mapping to recover ground-truth visible mesh edges. This matches ManiFabric's
offline fallback path. To use a trained EdgeGNN to classify visible mesh edges
instead, pass an EdgeGNN checkpoint:

```bash
python gnndom_graph/cli_build.py \
  --dataf data/camera_smoke \
  --graphf data/camera_smoke_vsbl_edgegnn_graphs \
  --graph-mode vsbl \
  --edge-model-path runs/edge/edge_gnn_best.pth \
  --edge-device cuda \
  --edge-threshold 0.5
```

Each saved visible graph records `mesh_edge_source` as either
`visible_ground_truth`, `edgegnn`, or `none`.

To build both graph types from the same rollout dataset:

```bash
python gnndom_graph/cli_build.py \
  --dataf data/camera_smoke \
  --graphf data/camera_smoke_graphs \
  --graph-mode both
```

This writes:

```text
data/camera_smoke_graphs/full/train/...
data/camera_smoke_graphs/vsbl/train/...
```

## Train Dynamic GNN

The Dynamic GNN supports the ManiFabric modes `vsbl`, `full`, and `graph_imit`.
The default training budget is 100 epochs with early stopping:

```text
--epochs 100
--patience 20
--min-delta 1e-5
```

Visible/student mode:

```bash
python gnndom_model/cli_train.py \
  --graphf data/camera_smoke_vsbl_graphs \
  --out-dir runs/smoke_vsbl \
  --train-mode vsbl \
  --epochs 1 \
  --batch-size 2 \
  --device cuda \
  --proc-layer 1 \
  --global-size 32
```

Full/teacher mode:

```bash
python gnndom_model/cli_train.py \
  --graphf data/smoke_graphs \
  --out-dir runs/smoke_full \
  --train-mode full \
  --epochs 1 \
  --batch-size 2 \
  --device cuda \
  --proc-layer 1 \
  --global-size 32
```

Graph imitation mode:

```bash
python gnndom_model/cli_train.py \
  --graphf data/smoke_graphs \
  --out-dir runs/smoke_imit \
  --train-mode graph_imit \
  --epochs 1 \
  --batch-size 2 \
  --device cuda \
  --proc-layer 1 \
  --global-size 32 \
  --tune-teach
```

For production graph imitation, train or load a full teacher first and pass:

```bash
--full-dyn-path runs/full/full_dyn_best.pth
```

## Train EdgeGNN

EdgeGNN rebuilds pointcloud radius edges and predicts which candidate edges are
cloth mesh adjacencies. It also defaults to 100 epochs with the same early
stopping settings:

```text
--epochs 100
--patience 20
--min-delta 1e-5
```

```bash
python gnndom_edge/cli_train_edge.py \
  --graphf data/smoke_graphs \
  --out-dir runs/smoke_edge \
  --epochs 1 \
  --batch-size 2 \
  --device cuda \
  --proc-layer 1 \
  --global-size 32 \
  --max-train-graphs 4 \
  --max-valid-graphs 4
```

## Suggested Larger Run

After all smoke checks pass, increase rollout count and model size:

```bash
python gnndom_dataset/cli_generate.py \
  --dataf data/train_v1 \
  --n-rollout 100 \
  --train-valid-ratio 0.9 \
  --device cuda

python gnndom_graph/cli_build.py \
  --dataf data/train_v1 \
  --graphf data/train_v1_graphs

python gnndom_model/cli_train.py \
  --graphf data/train_v1_graphs \
  --out-dir runs/vsbl_v1 \
  --train-mode vsbl \
  --epochs 100 \
  --patience 20 \
  --batch-size 8 \
  --device cuda \
  --proc-layer 10 \
  --global-size 128
```

## Notes

- The public schema is z-up. No y-up SoftGym coordinates are exposed.
- The first version of `vsbl` uses the current graph schema as a visible-compatible
  graph. A separate true partial-visible graph builder can be added later.
- Generated datasets and checkpoints can be large; keep them out of git unless
  they are tiny smoke fixtures.
