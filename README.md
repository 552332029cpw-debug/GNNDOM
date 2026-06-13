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
  --env-shape None \
  --device cpu
```

Expected output includes a line like:

```text
[SMOKE] ok ...
```

## Generate Rollout Dataset

By default, dataset generation expects an Isaac camera interface and uses
`--observation-mode isaac_camera`. For full-graph smoke runs without camera
observation, pass `--observation-mode full`.

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
  --iterations 1
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
step file. With `--save-rgbd`, it also stores `rgb` and `depth`.

## Build Graph Transitions

```bash
python gnndom_graph/cli_build.py \
  --dataf data/smoke \
  --graphf data/smoke_graphs \
  --n-his 5 \
  --pred-time-interval 1 \
  --neighbor-radius 0.045
```

This creates graph transition files under:

```text
data/smoke_graphs/train/
data/smoke_graphs/valid/
```

## Train Dynamic GNN

The Dynamic GNN supports the ManiFabric modes `vsbl`, `full`, and `graph_imit`.

Visible/student mode:

```bash
python gnndom_model/cli_train.py \
  --graphf data/smoke_graphs \
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
cloth mesh adjacencies.

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
