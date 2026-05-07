# LiDAR-AdverseWeather-Segmentation
Robust framework that ensures reliable segmentation under challenging adverse domains

## Framework Overview

Our framework learns weather-invariant representations through a dual-branch synergistic pipeline. For each input point cloud we generate two augmented views: a weakly-augmented reference that preserves the original distribution, and a strongly-augmented view (using Selective Jittering and Learnable Point Drop) to simulate weather degradations. Both views are processed by a 3D sparse backbone (MinkowskiNet) and a projector to produce normalized embeddings. A category-wise memory bank maintains momentum-updated class prototypes from the weak view; training combines supervised cross-entropy with a contrastive loss that pulls strong-view features to their class centers and pushes them away from others.

## Environment and dependencies

The environment is recorded in [`requirements.txt`](requirements.txt). This is a **version reference**.

Try `conda create --name <env_name> --file requirements.txt` on a compatible conda install.

If that fails, create **Python 3.8** plus your CUDA toolkit, install **PyTorch 1.13** and **torchvision** for that CUDA, then **torchsparse**, **torchpack**, **tensorpack**, and the rest (**numpy**, **scipy**, **h5py**, **tqdm**, **PyYAML**, **tensorboard**, etc.). Install **torchsparse** from a matching wheel.

**Data paths:** edit [`configs/kitti2stf/default.yaml`](configs/kitti2stf/default.yaml) (and the SynLiDAR variant under `configs/synlidar2stf/` if you use that setup) so `src_dataset.root` points to your **SemanticKITTI** sequence data and `tgt_dataset.root` to **SemanticSTF**. `evaluate_by_weather.py` expects `SemanticSTF` layout including `val/val.txt` with `filename,weather` lines (`dense_fog`, `light_fog`, `snow`, `rain`).


## Training

Example (KITTI → STF, MinkUNet, voxel ratio `cr=0.5` from [`configs/kitti2stf/minkunet/cr0p5.yaml`](configs/kitti2stf/minkunet/cr0p5.yaml)):

```bash
torchpack dist-run -np 1 python train.py configs/kitti2stf/minkunet/cr0p5.yaml \
  --run-dir work_dirs/sj_plus_base_aug \
  --num_epochs 20 \
  --model.learnable_drop True \
  --batch_size 2 \
  --workers_per_gpu 8
```

- **`--run-dir`:** where logs and checkpoints are written (omit to use torchpack’s auto run directory).
- **Checkpoints:** training registers `MaxSaver('iou/test')`, so the best test-IoU weights are saved as **`checkpoints/max-iou-test.pt`** under that run directory.

With **`distributed: False`** (default in [`configs/default.yaml`](configs/default.yaml)), you can run the same arguments as **`python train.py ...`** on a single GPU if you prefer not to use `dist-run`. Evaluation is best run through **`torchpack dist-run`** because the metric helpers call **`dist.allreduce`**.

For multiple GPUs, increase `-np` to the GPU count and set `distributed: True` (and any other settings your cluster needs) via the config file or CLI overrides.

## Evaluation

Point **`--checkpoint_path`** at a concrete file, e.g. `work_dirs/sj_plus_base_aug/checkpoints/max-iou-test.pt`, or another run’s `.../checkpoints/max-iou-test.pt`. The shell does not expand `*`; use your real path or your OS’s globbing.

**1. Overall mIoU on the test split** — [`evaluate.py`](evaluate.py):

```bash
torchpack dist-run -np 1 python evaluate.py configs/kitti2stf/minkunet/cr0p5.yaml \
  --checkpoint_path work_dirs/sj_plus_base_aug/checkpoints/max-iou-test.pt \
  --model.learnable_drop True
```

Optional: `--save_pred /path/to/dir` writes per-frame prediction labels (`.label`).

**2. mIoU broken down by weather** — [`evaluate_by_weather.py`](evaluate_by_weather.py) (uses `val/val.txt` under `tgt_dataset.root`):

```bash
torchpack dist-run -np 1 python evaluate_by_weather.py configs/kitti2stf/minkunet/cr0p5.yaml \
  --checkpoint_path work_dirs/sj_plus_base_aug/checkpoints/max-iou-test.pt \
  --model.learnable_drop True
```

Keep **`configs.model.name`** consistent with training (`minkunet`); the scripts assert this. For SynLiDAR → STF experiments, swap the config path for `configs/synlidar2stf/minkunet/cr0p5.yaml` (and matching data roots).

