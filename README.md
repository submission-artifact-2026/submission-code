# EgoEmg Benchmark

This repository contains the official baseline code for the **EgoEmg dataset** and benchmark.

EgoEmg is a multimodal egocentric dataset for bimanual hand pose estimation, providing synchronized bilateral wristband EMG, IMU, egocentric RGB video, external RGB-D video, and mocap-derived MANO hand pose annotations with wrist articulation angles.

## Repository Structure

```
├── emg2pose/              # Core package (models, datasets, training)
│   ├── train.py           # Main training entrypoint
│   ├── lightning.py       # PyTorch Lightning module
│   ├── datamodule.py      # Data loading pipeline
│   ├── metrics.py         # Evaluation metrics
│   ├── test_analysis.py   # Offline EMG evaluation script
│   ├── test_analysis_fusion.py  # Offline fusion evaluation script
│   ├── models/            # Model architectures
│   │   ├── modules/       # EMGFormer, vision backbones, fusion
│   │   ├── featurizers/   # EMG featurizers (TDS, NeuroPose)
│   │   ├── decoders/      # Transformer, LSTM, MLP decoders
│   │   └── heads/         # Prediction heads
│   ├── datasets/          # EMG2Pose and EgoEMG dataset classes
│   └── UmeTrack/          # Bundled hand kinematics library
├── config/                # Hydra experiment configuration
│   ├── base.yaml          # Root config with shared defaults
│   ├── experiment/        # Per-experiment configs
│   │   ├── emgformer/     # EMGFormer on EMG2Pose and EgoEMG
│   │   ├── emg2pose/      # Traditional EMG architectures
│   │   └── fusion/        # Vision-only and EMG+vision fusion
│   ├── module/            # Model component configs
│   ├── datamodule/        # Data loading configs
│   ├── dataset/           # Dataset class configs
│   ├── transforms/        # Augmentation configs
│   └── vision_*.yaml      # Symlinks to fusion/ for --config-name access
├── scripts/               # Utility scripts
│   ├── experiments/       # Batch experiment runners
│   └── download/          # Dataset & checkpoint download scripts
├── assets/                # Normalization statistics
├── environment.yml        # Conda environment specification
└── setup.py               # Package installation
```

## Setup

### Environment

```bash
conda env create -f environment.yml && conda activate egoemg
pip install -e .
```

### Data Preparation

The benchmark uses two datasets:

1. **EMG2Pose** ([Somasundaram et al., 2024](https://arxiv.org/abs/2412.02719)): Preprocessed into memmap format for efficient loading. Available in the same folder as the EgoEMG full dataset.
   ```bash
   pip install gdown
   bash scripts/download/download_emg2pose_data.sh /path/to/data
   ```

2. **EgoEmg** (our dataset): Available in two versions:
   - **Small sample** (1 episode, ~1.1 GB memmap): [Google Drive](https://drive.google.com/drive/folders/1ON2CXvW2qbndW3b2AmCDI4jZlOJEBf5N) — suitable for quick validation and development
   - **Full dataset**: [Google Drive](https://drive.google.com/drive/folders/12C6Q1CD1uihJhx4s0Rm2s7Um76Kh8rG1) — the complete EgoEMG benchmark

   Download the memmap data:
   ```bash
   pip install gdown
   # Small sample
   gdown --folder https://drive.google.com/drive/folders/1ON2CXvW2qbndW3b2AmCDI4jZlOJEBf5N
   # Full dataset
   gdown --folder https://drive.google.com/drive/folders/12C6Q1CD1uihJhx4s0Rm2s7Um76Kh8rG1
   ```

For vision and fusion experiments, pre-cropped hand images are required. The crop preparation pipeline will be released alongside the dataset.

### Pretrained Checkpoints

Six pretrained checkpoints covering the main benchmark tasks are provided on [Google Drive](https://drive.google.com/drive/folders/1_JcHDs9uBIbFxbH0f41Sk95pCqXCcTFG):

| Checkpoint | Task | Architecture |
|-----------|------|--------------|
| `emg2pose_emgformer_small.ckpt` | EMG-to-Pose (EMG2Pose) | EMGFormer-Small |
| `egoemg_emgformer_small.ckpt` | EMG-to-Pose (EgoEMG) | EMGFormer-Small |
| `vision_resnet18.ckpt` | Vision-to-Pose | ResNet-18 |
| `vision_vit_small.ckpt` | Vision-to-Pose | ViT-Small |
| `fusion_resnet_small_emgfusion_center.ckpt` | EMG+Vision Fusion | ResNet-18 + EMGFormer-Small |
| `fusion_vit_small_emgfusion_center.ckpt` | EMG+Vision Fusion | ViT-Small + EMGFormer-Small |

Download all checkpoints:
```bash
bash scripts/download/download_checkpoints.sh
```

### Configuration

All experiment configurations use [Hydra](https://hydra.cc/) with YAML files under `config/`. Key paths must be overridden at runtime:

| Parameter | Description |
|-----------|-------------|
| `data_location` | Path to EMG2Pose dataset |
| `egoemg_memmap_dir` | Path to EgoEMG memmap directory |
| `per_episode_crops_dir` | Path to pre-cropped hand images (vision/fusion only) |
| `per_dataset_norm_stats_path` | Path to `assets/per_dataset_norm_stats.json` |

You can override these via command line or by editing the experiment config files.

## Benchmark Tasks

The benchmark defines three tasks under a shared 22-DoF joint-angle prediction target:

1. **EMG-to-Pose**: Predict hand joint angles from bilateral EMG windows
2. **Vision-to-Pose**: Predict hand joint angles from egocentric RGB hand crops
3. **EMG+Vision Fusion**: Predict hand joint angles from both modalities

Evaluation uses cross-gesture, cross-user, and combined (both) generalization splits.

## Running Experiments

### Quick Verification (no data needed)

```bash
# Dry-run all experiment configs to verify correctness
bash scripts/experiments/run_all_experiments.sh --dry_run

# Dry-run a single experiment group
bash scripts/experiments/run_all_experiments.sh --dry_run --group vision
```

### EMGFormer on EMG2Pose

```bash
python -m emg2pose.train \
  train=True eval=True \
  experiment=emgformer/emg2pose_emgformer_small \
  data_location=/path/to/emg2pose_v3

# Other model sizes:
# experiment=emgformer/emg2pose_emgformer_middle
# experiment=emgformer/emg2pose_emgformer_large
```

### EMGFormer on EgoEmg

```bash
python -m emg2pose.train \
  train=True eval=True \
  experiment=emgformer/egoemg_emgformer_small \
  egoemg_memmap_dir=/path/to/EgoEMG_memmap

# Other model sizes:
# experiment=emgformer/egoemg_emgformer_middle
# experiment=emgformer/egoemg_emgformer_large
```

### Traditional EMG Architectures on EgoEMg

```bash
# vEMG2Pose (LSTM)
python -m emg2pose.train \
  experiment=emg2pose/egoemg_vemg2pose \
  egoemg_memmap_dir=/path/to/EgoEMG_memmap

# EMG2Pose (TDS + MLP)
python -m emg2pose.train \
  experiment=emg2pose/egoemg_emg2pose \
  egoemg_memmap_dir=/path/to/EgoEMG_memmap

# NeuroPose (CNN encoder-decoder)
python -m emg2pose.train \
  experiment=emg2pose/egoemg_neuropose \
  egoemg_memmap_dir=/path/to/EgoEMG_memmap
```

### Vision-to-Pose

```bash
# ResNet backbones
python -m emg2pose.train \
  --config-name vision_resnet18 \
  egoemg_memmap_dir=/path/to/EgoEMG_memmap \
  per_episode_crops_dir=/path/to/EgoEMG_crops

# Other backbones: vision_resnet50, vision_resnet152,
#                  vision_vit_small, vision_vit_base, vision_vit_large
```

### EMG+Vision Fusion

```bash
# ResNet-18 + EMGFormer-Small residual fusion
python -m emg2pose.train \
  --config-name vision_resnet_small_emgfusion_center \
  egoemg_memmap_dir=/path/to/EgoEMG_memmap \
  per_episode_crops_dir=/path/to/EgoEMG_crops

# ViT-Small + EMGFormer-Small residual fusion
python -m emg2pose.train \
  --config-name vision_vit_small_emgfusion_center \
  egoemg_memmap_dir=/path/to/EgoEMG_memmap \
  per_episode_crops_dir=/path/to/EgoEMG_crops
```

The fusion configs automatically load pretrained vision and EMG checkpoints. Set the `vision_resnet_checkpoint` and `pretrained_emg_checkpoint` fields to point to your trained models.

### Evaluation Only

```bash
python -m emg2pose.train \
  train=False eval=True \
  experiment=emgformer/emg2pose_emgformer_small \
  checkpoint=/path/to/checkpoint.ckpt \
  data_location=/path/to/emg2pose_v3
```

### Offline Test Analysis

```bash
python -m emg2pose.test_analysis \
  experiment=emgformer/emg2pose_emgformer_middle \
  checkpoint=/path/to/checkpoint.ckpt \
  data_location=/path/to/emg2pose_v3
```

For fusion model analysis:
```bash
python -m emg2pose.test_analysis_fusion \
  --config-name vision_resnet_small_emgfusion_center \
  --checkpoint /path/to/checkpoint.ckpt
```

## Reproducing Paper Results

Key numbers from the paper, reproduced by the provided checkpoints:

| Task | Method | Checkpoint | Test MAE (rad) | Test MAE (°) | Aggregation |
|------|--------|-----------|---------------|-------------|-------------|
| EMG-to-Pose (EMG2Pose) | EMGFormer-Small | `emg2pose_emgformer_small.ckpt` | 0.2153 | 12.34° | user_stage |
| EMG-to-Pose (EgoEMG) | EMGFormer-Small | `egoemg_emgformer_small.ckpt` | 0.262 | 15.0° | sample_weighted |
| Vision-to-Pose | ResNet-18 | `vision_resnet18.ckpt` | 0.1021 | 5.85° | sample_weighted |
| Vision-to-Pose | ViT-Small | `vision_vit_small.ckpt` | 0.1052 | 6.03° | sample_weighted |
| EMG+Vision Fusion | ResNet-18 + EMGFormer-Small | `fusion_resnet_small_emgfusion_center.ckpt` | 0.0978 | 5.60° | center-frame |
| EMG+Vision Fusion | ViT-Small + EMGFormer-Small | `fusion_vit_small_emgfusion_center.ckpt` | 0.0966 | 5.53° | center-frame |

Aggregation strategy used by `test_analysis.py` and `test_analysis_fusion.py`:
**user_stage** = held-out user generalization averaged across stage splits;
**sample_weighted** = weighted average across all 6 EgoEMG splits (user/left,
user/right, gesture/left, gesture/right, both/left, both/right);
**center-frame** = MAE computed on the center frame of the sliding window,
used for vision and fusion models that predict on single frames.

## License

The baseline code is distributed under the MIT License. The EgoEmg dataset will be released under CC-BY-NC 4.0 for research use. Third-party assets (MANO model, pretrained vision backbones) remain subject to their original licenses.

## Citation

If you use this benchmark or dataset in your research, please cite:

```bibtex
@article{egoemg2026,
  title={EgoEmg: A Multimodal Egocentric Dataset with Bilateral EMG and Vision for Hand Pose Estimation},
  author={Anonymous},
  year={2026}
}
```
