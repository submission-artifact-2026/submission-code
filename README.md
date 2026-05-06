# EgoEmg Benchmark

This repository contains the official baseline code for the **EgoEmg dataset** and benchmark, submitted to NeurIPS 2026 Datasets & Benchmarks Track.

EgoEmg is a multimodal egocentric dataset for bimanual hand pose estimation, providing synchronized bilateral wristband EMG, IMU, egocentric RGB video, external RGB-D video, and mocap-derived MANO hand pose annotations with wrist articulation angles.

## Repository Structure

```
├── emg2pose/              # Core package (models, datasets, training)
│   ├── train.py           # Main training entrypoint
│   ├── lightning.py       # PyTorch Lightning module
│   ├── datamodule.py      # Data loading pipeline
│   ├── metrics.py         # Evaluation metrics
│   ├── test_analysis.py   # Offline evaluation script
│   ├── models/            # Model architectures
│   │   ├── modules/       # EMGFormer, vision backbones, fusion
│   │   ├── featurizers/   # EMG featurizers (TDS, NeuroPose)
│   │   └── decoders/      # Transformer, LSTM, MLP decoders
│   └── datasets/          # EMG2Pose and EgoEMG dataset classes
├── config/                # Hydra experiment configuration
│   ├── experiment/        # Per-experiment configs
│   │   ├── emgformer/     # EMGFormer on EMG2Pose and EgoEMG
│   │   ├── emg2pose/      # Traditional EMG architectures
│   │   └── fusion/        # Vision-only and EMG+vision fusion
│   └── module/            # Model component configs
├── test_results/          # Reproduced experiment results
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

1. **EMG2Pose** ([Somasundaram et al., 2024](https://arxiv.org/abs/2412.02719)): Preprocessed into memmap format for efficient loading. Place the dataset and its memmap version at a location accessible to the training scripts.

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
# Dry-run all 23 experiment configs to verify correctness
bash scripts/experiments/run_all_experiments.sh --dry_run

# Dry-run a single experiment group
bash scripts/experiments/run_all_experiments.sh --dry_run --group vision
```

### EMGFormer on EMG2Pose

```bash
python -m emg2pose.train \
  train=True eval=True \
  experiment=emgformer/regression_emgformer_small_aggressive \
  data_location=/path/to/emg2pose_v3

# Other model sizes:
# experiment=emgformer/regression_emgformer_middle_aggressive
# experiment=emgformer/regression_emgformer_large_aggressive
```

### EMGFormer on EgoEmg

```bash
# With augmentation
python -m emg2pose.train \
  train=True eval=True \
  experiment=emgformer/regression_emgformer_small_aggressive_egoemg \
  egoemg_memmap_dir=/path/to/EgoEMG_memmap

# Without augmentation (scratch training)
python -m emg2pose.train \
  train=True eval=True \
  experiment=emgformer/regression_emgformer_small_aggressive_egoemg_wo_aug \
  egoemg_memmap_dir=/path/to/EgoEMG_memmap
```

### Traditional EMG Architectures on EgoEMg

```bash
# vEMG2Pose (LSTM)
python -m emg2pose.train \
  experiment=emg2pose/regression_vemg2pose_egoemg

# EMG2Pose (TDS + MLP)
python -m emg2pose.train \
  experiment=emg2pose/regression_emg2pose_egoemg

# NeuroPose (CNN encoder-decoder)
python -m emg2pose.train \
  experiment=emg2pose/regression_neuropose_egoemg
```

Each also has a `_with_aug` variant with training-time EMG augmentation.

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
  experiment=emgformer/regression_emgformer_small_aggressive \
  checkpoint=/path/to/checkpoint.ckpt \
  data_location=/path/to/emg2pose_v3
```

### Offline Test Analysis

```bash
python -m emg2pose.test_analysis \
  experiment=emgformer/regression_emgformer_middle_aggressive \
  checkpoint=/path/to/checkpoint.ckpt \
  data_location=/path/to/emg2pose_v3
```

For fusion model analysis:
```bash
python -m emg2pose.test_analysis_fusion \
  --config-name vision_resnet_small_emgfusion_center \
  checkpoint=/path/to/checkpoint.ckpt
```

## Reproducing Paper Results

Reproduced results for all experiments reported in the paper are stored in `test_results/`. Each subdirectory contains:
- `results.csv`: Primary metrics on test splits
- `README.md`: Experiment-specific configuration details

The `test_results/all_results_index.csv` provides a one-line index of every experiment with its primary metric.

Key numbers from the paper:

| Task | Method | Primary Metric |
|------|--------|---------------|
| EMG-to-Pose (EMG2Pose) | EMGFormer-Small | 12.34° MAE (user_stage) |
| EMG-to-Pose (EgoEMG) | EMGFormer-Small | 15.0° MAE |
| Vision-to-Pose | ResNet-152 | 5.1° MAE |
| EMG+Vision Fusion | ResNet-18 + EMGFormer-Small | 5.4° MAE |

## Compute Requirements

All experiments were conducted on NVIDIA RTX 4090 GPUs (24 GB each) with bf16 mixed precision:
- EMGFormer-Small: ~2 GPU-hours (200 epochs)
- EMGFormer-Middle: ~4 GPU-hours
- EMGFormer-Large: ~8 GPU-hours
- Vision baselines: ~4-12 GPU-hours each
- Fusion baselines: ~8-12 GPU-hours each

Total compute for all reported experiments: approximately 163 GPU-hours.

## License

The baseline code is distributed under the MIT License. The EgoEmg dataset will be released under CC-BY-NC 4.0 for research use. Third-party assets (MANO model, pretrained vision backbones) remain subject to their original licenses.

## Citation

If you use this benchmark or dataset in your research, please cite:

```bibtex
@inproceedings{egoemg2026,
  title={EgoEmg: A Multimodal Egocentric Dataset with Bilateral EMG and Vision for Hand Pose Estimation},
  author={Anonymous},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS) Datasets and Benchmarks Track},
  year={2026}
}
```
