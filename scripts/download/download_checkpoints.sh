#!/bin/bash
# Download pretrained checkpoints for EgoEmg benchmark experiments.
#
# Checkpoints are hosted on Google Drive. This script uses gdown to download
# them into the checkpoints/ directory.
#
# Usage:
#   bash scripts/download/download_checkpoints.sh
#
# NOTE: Coming soon.
#
# Six checkpoints are provided:
#   ┌──────────────────────────────────────────────────┬──────────────────────────────┐
#   │ Experiment                                       │ Checkpoint                   │
#   ├──────────────────────────────────────────────────┼──────────────────────────────┤
#   │ EMG2Pose EMGFormer-Small                         │ *_emgformer_small            │
#   │ EgoEmg EMGFormer-Small                           │ *_emgformer_small            │
#   │ Vision ResNet-18                                 │ *_vision_resnet18            │
#   │ Vision ViT-Small                                  │ *_vision_vit_small           │
#   │ Fusion ResNet-18 + EMGFormer-Small               │ *_fusion_resnet_emgfusion    │
#   │ Fusion ViT-Small + EMGFormer-Small               │ *_fusion_vit_emgfusion       │
#   └──────────────────────────────────────────────────┴──────────────────────────────┘

set -euo pipefail
cd "$(dirname "$0")/../.."

# ── Google Drive folder containing all checkpoints ──────────────────────────
# Same folder as the EgoEMG full dataset.
GDRIVE_FOLDER_ID="12C6Q1CD1uihJhx4s0Rm2s7Um76Kh8rG1"

declare -A CKPT_NAMES=(
  [emg2pose_emgformer_small]="emg2pose_emgformer_small.ckpt"
  [egoemg_emgformer_small]="egoemg_emgformer_small.ckpt"
  [vision_resnet18]="vision_resnet18.ckpt"
  [vision_vit_small]="vision_vit_small.ckpt"
  [fusion_resnet_small_emgfusion_center]="fusion_resnet_small_emgfusion_center.ckpt"
  [fusion_vit_small_emgfusion_center]="fusion_vit_small_emgfusion_center.ckpt"
)

mkdir -p checkpoints

echo "NOTE: Pretrained checkpoints are coming soon."

for name in "${!CKPT_NAMES[@]}"; do
  filename="${CKPT_NAMES[$name]}"
  echo "Downloading $name ..."
  gdown "https://drive.google.com/drive/folders/${GDRIVE_FOLDER_ID}" -O "checkpoints/${filename}" --remaining-ok 2>/dev/null || {
    echo "Checkpoint $name is not yet available (coming soon)."
    continue
  }
done

echo "Done."
