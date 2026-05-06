#!/bin/bash
# Download pretrained checkpoints for EgoEmg benchmark experiments.
#
# Checkpoints are hosted on Google Drive. This script uses gdown to download
# them into the checkpoints/ directory.
#
# Usage:
#   bash scripts/download/download_checkpoints.sh
#
# Six checkpoints are provided:
#   ┌──────────────────────────────────────────────────┬──────────────────────────────┐
#   │ Experiment                                       │ Checkpoint                   │
#   ├──────────────────────────────────────────────────┼──────────────────────────────┤
#   │ EMG2Pose EMGFormer-Small                         │ *_emgformer_small          │
#   │ EgoEmg EMGFormer-Small                           │ *_emgformer_small            │
#   │ Vision ResNet-18                                 │ *_vision_resnet18            │
#   │ Vision ViT-Small                                  │ *_vision_vit_small           │
#   │ Fusion ResNet-18 + EMGFormer-Small               │ *_fusion_resnet_emgfusion    │
#   │ Fusion ViT-Small + EMGFormer-Small               │ *_fusion_vit_emgfusion       │
#   └──────────────────────────────────────────────────┴──────────────────────────────┘

set -euo pipefail
cd "$(dirname "$0")/../.."

# ── Google Drive file IDs ──────────────────────────────────────────────────
# These IDs correspond to the six checkpoints listed above.
# Replace with actual IDs after uploading to Google Drive.
declare -A CKPT_IDS=(
  [emg2pose_emgformer_small]="<GDRIVE_FILE_ID>"
  [egoemg_emgformer_small]="<GDRIVE_FILE_ID>"
  [vision_resnet18]="<GDRIVE_FILE_ID>"
  [vision_vit_small]="<GDRIVE_FILE_ID>"
  [fusion_resnet_small_emgfusion_center]="<GDRIVE_FILE_ID>"
  [fusion_vit_small_emgfusion_center]="<GDRIVE_FILE_ID>"
)

mkdir -p checkpoints

for name in "${!CKPT_IDS[@]}"; do
  file_id="${CKPT_IDS[$name]}"
  if [[ "$file_id" == "<GDRIVE_FILE_ID>" ]]; then
    echo "Skipping $name (no Google Drive ID set)"
    continue
  fi
  echo "Downloading $name ..."
  gdown "https://drive.google.com/uc?id=$file_id" -O "checkpoints/${name}.ckpt"
done

echo "All checkpoints downloaded."
