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
#   │ EMG2Pose EMGFormer-Small                         │ emg2pose_emgformer_small    │
#   │ EgoEmg EMGFormer-Small                           │ egoemg_emgformer_small       │
#   │ Vision ResNet-18                                 │ vision_resnet18              │
#   │ Vision ViT-Small                                  │ vision_vit_small             │
#   │ Fusion ResNet-18 + EMGFormer-Small               │ fusion_resnet_emgfusion      │
#   │ Fusion ViT-Small + EMGFormer-Small               │ fusion_vit_emgfusion         │
#   └──────────────────────────────────────────────────┴──────────────────────────────┘

set -euo pipefail
cd "$(dirname "$0")/../.."

GDRIVE_FOLDER_ID="1_JcHDs9uBIbFxbH0f41Sk95pCqXCcTFG"

mkdir -p checkpoints

echo "Downloading pretrained checkpoints from Google Drive ..."
gdown --folder "https://drive.google.com/drive/folders/${GDRIVE_FOLDER_ID}" -O checkpoints

echo "Done."
