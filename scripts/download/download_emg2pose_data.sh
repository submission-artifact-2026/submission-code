#!/bin/bash
# Download EMG2Pose dataset (memmap format) for the EgoEmg benchmark.
#
# Usage:
#   bash scripts/download/download_emg2pose_data.sh /path/to/data
#
# This downloads the EMG2Pose memmap dataset to the specified directory.
#
# NOTE: Coming soon.

set -euo pipefail

DATA_DIR="${1:-data/emg2pose_v3}"
mkdir -p "$DATA_DIR"

# ── Google Drive folder ID ─────────────────────────────────────────────────
# Same folder as the EgoEMG full dataset.
GDRIVE_FOLDER_ID="12C6Q1CD1uihJhx4s0Rm2s7Um76Kh8rG1"

echo "Downloading EMG2Pose memmap dataset to $DATA_DIR ..."
echo "NOTE: EMG2Pose data is coming soon."
gdown --folder "https://drive.google.com/drive/folders/${GDRIVE_FOLDER_ID}" -O "$DATA_DIR" 2>/dev/null || {
  echo "EMG2Pose data is not yet available (coming soon)."
  exit 1
}

echo "EMG2Pose dataset downloaded to $DATA_DIR"
