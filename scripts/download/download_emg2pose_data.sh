#!/bin/bash
# Download EMG2Pose dataset (memmap format) for the EgoEmg benchmark.
#
# Usage:
#   bash scripts/download/download_emg2pose_data.sh /path/to/data
#
# This downloads the EMG2Pose memmap dataset to the specified directory.

set -euo pipefail

DATA_DIR="${1:-data/emg2pose_v3}"
mkdir -p "$DATA_DIR"

# ── Google Drive folder ID ─────────────────────────────────────────────────
# Replace with actual ID after uploading to Google Drive.
GDRIVE_FOLDER_ID="<GDRIVE_FOLDER_ID>"

if [[ "$GDRIVE_FOLDER_ID" == "<GDRIVE_FOLDER_ID>" ]]; then
  echo "Error: Google Drive folder ID not set."
  echo "Please update GDRIVE_FOLDER_ID in this script with the actual ID."
  exit 1
fi

echo "Downloading EMG2Pose memmap dataset to $DATA_DIR ..."
gdown --folder "https://drive.google.com/drive/folders/${GDRIVE_FOLDER_ID}" -O "$DATA_DIR"

echo "EMG2Pose dataset downloaded to $DATA_DIR"
