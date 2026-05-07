#!/bin/bash
# EMG2Pose: EMGFormer supervised training on EMG2Pose dataset
# Table: EMG2Pose benchmark results (per-dataset normalization)
#
# Usage:
#   bash scripts/experiments/run_emg2pose_emgformer.sh
#   bash scripts/experiments/run_emg2pose_emgformer.sh --data /path/to/data
#
# Dry-run (verify config only, no training):
#   bash scripts/experiments/run_emg2pose_emgformer.sh --dry_run

set -euo pipefail
cd "$(dirname "$0")/../.."

DATA_LOCATION="${DATA_LOCATION:-/path/to/emg2pose_v3}"
DRY_RUN=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data) DATA_LOCATION="$2"; shift 2 ;;
    --dry_run) DRY_RUN=true; shift ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

EMGFORMER_EXPS=(
  emgformer/emg2pose_emgformer_small
  emgformer/emg2pose_emgformer_middle
  emgformer/emg2pose_emgformer_large
)

for exp in "${EMGFORMER_EXPS[@]}"; do
  echo "=== Running: experiment=$exp ==="
  if $DRY_RUN; then
    python -m emg2pose.train \
      train=False eval=False \
      experiment="$exp" \
      data_location="$DATA_LOCATION"
  else
    python -m emg2pose.train \
      train=True eval=True \
      experiment="$exp" \
      data_location="$DATA_LOCATION"
  fi
  echo ""
done

echo "All EMG2Pose EMGFormer experiments completed."
