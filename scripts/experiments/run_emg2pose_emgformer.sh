#!/bin/bash
# EMG2Pose: EMGFormer supervised training on EMG2Pose dataset
# Table: EMG2Pose benchmark results (per-dataset normalization)
#
# Usage:
#   bash scripts/experiments/run_emg2pose_emgformer.sh
#   bash scripts/experiments/run_emg2pose_emgformer.sh --data_location /path/to/data
#
# Dry-run (verify config only, no training):
#   bash scripts/experiments/run_emg2pose_emgformer.sh --dry_run

set -euo pipefail
cd "$(dirname "$0")/../.."

DATA_LOCATION="${1:-/path/to/emg2pose_v3}"
DRY_RUN="${2:-}"

EMGFORMER_EXPS=(
  emgformer/regression_emgformer_small_aggressive
  emgformer/regression_emgformer_middle_aggressive
  emgformer/regression_emgformer_large_aggressive
)

for exp in "${EMGFORMER_EXPS[@]}"; do
  echo "=== Running: experiment=$exp ==="
  if [ "$DRY_RUN" = "--dry_run" ]; then
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
