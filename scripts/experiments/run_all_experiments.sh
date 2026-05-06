#!/bin/bash
# Master experiment runner for the EgoEMG benchmark paper.
#
# All experiments are listed below with their exact paper names.
# Run with --dry_run to validate configs without actual training.
#
# Usage:
#   # Dry-run verification (no data needed)
#   bash scripts/experiments/run_all_experiments.sh --dry_run
#
#   # Run a specific experiment
#   bash scripts/experiments/run_all_experiments.sh --exp emg2pose_small
#
#   # Run a group of experiments
#   bash scripts/experiments/run_all_experiments.sh --group emg2pose
#   bash scripts/experiments/run_all_experiments.sh --group egoemg_emgformer
#   bash scripts/experiments/run_all_experiments.sh --group egoemg_baselines
#   bash scripts/experiments/run_all_experiments.sh --group vision
#   bash scripts/experiments/run_all_experiments.sh --group fusion

set -euo pipefail
cd "$(dirname "$0")/../.."

# ── Default settings ────────────────────────────────────────────────────────
DRY_RUN=false
MODE="all"
EXP=""
GROUP=""
DATA_LOCATION="${DATA_LOCATION:-/path/to/data}"
EGOEMG_MEMAP_DIR="${EGOEMG_MEMAP_DIR:-/path/to/EgoEMG_memmap}"

# ── Parse arguments ─────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry_run) DRY_RUN=true; shift ;;
    --exp) EXP="$2"; shift 2 ;;
    --group) GROUP="$2"; shift 2 ;;
    --data) DATA_LOCATION="$2"; shift 2 ;;
    --egoemg) EGOEMG_MEMAP_DIR="$2"; shift 2 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

TRAIN_FLAG="train=True eval=True"
if $DRY_RUN; then
  TRAIN_FLAG="train=False eval=False"
  echo "=== DRY RUN MODE: config validation only ==="
fi

# ── All experiments by group ────────────────────────────────────────────────
declare -A EMG2POSE_EXPS=(
  [emg2pose_small]="emgformer/regression_emgformer_small_aggressive"
  [emg2pose_middle]="emgformer/regression_emgformer_middle_aggressive"
  [emg2pose_large]="emgformer/regression_emgformer_large_aggressive"
)

declare -A EGOEMG_EMGFORMER_EXPS=(
  [egoemg_small_aug]="emgformer/regression_emgformer_small_aggressive_egoemg"
  [egoemg_middle_aug]="emgformer/regression_emgformer_middle_aggressive_egoemg"
  [egoemg_large_aug]="emgformer/regression_emgformer_large_aggressive_egoemg"
  [egoemg_small_wo_aug]="emgformer/regression_emgformer_small_aggressive_egoemg_wo_aug"
  [egoemg_middle_wo_aug]="emgformer/regression_emgformer_middle_aggressive_egoemg_wo_aug"
  [egoemg_large_wo_aug]="emgformer/regression_emgformer_large_aggressive_egoemg_wo_aug"
)

declare -A EGOEMG_BASELINE_EXPS=(
  [vemg2pose_wo_aug]="emg2pose/regression_vemg2pose_egoemg"
  [vemg2pose_with_aug]="emg2pose/regression_vemg2pose_egoemg_with_aug"
  [emg2pose_wo_aug]="emg2pose/regression_emg2pose_egoemg"
  [emg2pose_with_aug]="emg2pose/regression_emg2pose_egoemg_with_aug"
  [neuropose_wo_aug]="emg2pose/regression_neuropose_egoemg"
  [neuropose_with_aug]="emg2pose/regression_neuropose_egoemg_with_aug"
)

declare -A VISION_EXPS=(
  [vision_resnet18]="--config-name vision_resnet18"
  [vision_resnet50]="--config-name vision_resnet50"
  [vision_resnet152]="--config-name vision_resnet152"
  [vision_vit_small]="--config-name vision_vit_small"
  [vision_vit_base]="--config-name vision_vit_base"
  [vision_vit_large]="--config-name vision_vit_large"
)

declare -A FUSION_EXPS=(
  [fusion_resnet_emg]="--config-name vision_resnet_small_emgfusion_center"
  [fusion_vit_emg]="--config-name vision_vit_small_emgfusion_center"
)

# ── Runner helpers ──────────────────────────────────────────────────────────
run_emg_exp() {
  local exp="$1"
  echo "=== Running: experiment=$exp ==="
  python -m emg2pose.train \
    $TRAIN_FLAG \
    experiment="$exp" \
    data_location="$DATA_LOCATION" \
    egoemg_memmap_dir="$EGOEMG_MEMAP_DIR"
  echo ""
}

run_vision_exp() {
  local cfg="$1"
  echo "=== Running: $cfg ==="
  python -m emg2pose.train \
    $TRAIN_FLAG \
    $cfg \
    egoemg_memmap_dir="$EGOEMG_MEMAP_DIR"
  echo ""
}

run_group() {
  local -n exps=$1
  for key in "${!exps[@]}"; do
    echo ""
    echo "=========================================="
    echo "  Experiment: $key"
    echo "=========================================="
    if [[ "${exps[$key]}" == "--config-name"* ]]; then
      run_vision_exp "${exps[$key]}"
    else
      run_emg_exp "${exps[$key]}"
    fi
  done
}

# ── Execute ─────────────────────────────────────────────────────────────────
if [ -n "$EXP" ]; then
  # Single experiment lookup
  for group in EMG2POSE_EXPS EGOEMG_EMGFORMER_EXPS EGOEMG_BASELINE_EXPS VISION_EXPS FUSION_EXPS; do
    declare -n lookup=$group
    if [ -n "${lookup[$EXP]:-}" ]; then
      if [[ "${lookup[$EXP]}" == "--config-name"* ]]; then
        run_vision_exp "${lookup[$EXP]}"
      else
        run_emg_exp "${lookup[$EXP]}"
      fi
      exit 0
    fi
  done
  echo "Unknown experiment: $EXP"
  echo "Available: ${!EMG2POSE_EXPS[*]} ${!EGOEMG_EMGFORMER_EXPS[*]} ${!EGOEMG_BASELINE_EXPS[*]} ${!VISION_EXPS[*]} ${!FUSION_EXPS[*]}"
  exit 1
fi

if [ -n "$GROUP" ]; then
  case "$GROUP" in
    emg2pose)        run_group EMG2POSE_EXPS ;;
    egoemg_emgformer) run_group EGOEMG_EMGFORMER_EXPS ;;
    egoemg_baselines) run_group EGOEMG_BASELINE_EXPS ;;
    vision)           run_group VISION_EXPS ;;
    fusion)           run_group FUSION_EXPS ;;
    *) echo "Unknown group: $GROUP (available: emg2pose, egoemg_emgformer, egoemg_baselines, vision, fusion)"; exit 1 ;;
  esac
else
  # Run all experiments
  echo "Running all experiments..."
  run_group EMG2POSE_EXPS
  run_group EGOEMG_EMGFORMER_EXPS
  run_group EGOEMG_BASELINE_EXPS
  run_group VISION_EXPS
  run_group FUSION_EXPS
  echo "All experiments completed."
fi
