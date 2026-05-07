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
#   bash scripts/experiments/run_all_experiments.sh --exp emg2pose_emgformer_small
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
DATA_LOCATION="${DATA_LOCATION:-/path/to/emg2pose_v3}"
EGOEMG_MEMMAP_DIR="${EGOEMG_MEMMAP_DIR:-/path/to/EgoEMG_memmap}"
PER_EPISODE_CROPS_DIR="${PER_EPISODE_CROPS_DIR:-/path/to/EgoEMG_crops}"
VISION_RESNET_CHECKPOINT="${VISION_RESNET_CHECKPOINT:-}"
VISION_VIT_CHECKPOINT="${VISION_VIT_CHECKPOINT:-}"
PRETRAINED_EMG_CHECKPOINT="${PRETRAINED_EMG_CHECKPOINT:-}"

# ── Parse arguments ─────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry_run) DRY_RUN=true; shift ;;
    --exp) EXP="$2"; shift 2 ;;
    --group) GROUP="$2"; shift 2 ;;
    --data) DATA_LOCATION="$2"; shift 2 ;;
    --egoemg) EGOEMG_MEMMAP_DIR="$2"; shift 2 ;;
    --crops) PER_EPISODE_CROPS_DIR="$2"; shift 2 ;;
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
  [emg2pose_emgformer_small]="emgformer/emg2pose_emgformer_small"
  [emg2pose_emgformer_middle]="emgformer/emg2pose_emgformer_middle"
  [emg2pose_emgformer_large]="emgformer/emg2pose_emgformer_large"
)

declare -A EGOEMG_EMGFORMER_EXPS=(
  [egoemg_emgformer_small]="emgformer/egoemg_emgformer_small"
  [egoemg_emgformer_middle]="emgformer/egoemg_emgformer_middle"
  [egoemg_emgformer_large]="emgformer/egoemg_emgformer_large"
)

declare -A EGOEMG_BASELINE_EXPS=(
  [egoemg_vemg2pose]="emg2pose/egoemg_vemg2pose"
  [egoemg_emg2pose]="emg2pose/egoemg_emg2pose"
  [egoemg_neuropose]="emg2pose/egoemg_neuropose"
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
    egoemg_memmap_dir="$EGOEMG_MEMMAP_DIR"
  echo ""
}

run_vision_exp() {
  local cfg="$1"
  echo "=== Running: $cfg ==="
  python -m emg2pose.train \
    $cfg \
    $TRAIN_FLAG \
    egoemg_memmap_dir="$EGOEMG_MEMMAP_DIR" \
    per_episode_crops_dir="$PER_EPISODE_CROPS_DIR"
  echo ""
}

run_fusion_exp() {
  local cfg="$1"
  local extra_args=()
  # Pass checkpoint paths if set via environment
  [[ -n "$VISION_RESNET_CHECKPOINT" ]] && extra_args+=("vision_resnet_checkpoint=$VISION_RESNET_CHECKPOINT")
  [[ -n "$VISION_VIT_CHECKPOINT" ]] && extra_args+=("vision_vit_checkpoint=$VISION_VIT_CHECKPOINT")
  [[ -n "$PRETRAINED_EMG_CHECKPOINT" ]] && extra_args+=("pretrained_emg_checkpoint=$PRETRAINED_EMG_CHECKPOINT")
  echo "=== Running: $cfg ==="
  python -m emg2pose.train \
    $cfg \
    $TRAIN_FLAG \
    egoemg_memmap_dir="$EGOEMG_MEMMAP_DIR" \
    per_episode_crops_dir="$PER_EPISODE_CROPS_DIR" \
    "${extra_args[@]:-}"
  echo ""
}

run_group() {
  local name="$1"
  local -n exps=$2
  for key in "${!exps[@]}"; do
    echo ""
    echo "=========================================="
    echo "  Experiment: $key"
    echo "=========================================="
    if [[ "${exps[$key]}" == "--config-name"* ]]; then
      if [[ "$name" == "fusion" ]]; then
        run_fusion_exp "${exps[$key]}"
      else
        run_vision_exp "${exps[$key]}"
      fi
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
        if [[ "$group" == "FUSION_EXPS" ]]; then
          run_fusion_exp "${lookup[$EXP]}"
        else
          run_vision_exp "${lookup[$EXP]}"
        fi
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
    emg2pose)        run_group emg2pose EMG2POSE_EXPS ;;
    egoemg_emgformer) run_group egoemg_emgformer EGOEMG_EMGFORMER_EXPS ;;
    egoemg_baselines) run_group egoemg_baselines EGOEMG_BASELINE_EXPS ;;
    vision)           run_group vision VISION_EXPS ;;
    fusion)           run_group fusion FUSION_EXPS ;;
    *) echo "Unknown group: $GROUP (available: emg2pose, egoemg_emgformer, egoemg_baselines, vision, fusion)"; exit 1 ;;
  esac
else
  # Run all experiments
  echo "Running all experiments..."
  run_group emg2pose EMG2POSE_EXPS
  run_group egoemg_emgformer EGOEMG_EMGFORMER_EXPS
  run_group egoemg_baselines EGOEMG_BASELINE_EXPS
  run_group vision VISION_EXPS
  run_group fusion FUSION_EXPS
  echo "All experiments completed."
fi
