# Fusion Experiments

EMG + Vision fusion results on EgoEMG.

## F-RN18+S (version_12, center_supervised, all trainable)

Model: ResNet18 + EMGFormer Small
Checkpoint: `logs/fusion/resnet_small_emgfusion_center/version_12/checkpoints/resnet-small-centerfusion-epoch=002-val_mae=0.0978.ckpt`
Params: 15.5M

| Split | Hand | MAE (rad) | MAE (°) | Fingertip (mm) | Landmark (mm) |
|-------|------|-----------|---------|-----------------|---------------|
| user | left | 0.1202 | 6.89 | 15.10 | 9.45 |
| user | right | 0.1110 | 6.36 | 14.30 | 8.96 |
| gesture | left | 0.0837 | 4.80 | 10.90 | 6.81 |
| gesture | right | 0.0813 | 4.66 | 10.73 | 6.66 |
| both | left | 0.1212 | 6.95 | 14.69 | 9.29 |
| both | right | 0.1107 | 6.35 | 13.65 | 8.70 |
| **Mean** | | **0.0978** | **5.60** | **12.56** | **8.31** |

Per-split (sample-weighted, °): gesture=4.7±1.3, user=6.6±1.4, both=6.6±0.8

## F-ViTS+S (version_7, center_supervised, all trainable)

Model: ViT-Small (DINOv2) + EMGFormer Small
Checkpoint: `logs/fusion/vit_small_emgfusion_center/version_7/checkpoints/vit-small-centerfusion-epoch=088-val_mae=0.0968.ckpt`
Params: 25.9M

| Split | Hand | MAE (rad) | MAE (°) | Fingertip (mm) | Landmark (mm) |
|-------|------|-----------|---------|-----------------|---------------|
| user | left | 0.1232 | 7.06 | 14.97 | 9.42 |
| user | right | 0.1147 | 6.57 | 14.25 | 8.99 |
| gesture | left | 0.0786 | 4.50 | 10.45 | 6.53 |
| gesture | right | 0.0769 | 4.41 | 10.30 | 6.41 |
| both | left | 0.1236 | 7.08 | 14.86 | 9.40 |
| both | right | 0.1120 | 6.42 | 13.21 | 8.46 |
| **Mean** | | **0.0966** | **5.54** | **12.28** | **8.20** |

Per-split (sample-weighted, °): gesture=4.5±1.3, user=6.8±1.2, both=6.8±0.6

## Old Results (version_9, superseded)

Model: ResNet18 + EMGFormer Small
Checkpoint: `test_results/fusion/checkpoints/best.ckpt` (epoch=198, val_mae=0.0984)
CSV: `version_9_test.csv` (no per-user std)

## Delta Contribution

See `architecture.md` for full pipeline documentation and delta contribution analysis.
