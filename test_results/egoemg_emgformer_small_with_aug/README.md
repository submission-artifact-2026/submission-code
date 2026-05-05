# egoemg_emgformer_small_with_aug

EMGFormer Small (256d, 4 heads, 3 layers) on EgoEMG, aggressive augmentation.

- **Eval date**: 2026-05-03
- **Checkpoint**: `checkpoints/egoemg_emgformer_small_with_aug_mae0.2618_epoch091.ckpt`
- **Config**: `experiment=emgformer/regression_emgformer_small_aggressive_egoemg`
- **With augmentation** (EgoEMG transform bug fixed)

## Pooled left+right per-user (n users per split)

| split | test_mae | per-user ± std | n |
|---|---|---|---|
| gesture | 0.244 | 0.246 ± 0.026 | 36 |
| user | 0.281 | 0.260 ± 0.046 | 6 |
| both | 0.288 | 0.284 ± 0.018 | 5 |

## Per-hand per-group stats (6 splits)

| split | test_mae | per-user ± std (n) | per-gesture ± std (n) |
|---|---|---|---|
| user/left | 0.275 | 0.258 ± 0.034 (6) | 0.272 ± 0.049 (52) |
| user/right | 0.287 | 0.261 ± 0.058 (6) | 0.283 ± 0.065 (52) |
| gesture/left | 0.234 | 0.238 ± 0.029 (36) | 0.201 ± 0.084 (17) |
| gesture/right | 0.254 | 0.255 ± 0.030 (36) | 0.224 ± 0.083 (17) |
| both/left | 0.280 | 0.274 ± 0.023 (5) | 0.269 ± 0.091 (16) |
| both/right | 0.296 | 0.293 ± 0.014 (5) | 0.283 ± 0.084 (16) |

Simple mean test_mae: **0.271**
