

import logging

from collections.abc import Mapping
from pathlib import Path
import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities import rank_zero_only

from emg2pose import utils
from emg2pose.metrics import get_default_metrics
from emg2pose.models.modules import BaseModule
from hydra.utils import instantiate

from omegaconf import DictConfig

log = logging.getLogger(__name__)


def _load_state_dict_from_checkpoint(checkpoint_path: str) -> dict[str, torch.Tensor]:
    """Load state_dict from checkpoint file, handling various formats."""
    path = Path(checkpoint_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Pretrained checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location="cpu")
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            return checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            return checkpoint["state_dict"]
        else:
            return checkpoint
    return checkpoint


class EmgPredictionModule(pl.LightningModule):
    def __init__(
        self,
        module_conf: DictConfig,
        optimizer_conf: DictConfig,
        lr_scheduler_conf: DictConfig,
        loss_weights: dict[str, float] | None = None,
        task_type: str = "regression",  # regression | discrete
        ignore_index: int = -100,
        label_smoothing: float = 0.0,
        gumbel_recon: DictConfig | None = None,
        pretrained_checkpoint: str | None = None,
        pretrained_strict: bool = False,
        pretrained_emg_checkpoint: str | None = None,
        freeze_backbone: bool = False,
        ignore_head_tail_dims: int = 0,
        datamodule: DictConfig | None = None,
        stage2_vision_checkpoint: str | None = None,
        component_lr_scales: dict[str, float] | None = None,
    ) -> None:

        super().__init__()
        self.save_hyperparameters()
        self.model: BaseModule = instantiate(module_conf, _convert_="all")
        self.loss_weights = loss_weights or {"mae": 1}
        self.task_type = task_type
        self.ignore_index = ignore_index
        self.label_smoothing = float(label_smoothing)
        self.gumbel_recon = gumbel_recon or {}
        self.use_gumbel_recon = True
        self.gumbel_tau = float(self.gumbel_recon.get("temperature", 1.0))
        self.gumbel_hard = bool(self.gumbel_recon.get("hard", False))
        self.gumbel_weight = float(self.gumbel_recon.get("weight", 1.0))
        self._warned_gumbel_unfrozen = False
        self.ignore_head_tail_dims = int(ignore_head_tail_dims)
        self.component_lr_scales = component_lr_scales or {}

        # Metrics sets
        self.regression_metrics = get_default_metrics()

        if pretrained_checkpoint is not None:
            self._load_pretrained_backbone(
                pretrained_checkpoint, strict=bool(pretrained_strict)
            )
            self._load_pretrained_angle_head(
                pretrained_checkpoint, strict=bool(pretrained_strict)
            )

        if pretrained_emg_checkpoint is not None:
            self._load_pretrained_backbone(
                pretrained_emg_checkpoint, strict=bool(pretrained_strict)
            )

        if stage2_vision_checkpoint is not None:
            self._load_fusion_vision_weights(stage2_vision_checkpoint)

        if freeze_backbone:
            self._freeze_backbone()

        # Apply component-level freezing for scale=0 entries
        for comp, scale in self.component_lr_scales.items():
            if scale == 0.0:
                self._freeze_component(comp)

    def on_fit_start(self) -> None:
        super().on_fit_start()
        self._log_param_breakdown()

    def _load_pretrained_backbone(self, checkpoint_path: str, strict: bool = False) -> None:
        state_dict = _load_state_dict_from_checkpoint(checkpoint_path)

        model_state = self.model.state_dict()
        filtered: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            stripped = key[6:] if key.startswith("model.") else key
            if not stripped.startswith((
                "featurizer.", "decoder.", "backbone.", "avgpool.",
                "vision_backbone.", "vision_proj.", "fusion_proj.", "head_vision.",
            )):
                continue
            if stripped in model_state and model_state[stripped].shape == value.shape:
                filtered[stripped] = value

        missing_keys, unexpected_keys = self.model.load_state_dict(
            filtered, strict=False
        )
        print(f"Missing keys: {missing_keys}")
        print(f"Unexpected keys: {unexpected_keys}")
        log.info(
            "Loaded pretrained backbone from %s (matched %d/%d keys).",
            Path(checkpoint_path).expanduser(),
            len(filtered),
            len(model_state),
        )

        if strict:
            missing_backbone = [
                key
                for key in missing_keys
                if key.startswith((
                    "featurizer.", "decoder.", "backbone.", "avgpool.",
                    "vision_backbone.", "vision_proj.", "fusion_proj.", "head_vision.",
                ))
            ]
            if missing_backbone or unexpected_keys:
                raise RuntimeError(
                    "Error(s) in loading pretrained backbone:\n"
                    f"\tMissing backbone keys: {missing_backbone}\n"
                    f"\tUnexpected keys: {unexpected_keys}\n"
                )

    def _load_pretrained_angle_head(self, checkpoint_path: str, strict: bool = False) -> None:
        state_dict = _load_state_dict_from_checkpoint(checkpoint_path)

        head = getattr(self.model, "head", None) or getattr(self.model, "angle_head", None)
        if head is None:
            log.warning("Model has neither 'head' nor 'angle_head' — skipping angle_head loading.")
            return

        head_state = head.state_dict()
        filtered: dict[str, torch.Tensor] = {}
        matched = 0

        for key, value in state_dict.items():
            # Pretrain checkpoint uses "model.angle_head." prefix
            if key.startswith("model.angle_head."):
                mapped = key[len("model.angle_head.") :]
            elif key.startswith("angle_head."):
                mapped = key[len("angle_head.") :]
            # Regular model checkpoint uses "model.head." prefix
            elif key.startswith("model.head."):
                mapped = key[len("model.head.") :]
            elif key.startswith("head."):
                mapped = key[len("head.") :]
            else:
                continue

            if mapped not in head_state:
                continue

            target = head_state[mapped]
            if target.shape == value.shape:
                filtered[mapped] = value
                matched += 1
                continue

            if (
                value.ndim >= 1
                and target.shape[0] < value.shape[0]
                and target.shape[1:] == value.shape[1:]
            ):
                # Loaded has more dims than model: truncate
                filtered[mapped] = value[: target.shape[0]].clone()
                matched += 1
                continue

            if (
                value.ndim >= 1
                and target.shape[0] > value.shape[0]
                and target.shape[1:] == value.shape[1:]
            ):
                # Loaded has fewer dims than model: pad with zeros
                padded = torch.zeros_like(target)
                padded[: value.shape[0]] = value
                filtered[mapped] = padded
                matched += 1
                continue

        missing_keys, unexpected_keys = head.load_state_dict(
            filtered, strict=False
        )
        log.info(
            "Loaded pretrained angle_head from %s (matched %d/%d keys).",
            Path(checkpoint_path).expanduser(),
            matched,
            len(head_state),
        )
        if matched == 0:
            log.warning(
                "No angle_head weights matched. Check head architecture and "
                "out_channels vs pretrain."
            )

        if strict:
            if missing_keys or unexpected_keys:
                raise RuntimeError(
                    "Error(s) in loading pretrained angle_head:\n"
                    f"\tMissing keys: {missing_keys}\n"
                    f"\tUnexpected keys: {unexpected_keys}\n"
                )

    @rank_zero_only
    def _log_param_breakdown(self) -> None:
        def _count_params(module: torch.nn.Module) -> tuple[int, int]:
            total = sum(p.numel() for p in module.parameters())
            trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
            return total, trainable

        model = self.model
        parts: dict[str, tuple[int, int]] = {}
        if getattr(model, "featurizer", None) is not None:
            parts["featurizer"] = _count_params(model.featurizer)
        if getattr(model, "decoder", None) is not None:
            parts["decoder"] = _count_params(model.decoder)

        if parts:
            formatted = ", ".join(
                f"{name}={total}/{trainable}"
                for name, (total, trainable) in parts.items()
            )
            log.info("Parameter breakdown (total/trainable): %s", formatted)

    def _freeze_backbone(self) -> None:
        for name, param in self.model.named_parameters():
            if name.startswith(("featurizer.", "decoder.")):
                param.requires_grad = False
        log.info("Backbone frozen (featurizer + decoder).")

    _COMPONENT_PREFIX_MAP: dict[str, list[str]] = {
        "featurizer": ["featurizer."],
        "decoder": ["decoder."],
        "vision_proj": ["vision_proj."],
        "fusion_proj": ["fusion_proj."],
        "head": ["head."],
        "head_vision": ["head_vision."],
        "vision_backbone": ["vision_backbone."],
        "backbone": ["backbone."],
    }

    def _params_by_component(self) -> dict[str, list[torch.nn.Parameter]]:
        """Group model parameters by component prefix."""
        groups: dict[str, list[torch.nn.Parameter]] = {c: [] for c in self._COMPONENT_PREFIX_MAP}
        unassigned: list[torch.nn.Parameter] = []
        for name, param in self.model.named_parameters():
            matched = False
            for comp, prefixes in self._COMPONENT_PREFIX_MAP.items():
                if any(name.startswith(p) for p in prefixes):
                    groups[comp].append(param)
                    matched = True
                    break
            if not matched:
                unassigned.append(param)
        if unassigned:
            groups["_unassigned"] = unassigned
        return groups

    def _freeze_component(self, component: str) -> None:
        prefixes = self._COMPONENT_PREFIX_MAP.get(component)
        if prefixes is None:
            log.warning("Unknown component '%s' for freezing, skipping.", component)
            return
        for name, param in self.model.named_parameters():
            if any(name.startswith(p) for p in prefixes):
                param.requires_grad = False
        log.info("Component '%s' frozen.", component)

    def _load_fusion_vision_weights(self, checkpoint_path: str) -> None:
        """Load vision_proj, fusion_proj, and head weights from a vision_only checkpoint."""
        state_dict = _load_state_dict_from_checkpoint(checkpoint_path)

        vision_prefixes = ("vision_proj.", "head_vision.", "vision_backbone.")
        model_state = self.model.state_dict()
        filtered: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            stripped = key[6:] if key.startswith("model.") else key
            if not any(stripped.startswith(p) for p in vision_prefixes):
                continue
            if stripped in model_state and model_state[stripped].shape == value.shape:
                filtered[stripped] = value

        if filtered:
            self.model.load_state_dict(filtered, strict=False)
            log.info(
                "Loaded %d vision/fusion/head keys from %s",
                len(filtered),
                Path(checkpoint_path).expanduser(),
            )
        else:
            log.warning("No matching vision/fusion/head keys found in %s", checkpoint_path)

    def forward(
        self, batch: Mapping[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        out = self.model.forward(batch)
        if self.task_type == "discrete":
            return self._prepare_discrete(out, batch)

        # Handle tuple output from BaseModule.forward() which returns (preds, targets, mask)
        if isinstance(out, tuple):
            preds, targets, mask = out
        # Handle dict output from EmgformerPretrain
        elif isinstance(out, dict):
            preds = out.get("angles", out.get("recon"))
            if preds is None:
                raise ValueError(
                    f"Model output dict must contain 'angles' or 'recon' key. "
                    f"Got keys: {list(out.keys())}."
                )
            # For dict output, derive targets and mask from batch
            joint_angles = batch.get("joint_angles", batch.get("angle_target"))
            mask = batch.get("label_valid_mask", batch.get("angle_mask"))
            if joint_angles is None or mask is None:
                raise KeyError(
                    "Batch must contain either (joint_angles, label_valid_mask) "
                    f"or (angle_target, angle_mask). Got keys: {list(batch.keys())}"
                )
            start = self.model.left_context
            stop = None if self.model.right_context == 0 else -self.model.right_context
            targets = joint_angles[..., slice(start, stop)]
            mask = mask[..., slice(start, stop)]
            # Align predictions and mask up to targets' time dimension
            n_time = targets.shape[-1]
            preds = self.model.align_predictions(preds, n_time)
            if mask.ndim == 2:
                mask = self.model.align_mask(mask, n_time)
            elif mask.ndim == 3:
                mask = self.model.align_mask(
                    mask.mean(dim=1), n_time
                )  # (B, C, T) -> (B, T) -> align -> (B, T)
        else:
            # Legacy tensor-only output (e.g., Emg2PoseFormer, VEMG2PoseWithInitialState)
            preds = out
            joint_angles = batch.get("joint_angles", batch.get("angle_target"))
            mask = batch.get("label_valid_mask", batch.get("angle_mask"))
            if joint_angles is None or mask is None:
                raise KeyError(
                    "Batch must contain either (joint_angles, label_valid_mask) "
                    f"or (angle_target, angle_mask). Got keys: {list(batch.keys())}"
                )
            start = self.model.left_context
            stop = None if self.model.right_context == 0 else -self.model.right_context
            targets = joint_angles[..., slice(start, stop)]
            mask = mask[..., slice(start, stop)]
            # Align predictions and mask up to targets' time dimension
            n_time = targets.shape[-1]
            preds = self.model.align_predictions(preds, n_time)
            if mask.ndim == 2:
                mask = self.model.align_mask(mask, n_time)
            elif mask.ndim == 3:
                mask = self.model.align_mask(
                    mask.mean(dim=1), n_time
                )  # (B, C, T) -> (B, T) -> align -> (B, T)

        if self.ignore_head_tail_dims > 0:
            if preds.ndim == 2:
                preds = preds[..., None]
            if self.ignore_head_tail_dims >= preds.shape[1]:
                raise ValueError(
                    "ignore_head_tail_dims must be smaller than prediction channels."
                )
            preds = preds[:, : -self.ignore_head_tail_dims, :]

        # Handle prediction-target channel mismatch (e.g., pretrain model outputs
        # 22 channels but dataset only provides 20 joint angles)
        if preds.shape[1] != targets.shape[1]:
            n_ch = min(preds.shape[1], targets.shape[1])
            preds = preds[:, :n_ch, :]
            targets = targets[:, :n_ch, :]

        # For 3D masks (angle_mask: B,C,T) the time is already aligned with targets;
        # for 2D masks (label_valid_mask: B,T) we need temporal alignment
        if mask.ndim == 3:
            # Already time-aligned; just trim channels if needed
            if mask.shape[1] != targets.shape[1]:
                mask = mask[:, :targets.shape[1], :]
        elif mask.ndim == 2:
            n_time = targets.shape[-1]
            mask = self.model.align_mask(mask, n_time)

        return preds, targets, mask

    def _prepare_discrete(
        self, preds: torch.Tensor, batch: Mapping[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        joint_angles = batch["joint_angles"]
        start = self.model.left_context
        stop = None if self.model.right_context == 0 else -self.model.right_context
        targets_full = joint_angles[..., slice(start, stop)]
        code_indices = self._quantize_angles(targets_full)  # (B, L, T_code)
        logits = preds.permute(0, 3, 2, 1).contiguous()  # (B, L, T_pred, num_codes)
        return logits, code_indices

    def _quantize_angles(self, joint_angles: torch.Tensor) -> torch.Tensor:
        vqvae = self.model.head.vqvae_module
        b, j, t = joint_angles.shape
        with torch.no_grad():
            if hasattr(vqvae.model, "quantize_angles"):
                return vqvae.model.quantize_angles(joint_angles)
            flat = joint_angles.transpose(1, 2).reshape(-1, j)
            repr_in = vqvae._encode_representation(flat)
            z_e = vqvae.model.encoder(repr_in)
            _, indices, _, _, _ = vqvae.model.quantizer(z_e)
        if indices.ndim == 1:
            indices = indices.unsqueeze(0)
        if indices.shape[0] == flat.shape[0]:
            indices = indices.transpose(0, 1)
        return indices.view(indices.shape[0], b, t).permute(1, 0, 2)

    def _step(
        self, batch: Mapping[str, torch.Tensor], stage: str = "train"
    ) -> torch.Tensor:

        # Generate predictions
        if getattr(self.hparams, "datamodule", None) and self.hparams.datamodule.get("norm_mode") == "batch":
            emg = batch["emg"]
            mean = emg.mean()
            std = emg.std()
            batch["emg"] = (emg - mean) / (std + 1e-6)
        preds, targets, mask = self.forward(batch)
        batch_size = batch["emg"].shape[0]
        n_time = targets.shape[-1]
        if self.task_type == "discrete":
            code_sub = targets
            cls_loss = self._discrete_loss(preds, code_sub, None, stage)
            cls_weight = self.loss_weights.get("cls", 1.0)
            loss = cls_loss * cls_weight

            if self.use_gumbel_recon:
                gumbel_loss = self._gumbel_recon_loss(preds, code_sub, None, stage)
                loss = loss + gumbel_loss * self.gumbel_weight

            decoded_angles = self.model.head.decode_from_logits(
                preds.permute(0, 3, 2, 1), target_t=n_time
            )  # (B, J, T_pred)
            if decoded_angles.ndim == 2:
                decoded_angles = decoded_angles[..., None]

            # Support both legacy and PretrainWrapperDataset field names
            mask_full = batch.get("label_valid_mask", batch.get("angle_mask"))
            if mask_full is None:
                mask_full = torch.ones_like(targets[:, 0, :])
            mask_full = mask_full[..., slice(self.model.left_context, None if self.model.right_context == 0 else -self.model.right_context)]
            mask_aligned = self.model.align_mask(mask_full, n_time)
            # align targets_full to match decoded_aligned length
            valid_mask = mask_aligned.bool()
            metrics = {}

            for metric in self.regression_metrics:
                metrics.update(metric(decoded_angles, targets, valid_mask, stage))
            self.log_dict(metrics, sync_dist=True, batch_size=batch_size)
            vqvae = self.model.head.vqvae_module
            if hasattr(vqvae.model, "quantize_angles"):
                recon_angles, _, _, _, _ = vqvae(targets)
            else:
                recon_angles, _, _, _, _ = vqvae(targets.reshape(-1, 20))
                recon_angles = recon_angles.reshape(targets.shape)
            recon_diff = torch.abs(recon_angles - targets)
            denom = valid_mask.sum() * recon_diff.shape[1]
            recon_mae = (recon_diff * valid_mask[:, None, :]).sum() / denom
            recon_mae_deg = recon_mae * (180.0 / torch.pi)
            self.log(
                f"{stage}_recon_mae", recon_mae, sync_dist=True, batch_size=batch_size
            )
            self.log(
                f"{stage}_recon_mae_deg",
                recon_mae_deg,
                sync_dist=True,
                batch_size=batch_size,
            )
            self.log(f"{stage}_loss", loss, sync_dist=True, batch_size=batch_size)
            return loss

        # regression path
        valid_mask = mask.bool()
        metrics = {}
        for metric in self.regression_metrics:
            metrics.update(metric(preds, targets, valid_mask, stage))
        self.log_dict(metrics, sync_dist=True, batch_size=batch_size)

        # ── Center-frame MAE (for vision / fusion models) ──────────
        # Only needed when T > 1 (broadcast case).  When the dataset already
        # returns center-frame-only targets (T == 1) the regular metrics above
        # cover the single time step.
        if ("vision_features" in batch or "vision_img" in batch) and mask.shape[-1] > 1:
            T = mask.shape[-1]
            center = T // 2
            center_mask = torch.zeros_like(mask, dtype=torch.float32)
            center_mask[..., center] = mask[..., center].float()
            center_valid = center_mask.bool()
            if center_valid.any():
                center_mae = {}
                for metric in self.regression_metrics:
                    center_mae.update(metric(preds, targets, center_valid, f"{stage}_center"))
                self.log_dict(center_mae, sync_dist=True, batch_size=batch_size)

        loss = 0.0
        for loss_name, weight in self.loss_weights.items():
            loss += metrics.get(f"{stage}_{loss_name}", 0.0) * weight

        # ── Delta L2: always report magnitude, optionally regularize ──
        delta_reg_weight = float(self.loss_weights.get("delta_reg", 0.0))
        delta = getattr(self.model, "_last_delta", None)
        if delta is not None:
            delta_l2 = (delta ** 2).mean()
            self.log(f"{stage}_delta_l2", delta_l2, sync_dist=True, batch_size=batch_size)
            if delta_reg_weight > 0 and stage == "train":
                loss = loss + delta_reg_weight * delta_l2

        self.log(f"{stage}_loss", loss, sync_dist=True, batch_size=batch_size)
        return loss
        
    def training_step(self, batch, batch_idx) -> torch.Tensor:
        result = self._step(batch, stage="train")
        # Log learning rate
        sch = self.lr_schedulers()
        if sch is not None:
            self.log("lr", sch.get_last_lr()[0], on_step=False, on_epoch=True, prog_bar=True)
        return result

    def validation_step(self, batch, batch_idx) -> torch.Tensor:
        return self._step(batch, stage="val")

    def test_step(
        self, batch, batch_idx, dataloader_idx: int | None = None
    ) -> torch.Tensor:
        return self._step(batch, stage="test")

    def configure_optimizers(self):
        vqvae = getattr(getattr(self.model, "head", None), "vqvae_module", None)
        excluded = {id(p) for p in vqvae.parameters()} if vqvae is not None else set()

        scales = self.component_lr_scales
        if not scales:
            params = [p for p in self.parameters() if id(p) not in excluded]
            return utils.instantiate_optimizer_and_scheduler(
                params,
                optimizer_config=self.hparams.optimizer_conf,
                lr_scheduler_config=self.hparams.lr_scheduler_conf,
            )

        # Per-component param groups with scaled learning rates
        base_lr = float(self.hparams.optimizer_conf.lr)
        comp_params = self._params_by_component()
        param_groups = []
        for comp, params in comp_params.items():
            params = [p for p in params if p.requires_grad and id(p) not in excluded]
            if not params:
                continue
            scale = scales.get(comp, 1.0)
            param_groups.append({
                "params": params,
                "lr": base_lr * scale,
                "name": comp,
            })

        if not param_groups:
            raise RuntimeError("All parameters frozen — nothing to optimize.")

        names_and_scales = ", ".join(
            f"{g['name']}={scales.get(g['name'], 1.0):.0e}"
            for g in param_groups
        )
        log.info("Per-component LR scales: %s (base_lr=%.0e)", names_and_scales, base_lr)

        import copy
        lr_scheduler_conf = self.hparams.lr_scheduler_conf
        return utils.instantiate_optimizer_and_scheduler(
            param_groups,
            optimizer_config=self.hparams.optimizer_conf,
            lr_scheduler_config=lr_scheduler_conf,
        )

    def _discrete_loss(
        self,
        logits: torch.Tensor,
        code_indices: torch.Tensor,
        mask: torch.Tensor,
        stage: str,
    ) -> torch.Tensor:
        """
        logits: (B, L, T_pred, num_codes)
        code_indices: (B, T_pred, L)
        mask: (B, T_pred)
        """
        B, L, T_pred, num_codes = logits.shape
        code_indices = code_indices.long()

        logits_flat = logits.permute(0, 2, 1, 3).reshape(-1, num_codes)  # (B*T_pred*L, K)
        targets_flat = code_indices.reshape(-1)  # (B*T_pred*L,)
        if mask is not None:
            mask_flat = mask.unsqueeze(-1).expand(-1, -1, L).reshape(-1)  # (B*T_pred*L,)
            valid = mask_flat.bool()
            logits_flat = logits_flat[valid]
            targets_flat = targets_flat[valid]
        ce_loss = torch.nn.functional.cross_entropy(
            logits_flat,
            targets_flat,
            ignore_index=self.ignore_index,
            label_smoothing=self.label_smoothing,
        )

        # Accuracy
        with torch.no_grad():
            pred_idx = logits.argmax(dim=-1)  # (B, L, T_pred)

            correct = (pred_idx == code_indices).to(torch.float32)
            # if mask is not None:
            #     correct = correct * mask.unsqueeze(-1)
            acc = correct.sum() / (mask.unsqueeze(-1).sum() + 1e-6) if mask is not None else correct.mean()

        batch_size = logits.shape[0]
        self.log(f"{stage}_cls_ce", ce_loss, sync_dist=True, batch_size=batch_size)
        self.log(f"{stage}_cls_acc", acc, sync_dist=True, batch_size=batch_size)

        return ce_loss

    def _gumbel_recon_loss(
        self,
        logits: torch.Tensor,
        code_indices: torch.Tensor,
        mask: torch.Tensor,
        stage: str,
    ) -> torch.Tensor:
        self._warn_if_gumbel_unfrozen()
        logits_for_head = logits.permute(0, 3, 2, 1).contiguous()  # (B, K, T, L)
        # import ipdb;ipdb.set_trace()
        pred_angles = self.model.head.decode_from_logits_gumbel(
            logits_for_head, tau=self.gumbel_tau, hard=self.gumbel_hard
        )
        target_angles = self.model.head.decode_from_indices(code_indices.permute(0, 2, 1))
        diff = torch.abs(pred_angles - target_angles)
        if mask is not None:
            valid = mask.bool()
            denom = valid.sum().clamp(min=1) * diff.shape[1]
            loss = (diff * valid[:, None, :]).sum() / denom
        else:
            loss = diff.mean()
        batch_size = logits.shape[0]
        self.log(
            f"{stage}_gumbel_recon_mae",
            loss,
            sync_dist=True,
            batch_size=batch_size,
        )
        self.log(
            f"{stage}_gumbel_recon_mae_deg",
            loss * (180.0 / torch.pi),
            sync_dist=True,
            batch_size=logits.shape[0],
        )
        return loss

    def _warn_if_gumbel_unfrozen(self) -> None:
        if self._warned_gumbel_unfrozen or not self.use_gumbel_recon:
            return
        head = getattr(self.model, "head", None)
        if head is None or not hasattr(head, "vqvae_module"):
            self._warned_gumbel_unfrozen = True
            return
        model = head.vqvae_module.model
        quantizer = model.quantizer
        codebook_trainable = any(p.requires_grad for p in quantizer.parameters())
        decoder_modules = []
        if hasattr(model, "decoder"):
            decoder_modules.append(model.decoder)
        if hasattr(model, "upsample"):
            decoder_modules.append(model.upsample)
        if decoder_modules:
            decoder_trainable = any(
                p.requires_grad for module in decoder_modules for p in module.parameters()
            )
        else:
            decoder_trainable = False
        if codebook_trainable or decoder_trainable:
            log.warning(
                "Gumbel recon enabled but VQ codebook/decoder are trainable. "
                "Consider freeze_codebook=True and freeze_decoder=True."
            )
        self._warned_gumbel_unfrozen = True
