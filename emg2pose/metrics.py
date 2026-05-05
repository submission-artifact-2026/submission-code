

import torch
import torch.nn.functional as F
from emg2pose.constants import (
    EMG_SAMPLE_RATE,
    FINGERS,
    JOINTS,
    LANDMARKS,
    NO_MOVEMENT_LANDMARKS,
    NUM_JOINTS,
    PD_GROUPS,
)

from emg2pose.kinematics import (
    forward_kinematics,
    load_default_hand_model,
    TorchHandModel,
)


class Metric:
    """Compute a dictionary of metrics from predicted and target joint angles."""

    weight: float = 0

    def __call__(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: bool,
        stage: str,
    ) -> dict[str, torch.Tensor]:
        raise NotImplementedError


class AnglularDerivatives(Metric):
    """Mean absolute value of angular velocity, acceleration, and jerk."""

    def __call__(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
        stage: str,
    ) -> dict[str, torch.Tensor]:

        vel = torch.diff(pred, dim=-1)
        acc = torch.diff(vel, dim=-1)
        jerk = torch.diff(acc, dim=-1)

        # Use prediction's channel dimension instead of hardcoded NUM_JOINTS
        num_joints = pred.shape[1]
        mask = mask.unsqueeze(1).expand(-1, num_joints, -1)  # BT -> BCT
        mask_vel = self.adjust_mask(mask)
        mask_acc = self.adjust_mask(mask_vel)
        mask_jerk = self.adjust_mask(mask_acc)

        # vel, acc, and jerk are in (radians / sample), so we multiply by the emg sample
        # rate to get (radians / second). Also take absolute value.
        def safe_mean(tensor, mask_tensor):
            if mask_tensor.sum() == 0:
                # Keep graph connectivity to avoid DDP unused-parameter errors when
                # an entire batch is masked out.
                return tensor.sum() * 0.0
            return tensor[mask_tensor].abs().mean() * EMG_SAMPLE_RATE

        return {
            f"{stage}_vel": safe_mean(vel, mask_vel),
            f"{stage}_acc": safe_mean(acc, mask_acc),
            f"{stage}_jerk": safe_mean(jerk, mask_jerk),
        }

    def adjust_mask(self, mask: torch.Tensor) -> torch.Tensor:
        """Adjust mask to eliminate boundaries between IK failures.

        This operation reduces the time length by 1 (to match `torch.diff`).
        For very short sequences (T < 2), return an empty mask to avoid pooling
        errors and to indicate that derivatives are undefined.
        """
        if mask.shape[-1] < 2:
            return mask[..., :0]
        return ~F.max_pool1d((~mask).float(), kernel_size=2, stride=1).to(bool)


class AngleMAE(Metric):
    """Angular mean absolute error."""

    def __call__(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
        stage: str,
    ) -> dict[str, torch.Tensor]:
        # Use prediction's channel dimension instead of hardcoded NUM_JOINTS
        num_joints = pred.shape[1]
        mask = mask.unsqueeze(1).expand(-1, num_joints, -1)
        if mask.sum() == 0:
            return {f"{stage}_mae": pred.sum() * 0.0}
        return {f"{stage}_mae": torch.nn.L1Loss()(pred[mask], target[mask])}


class PerFingerAngleMAE(Metric):
    """Angular mean absolute error for each finger."""

    def __call__(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
        stage: str,
    ) -> dict[str, torch.Tensor]:

        return {
            f"{stage}_mae_{finger}": self.get_error_for_finger(
                pred, target, mask, finger
            )
            for finger in FINGERS
        }

    @staticmethod
    def get_error_for_finger(pred, target, mask, finger: str):
        idxs = [j.index for j in JOINTS if finger in j.groups]
        mask = mask.unsqueeze(1).expand(-1, len(idxs), -1)
        if mask.sum() == 0:
            return pred.sum() * 0.0
        return torch.nn.L1Loss()(pred[:, idxs][mask], target[:, idxs][mask])


class PDAngleMAE(Metric):
    """Angular mean absolute error grouped by proximal-distal joints."""

    def __call__(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
        stage: str,
    ) -> dict[str, torch.Tensor]:

        return {
            f"{stage}_mae_{group}": self.get_error_for_group(pred, target, mask, group)
            for group in PD_GROUPS
        }

    @staticmethod
    def get_error_for_group(pred, target, mask, group: str):
        idxs = [j.index for j in JOINTS if group in j.groups]
        mask = mask.unsqueeze(1).expand(-1, len(idxs), -1)
        if mask.sum() == 0:
            return pred.sum() * 0.0
        return torch.nn.L1Loss()(pred[:, idxs][mask], target[:, idxs][mask])


class LandmarkDistances(Metric):
    """Mean Euclidian error for fingertips positions."""

    def __init__(self, downsampling: int = 40):
        self.downsampling = downsampling
        self.hand_model = TorchHandModel(load_default_hand_model())

    def __call__(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
        stage: str,
    ) -> dict[str, torch.Tensor]:

        if self.hand_model.device != pred.device:
            self.hand_model.to(pred.device)

        # If everything is masked out, return graph-connected zeros and skip FK.
        if mask.sum() == 0:
            z = pred.sum() * 0.0
            return {
                f"{stage}_fingertip_distance": z,
                f"{stage}_landmark_distance": z,
            }

        # Convert angles to 3D positions.
        # Pick frames that have at least one valid sample (mask.any) and
        # downsample those if needed – this keeps vision_only-style sparse
        # supervision (e.g. single center frame) from being entirely missed
        # by a uniform stride that does not align with the active frame.
        valid_frames = mask.any(dim=0).nonzero(as_tuple=True)[0]
        if len(valid_frames) > 100:
            step = max(1, len(valid_frames) // 100)
            valid_frames = valid_frames[::step]
        # FK uses ops such as torch.eye that are not implemented for bfloat16 on CPU.
        # Cast to float32 here so offline evaluation works consistently across devices.
        pred_pos = forward_kinematics(pred[:, :, valid_frames].float(), self.hand_model)
        target_pos = forward_kinematics(target[:, :, valid_frames].float(), self.hand_model)
        mask_sliced = mask[:, valid_frames]

        # Landmark distances
        lm_idxs = [lm.index for lm in LANDMARKS if lm.name not in NO_MOVEMENT_LANDMARKS]
        landmark_distance = self.get_mean_distance(
            pred_pos, target_pos, mask_sliced, lm_idxs
        )

        # Fingertip distances
        joint_idxs = [lm.index for lm in LANDMARKS if "fingertip" in lm.groups]
        fingertip_distance = self.get_mean_distance(
            pred_pos, target_pos, mask_sliced, joint_idxs
        )

        return {
            f"{stage}_fingertip_distance": fingertip_distance,
            f"{stage}_landmark_distance": landmark_distance,
        }

    def get_mean_distance(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
        idxs: list,
    ):
        """
        Mean distance for given landmark idxs. Inputs are (batch, time, landmark, xyz).
        """
        mask = mask[..., None].expand(-1, -1, len(idxs))
        masked = torch.linalg.norm(pred[:, :, idxs] - target[:, :, idxs], dim=-1)[mask]
        if masked.numel() == 0:
            return pred.sum() * 0.0
        return masked.mean()

class WristAngleMAE(Metric):
    """Angular mean absolute error for wrist angles (channels 20, 21)."""

    def __call__(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
        stage: str,
    ) -> dict[str, torch.Tensor]:
        # Only compute if model outputs >= 22 channels
        if pred.shape[1] < 22:
            return {
                f"{stage}_mae_wrist_flexion": pred.sum() * 0.0,
                f"{stage}_mae_wrist_deviation": pred.sum() * 0.0,
            }

        mask_flexion = mask.unsqueeze(1).expand(-1, 1, -1)
        mask_deviation = mask.unsqueeze(1).expand(-1, 1, -1)
        if mask.sum() == 0:
            return {
                f"{stage}_mae_wrist_flexion": pred.sum() * 0.0,
                f"{stage}_mae_wrist_deviation": pred.sum() * 0.0,
            }
        return {
            f"{stage}_mae_wrist_flexion": torch.nn.L1Loss()(pred[:, 20:21][mask_flexion], target[:, 20:21][mask_flexion]),
            f"{stage}_mae_wrist_deviation": torch.nn.L1Loss()(pred[:, 21:22][mask_deviation], target[:, 21:22][mask_deviation]),
        }


def get_default_metrics() -> list[Metric]:
    return [
        AngleMAE(),
        AnglularDerivatives(),
        PerFingerAngleMAE(),
        PDAngleMAE(),
        WristAngleMAE(),
        LandmarkDistances(),
    ]
