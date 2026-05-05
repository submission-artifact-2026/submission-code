from __future__ import annotations

from typing import TYPE_CHECKING

__all__ = [
    "Emg2PoseDataset",
    "PretrainWrapperDataset",
    "EgoEmgMemmapDataset",
    "EgoEmgVisionDataset",
]

if TYPE_CHECKING:
    from emg2pose.datasets.emg2pose_dataset import Emg2PoseDataset
    from emg2pose.datasets.pretrain_wrapper import PretrainWrapperDataset
    from emg2pose.datasets.egoemg_memmap_dataset import EgoEmgMemmapDataset
    from emg2pose.datasets.egoemg_vision_dataset import EgoEmgVisionDataset


def __getattr__(name: str):
    if name == "Emg2PoseDataset":
        from emg2pose.datasets.emg2pose_dataset import Emg2PoseDataset

        return Emg2PoseDataset
    if name == "PretrainWrapperDataset":
        from emg2pose.datasets.pretrain_wrapper import PretrainWrapperDataset

        return PretrainWrapperDataset
    if name == "EgoEmgMemmapDataset":
        from emg2pose.datasets.egoemg_memmap_dataset import EgoEmgMemmapDataset

        return EgoEmgMemmapDataset
    if name == "EgoEmgVisionDataset":
        from emg2pose.datasets.egoemg_vision_dataset import EgoEmgVisionDataset

        return EgoEmgVisionDataset
    raise AttributeError(name)
