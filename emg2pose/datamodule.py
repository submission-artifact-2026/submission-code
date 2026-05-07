from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from emg2pose import transforms
from emg2pose.datasets.pretrain_wrapper import _resolve_transform

log = logging.getLogger(__name__)


def make_data_module(config: DictConfig) -> WindowedEmgDataModule:
    """Create and configure datamodule from experiment config."""
    dataset_conf = OmegaConf.to_container(config.dataset, resolve=True)
    datamodule = instantiate(
        config.datamodule,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        dataset_conf=dataset_conf,
        _recursive_=False,
    )
    datamodule.train_transforms = transforms.Compose(
        [instantiate(cfg) for cfg in config.transforms.train]
    )
    datamodule.val_transforms = transforms.Compose(
        [instantiate(cfg) for cfg in config.transforms.val]
    )
    datamodule.test_transforms = transforms.Compose(
        [instantiate(cfg) for cfg in config.transforms.test]
    )
    return datamodule


class _EmptyDataset(Dataset):
    def __len__(self) -> int:
        return 0

    def __getitem__(self, idx: int) -> Any:
        raise IndexError(idx)


class WindowedEmgDataModule(pl.LightningDataModule):
    def __init__(
        self,
        window_length: int,
        stride: int | None,
        padding: tuple[int, int],
        batch_size: int,
        num_workers: int,
        val_test_window_length: int | None = None,
        val_test_stride: int | None = None,
        skip_ik_failures: bool = False,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        prefetch_factor: int = 2,
        max_open_files: int = 32,
        dataset_repeat: int = 1,
        norm_mode: str | None = None,
        norm_stats_path: str | None = None,
        per_dataset_norm_stats_path: str | None = None,
        norm_eps: float = 1e-6,
        dataset_conf: DictConfig | None = None,
    ) -> None:
        super().__init__()
        self.window_length = window_length
        self.val_test_window_length = val_test_window_length or window_length
        self.stride = stride
        self.val_test_stride = val_test_stride if val_test_stride is not None else stride
        self.padding = padding
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.prefetch_factor = prefetch_factor
        self.max_open_files = max_open_files
        self.dataset_repeat = dataset_repeat
        self.norm_mode = norm_mode
        self.norm_stats_path = norm_stats_path
        self.per_dataset_norm_stats_path = per_dataset_norm_stats_path
        self.norm_eps = norm_eps

        self.dataset_conf = dataset_conf or {}
        self.train_transforms = None
        self.val_transforms = None
        self.test_transforms = None

        self.train_dataset: Dataset | None = None
        self.val_dataset: Dataset | None = None
        self.test_dataset: Dataset | None = None

    def _normalize_dataset_configs(self, cfgs: Any) -> list[Any]:
        if cfgs is None:
            return []
        if OmegaConf.is_dict(cfgs):
            return [cfgs]
        if OmegaConf.is_list(cfgs):
            return list(cfgs)
        if isinstance(cfgs, Sequence) and not isinstance(cfgs, (str, bytes)):
            return list(cfgs)
        return [cfgs]

    def _build_dataset(self, split: str, transform: Any) -> Dataset:
        cfgs = self._normalize_dataset_configs(self.dataset_conf.get(split, []))
        datasets: list[Dataset] = []
        dataset_info: list[tuple[str, int]] = []

        for cfg in cfgs:
            # Inject norm settings into PretrainWrapperDataset configs
            if self.norm_mode == "per-dataset":
                cfg = OmegaConf.merge(
                    cfg,
                    {
                        "norm_mode": self.norm_mode,
                        "norm_stats_path": self.per_dataset_norm_stats_path,
                    },
                )

            # Instantiate without transform parameter
            dataset = instantiate(cfg, transform=None)
            # Set transform directly after instantiation to avoid conversion
            if hasattr(dataset, 'transform'):
                dataset.transform = transform
            # Also set _transform if it exists (for PretrainWrapperDataset)
            if hasattr(dataset, '_transform'):
                dataset._transform = _resolve_transform(transform) if transform is not None else None

            # Collect dataset info
            dataset_name = getattr(dataset, 'name', dataset.__class__.__name__)
            dataset_len = len(dataset)
            dataset_info.append((dataset_name, dataset_len))
            datasets.append(dataset)

        if not datasets:
            return _EmptyDataset()

        # Print dataset composition with aggregation
        if dataset_info:
            total_samples = sum(length for _, length in dataset_info)

            # Aggregate by dataset name
            from collections import defaultdict
            aggregated = defaultdict(lambda: {'count': 0, 'samples': 0})
            for name, length in dataset_info:
                aggregated[name]['count'] += 1
                aggregated[name]['samples'] += length

            log.info(f"\n{'='*70}")
            log.info(f"Dataset composition for '{split}' split:")
            log.info(f"{'-'*70}")

            for name in sorted(aggregated.keys()):
                info = aggregated[name]
                count = info['count']
                samples = info['samples']
                percentage = (samples / total_samples * 100) if total_samples > 0 else 0

                if count > 1:
                    log.info(f"  {name:20s}: {samples:8d} samples ({percentage:5.2f}%) [{count} subsets]")
                else:
                    log.info(f"  {name:20s}: {samples:8d} samples ({percentage:5.2f}%)")

            log.info(f"{'-'*70}")
            log.info(f"  {'Total':20s}: {total_samples:8d} samples")
            log.info(f"{'='*70}\n")

        if len(datasets) == 1:
            dataset = datasets[0]
        else:
            dataset = ConcatDataset(datasets)

        if self.dataset_repeat > 1:
            dataset = ConcatDataset([dataset] * self.dataset_repeat)
            log.info("Dataset repeated %dx → %d total samples", self.dataset_repeat, len(dataset))
        return dataset

    def setup(self, stage: str | None = None) -> None:
        if stage in (None, "fit"):
            self.train_dataset = self._build_dataset("train", self.train_transforms)
            self.val_dataset = self._build_dataset("val", self.val_transforms)
        if stage in (None, "validate"):
            if self.val_dataset is None:
                self.val_dataset = self._build_dataset("val", self.val_transforms)
        if stage in (None, "test"):
            self.test_dataset = self._build_dataset("test", self.test_transforms)
        if stage in (None, "predict"):
            if self.test_dataset is None:
                self.test_dataset = self._build_dataset("test", self.test_transforms)

    def _make_loader(self, dataset: Dataset, shuffle: bool) -> DataLoader:
        kwargs: dict[str, Any] = {
            "dataset": dataset,
            "batch_size": self.batch_size,
            "shuffle": shuffle,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "collate_fn": self._collate_fn,  # Use custom collate function
        }
        if self.num_workers > 0:
            kwargs["persistent_workers"] = self.persistent_workers
            kwargs["prefetch_factor"] = self.prefetch_factor
        return DataLoader(**kwargs)

    @staticmethod
    def _collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
        """Custom collate function to handle dataset_name."""
        import torch
        from torch.utils.data import default_collate

        if not batch:
            return {}

        dataset_names = [sample.pop("dataset_name", "unknown")
                        for sample in batch]

        # Use default collate for other fields
        collated = default_collate(batch)

        # Add dataset_name as list (not stacked)
        collated["dataset_name"] = dataset_names

        return collated

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            self.train_dataset = self._build_dataset("train", self.train_transforms)
        return self._make_loader(self.train_dataset, shuffle=True)

    def val_dataloader(self):
        if self.val_dataset is None:
            self.val_dataset = self._build_dataset("val", self.val_transforms)
        if len(self.val_dataset) == 0:
            return []
        return self._make_loader(self.val_dataset, shuffle=False)

    def test_dataloader(self):
        if self.test_dataset is None:
            self.test_dataset = self._build_dataset("test", self.test_transforms)
        if len(self.test_dataset) == 0:
            return []
        return self._make_loader(self.test_dataset, shuffle=False)
