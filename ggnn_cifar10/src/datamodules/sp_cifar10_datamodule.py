from typing import Optional, Sequence

import torch_geometric.transforms as T
from pytorch_lightning import LightningDataModule
from torch.utils.data import random_split
from torch_geometric.data import DataLoader, Dataset
from torch_geometric.transforms import NormalizeScale

from src.datamodules.datasets.sp_cifar10_dataset import CIFAR10SuperpixelsDataset


class CIFAR10SuperpixelsDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data/",
        train_val_test_split: Sequence[int] = (1300, 500, 200),
        n_segments: int = 100,
        sp_generation_workers: int = 4,
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = False,
        **kwargs,
    ):

        super().__init__()

        self.data_dir = data_dir
        self.train_val_test_split = train_val_test_split

        # superpixel graph parameters
        self.n_segments = n_segments
        self.sp_generation_workers = sp_generation_workers

        # dataloader parameters
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.slic_kwargs = kwargs

        self.pre_transform = T.Compose(
            [
                NormalizeScale(),
            ]
        )
        self.transform = None
        self.pre_filter = None

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    @property
    def num_classes() -> int:
        return 10

    @property
    def num_node_features() -> int:
        return 3

    def prepare_data(self):
        """Download data if needed. Generate superpixel graphs. Apply pre-transforms.
        This method is called only from a single GPU. Do not use it to assign state (self.x = y)."""
        CIFAR10SuperpixelsDataset(
            data_dir=self.data_dir,
            n_segments=self.n_segments,
            num_workers=self.sp_generation_workers,
            transform=self.transform,
            pre_transform=self.pre_transform,
            pre_filter=self.pre_filter,
            **self.slic_kwargs,
        )

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: self.data_train, self.data_val, self.data_test."""
        if not self.data_train and not self.data_val and not self.data_test:
            dataset = CIFAR10SuperpixelsDataset(
                data_dir=self.data_dir,
                n_segments=self.n_segments,
                num_workers=self.sp_generation_workers,
                transform=self.transform,
                pre_transform=self.pre_transform,
                pre_filter=self.pre_filter,
                **self.slic_kwargs,
            )
            self.data_train, self.data_val, self.data_test = random_split(
                dataset, self.train_val_test_split
            )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )
