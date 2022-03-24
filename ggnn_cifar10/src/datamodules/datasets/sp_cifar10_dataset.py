import os
from typing import Callable, Optional

import numpy as np
import torch
from torch_geometric.data import InMemoryDataset
from torchvision.datasets import CIFAR10
from torch_geometric.data import Data
from src.utils.superpixel_generation import convert_numpy_img_to_superpixel_graph


class CIFAR10SuperpixelsDataset(InMemoryDataset):
    """Dataset which converts CIFAR10 to dataset of superpixel graphs (on first run only)."""

    def __init__(
        self,
        data_dir: str = "data/",
        num_workers: int = 4,
        n_segments: int = 100,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        **kwargs,
    ):
        self.data_dir = os.path.join(data_dir, "CIFAR10")
        self.num_workers = num_workers
        self.n_segments = n_segments
        self.slic_kwargs = kwargs

        super().__init__(self.data_dir, transform, pre_transform, pre_filter)

        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        """Dynamically generates filename for processed dataset based on superpixel parameters."""
        filename = ""
        filename += f"sp({self.n_segments})"
        for name, value in self.slic_kwargs.items():
            filename += f"_{name}({value})"
        filename += ".pt"
        return filename

    def process(self):
        trainset = CIFAR10(self.data_dir, train=True, download=True)
        testset = CIFAR10(self.data_dir, train=False, download=True)


        labels = np.concatenate((trainset.targets, testset.targets))
        images = np.concatenate((trainset.data, testset.data))
        data_list = []
        print("aaaaaa")
        print(trainset.data.type())

        print(trainset.targets[0])
        print(trainset.data[0].shape)
        for i in range(images.shape[0]):
            x, edge_index, pos, y = convert_numpy_img_to_superpixel_graph(images[i], labels[i], self.n_segments)
            x = torch.as_tensor(x, dtype=torch.float32)
            edge_index = torch.as_tensor(edge_index, dtype=torch.long).T
            pos = torch.as_tensor(pos, dtype=torch.float32)
            y = torch.as_tensor(y, dtype=torch.long)
            data_list.append(Data(x=x, edge_index=edge_index, pos=pos, y=y))
            if (i%500 ==0):
                print(i/images.shape[0])


        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
