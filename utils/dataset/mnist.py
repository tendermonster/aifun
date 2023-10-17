from __future__ import annotations
import typing
from typing_extensions import override
import numpy as np
import os
import torch
from torchvision import transforms, datasets

import ssl

from utils.dataset.dataset import Dataset

ssl._create_default_https_context = ssl._create_unverified_context

if typing.TYPE_CHECKING:
    from torch import Tensor
    from typing import List, Tuple


class MNIST10(Dataset):
    dataset_path = "dataset/mnist10"

    shape = (3, 28, 28)
    # transform to
    img_wh = 28
    img_wh_net = 224
    # (tensor([0.1309, 0.1309, 0.1309]), tensor([0.3018, 0.3018, 0.3018]))
    # this is 3 channel grayscale mean/std of mnist
    data_mean = [0.1309, 0.1309, 0.1309]
    data_std = [0.3018, 0.3018, 0.3018]

    # for domain gap we use the mean and std of mnist-m

    training_file = "mnist_m_train.pt"
    test_file = "mnist_m_test.pt"
    classes = [
        "0 - zero",
        "1 - one",
        "2 - two",
        "3 - three",
        "4 - four",
        "5 - five",
        "6 - six",
        "7 - seven",
        "8 - eight",
        "9 - nine",
    ]

    def __init__(self, split=[0.70, 0.15, 0.15]) -> None:
        super().__init__(
            img_wh=self.img_wh,
            img_wh_net=self.img_wh_net,
            mean=self.data_mean,
            std=self.data_std,
            split=split,
        )

    def set_logger(self, logger):
        self.logger = logger

    @override
    def download_dataset(self):
        """Download the MNIST-M data."""
        if os.path.exists(self.dataset_path):
            return False

        os.makedirs(self.dataset_path, exist_ok=True)
        # os.makedirs(self.processed_folder, exist_ok=True)

        # download files
        datasets.MNIST(self.dataset_path, train=True, download=True)
        datasets.MNIST(self.dataset_path, train=False, download=True)
        return True

    @override
    def load_dataset(self) -> Tuple[Tensor, Tensor]:
        train = datasets.MNIST(
            self.dataset_path,
            train=True,
            download=False,
        )
        test = datasets.MNIST(
            self.dataset_path,
            train=False,
            download=False,
        )
        train_data = train.data
        # convert to (28,28) 3 channels
        train_data = train_data.unsqueeze(1)  # shape becomes (70000, 1, 28, 28)
        # Repeat across the channel dimension
        train_data = train_data.repeat(1, 3, 1, 1)
        train_targets = train.targets

        test_data = test.data
        test_data = test_data.unsqueeze(1)
        test_data = test_data.repeat(1, 3, 1, 1)
        test_targets = test.targets

        all_data = torch.cat((train_data, test_data))
        all_labels = torch.cat((train_targets, test_targets))
        return all_data, all_labels


if __name__ == "__main__":
    c = MNIST10()
    print(c.compute_mean_std())
