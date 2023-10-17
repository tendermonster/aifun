# TODO this class should ca a root class that other loaders extend to save coding
from __future__ import annotations
from abc import ABC, abstractmethod
from re import S
import torch
import numpy as np
import ssl
import typing
from torchvision import transforms

ssl._create_default_https_context = ssl._create_unverified_context

if typing.TYPE_CHECKING:
    from typing import List, Tuple
    from torch import Tensor
    from PIL.Image import Image


class Dataset(ABC):
    def __init__(
        self,
        img_wh: int,
        img_wh_net: int,
        mean: List[float],
        std: List[float],
        split: List[float] = [0.70, 0.15, 0.15],
    ) -> None:
        self.download_dataset()
        self.std = std
        self.mean = mean
        self.img_wh = img_wh
        self.img_wh_net = img_wh_net
        self.split = split
        self.__train, self.__test, self.__val = self.__format_data(self.split)

    @abstractmethod
    def download_dataset(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def load_dataset(self) -> Tuple[Tensor, Tensor]:
        raise NotImplementedError

    def compute_mean_std(self) -> Tuple[Tensor, Tensor]:
        data, _ = self.load_dataset()
        data = data.view(-1, 3, self.img_wh, self.img_wh)

        # todo make it work with any shape
        data = data / data.max()
        mean = torch.mean(data, dim=(2, 3))
        mean = torch.mean(mean, dim=0)
        std = torch.std(data, dim=(2, 3))
        std = torch.mean(std, dim=0)

        data[:, 0, :, :] = data[:, 0, :, :] - mean[0]
        data[:, 1, :, :] = data[:, 1, :, :] - mean[1]
        data[:, 2, :, :] = data[:, 2, :, :] - mean[2]
        data[:, 0, :, :] = data[:, 0, :, :] / std[0]
        data[:, 1, :, :] = data[:, 1, :, :] / std[1]
        data[:, 2, :, :] = data[:, 2, :, :] / std[2]

        # write assertion test to test of mean is 0 and std is 1
        mean_new = torch.mean(data, dim=(2, 3))
        mean_new = torch.mean(mean_new, dim=0)
        std_new = torch.std(data, dim=(2, 3))
        std_new = torch.std(std_new, dim=0)
        return mean, std

    def augment_train(self, data: Tensor):
        # random flipping
        # random cropping
        # random rotation
        # random color jitter
        # gausian noise

        # for now just to play it safe normalization is done after augmentation

        aug_part1 = transforms.Compose(
            [
                transforms.Resize(self.img_wh_net),
                transforms.RandomHorizontalFlip(p=0.25),
                transforms.RandomVerticalFlip(p=0.25),
                transforms.RandomErasing(p=0.5, value="random"),
            ]
        )
        data = aug_part1(data) / 255.0
        aug_part2 = transforms.Compose(
            [
                transforms.Normalize(
                    mean=self.mean, std=self.std
                ),  # normalize for values between 0 and 1
            ]
        )
        return aug_part2(data)

    def augment_test(self, data: Tensor):
        aug1 = transforms.Compose(
            [
                transforms.Resize(self.img_wh_net),
            ]
        )
        data = aug1(data)
        data = data / 255.0
        aug2 = transforms.Compose(
            [
                transforms.Normalize(
                    mean=self.mean, std=self.std
                ),  # normalize for values between 0 and 1
            ]
        )
        return aug2(data)

    def augment_input(self, data: Image):
        """
        Converts PIL image to proper model input

        Args:
            data (PIL.Image): PIL image

        Returns:
            torch.Tensor: model input
        """
        aug = transforms.Compose(
            [
                transforms.Resize(self.img_wh_net),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std),
            ]
        )
        return aug(data)

    def __split_data(
        self, data, split=[0.70, 0.15, 0.15]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns train test and validation sets

        Args:
            data (_type_): dataset
            split (_type_): split ratio

        Returns:
            _type_: train, test, validation sets
        """
        split = np.array(split) / np.sum(split)
        train = round(split[0], 2)
        test = round(split[1], 2)
        val = round(1 - train - test, 2)
        split = [train, test, val]
        assert sum(split) == 1
        train_size = int(split[0] * len(data))
        test_size = int(split[1] * len(data))
        val_size = len(data) - train_size - test_size
        train, val, test = (
            data[:train_size],
            data[train_size : train_size + test_size],
            data[-val_size:],
        )
        assert len(train) + len(test) + len(val) == len(data)
        return (
            train,
            test,
            val,
        )

    def __format_data(
        self, split
    ) -> Tuple[
        List[Tuple[Tensor, Tensor]],
        List[Tuple[Tensor, Tensor]],
        List[Tuple[Tensor, Tensor]],
    ]:
        data, labels = self.load_dataset()
        # convert to torch tensor
        data = torch.tensor(data, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)
        # reshape data
        print(data.shape)
        data = data.view(-1, 3, self.img_wh, self.img_wh)
        train, test, val = self.__split_data(data, split)
        # augment training dataset
        # training dataset is augmented during training process
        # after augmentation make sure that all the datasets have similar mean and std
        # otherwise this would mean that the augmentation is not done properly for training dataset
        # also all min / max values should be similar
        train_labels, test_labels, val_labels = self.__split_data(labels, split)
        train = list(zip(train, train_labels))
        test = list(zip(test, test_labels))
        val = list(zip(val, val_labels))
        assert len(train) == len(train_labels)
        assert len(test) == len(test_labels)
        assert len(val) == len(val_labels)
        return train, test, val

    def get_dataset(self) -> list[Tuple[Tensor, Tensor]]:
        return self.__train + self.__test + self.__val

    def get_train(self):
        return self.__train

    def get_test(self):
        return self.__test

    def get_val(self):
        return self.__val

    def visualize_examples(self):
        import matplotlib.pyplot as plt

        data = self.get_dataset()
        for i in data[:10]:
            im, label = i
            plt.imshow(im.permute(1, 2, 0))
            plt.title(str(label))
            plt.show()
