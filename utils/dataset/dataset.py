# TODO this class should ca a root class that other loaders extend to save coding
from __future__ import annotations
from abc import ABC, abstractmethod
from re import S
import torch
import numpy as np
import ssl
import typing
from torchvision import transforms
from matplotlib import pyplot as plt
from torchvision.transforms.functional import InterpolationMode

# from einops.layers.torch import Rearrange
import einops

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
        # load dataset should return the dataset and labels
        # dataset should be a tensor of shape (N, C, H, W)
        # labels should be a tensor of shape (N, )
        raise NotImplementedError

    def get_mean(self):
        return self.mean

    def get_std(self):
        return self.std

    def compute_mean_std(self) -> Tuple[Tensor, Tensor]:
        # compute the mean and std of the dataset in range of [0,1]
        data = [i[0].unsqueeze(0) for i in self.get_dataset()]
        print(data[0].shape)
        data = torch.cat(data, dim=0)
        # data = data.view(-1, 3, self.img_wh, self.img_wh)
        # todo make it work with any shape
        data = data / 255.0
        # Compute mean along dimensions 0 (images), 2 (height), and 3 (width)
        mean = torch.mean(data, dim=(0, 2, 3))
        std = torch.std(data, dim=(0, 2, 3))
        return mean, std

    def __normalize(self, data: Tensor) -> Tensor:
        # it seems that the normalization is more robust for od
        # but experiment with min/max standardization in future
        aug = transforms.Compose(
            [
                transforms.Normalize(
                    mean=self.mean, std=self.std
                ),  # normalize for values between 0 and  for original dataset
            ]
        )
        return aug(data)

    def __standardize(self, data: Tensor) -> Tensor:
        return data / 255.0

    def __resize(self, data: Tensor) -> Tensor:
        aug = transforms.Compose(
            [
                transforms.Resize(
                    self.img_wh_net, interpolation=InterpolationMode.NEAREST
                ),
            ]
        )
        return aug(data)

    def augment_train(self, data: Tensor):
        # for now just to play it safe normalization is done after augmentation
        # TODO see if bbox need adjustement if transforms is used
        aug = transforms.Compose(
            [
                transforms.Resize(
                    self.img_wh_net, interpolation=InterpolationMode.NEAREST
                ),
                transforms.RandomHorizontalFlip(p=0.25),
                transforms.RandomVerticalFlip(p=0.25),
                transforms.RandomErasing(p=0.5, value="random"),
                # transforms.RandomAutocontrast(),
            ]
        )
        data = aug(data)
        data = self.__standardize(data)
        data = self.__normalize(data)
        return data

    def augment_test(self, data: Tensor):
        data = self.__resize(data)
        data = self.__standardize(data)
        data = self.__normalize(data)
        return data

    def augment_input(self, data: Image) -> Tensor:
        """
        Converts PIL image to proper model input

        Args:
            data (PIL.Image): PIL image

        Returns:
            torch.Tensor: model input
        """
        aug = transforms.Compose(
            [
                transforms.PILToTensor(),
            ]
        )
        data_tensor: Tensor = aug(data)
        data_tensor = self.__resize(data_tensor)
        data_tensor = self.__standardize(data_tensor)
        data_tensor = self.__normalize(data_tensor)
        return data_tensor

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
        data = data.clone().detach().to(dtype=torch.float32)
        labels = labels.clone().detach().to(dtype=torch.uint8)
        assert data.dtype == torch.float32
        assert labels.dtype == torch.uint8
        # should have this shape (N, C, H, W)
        assert data.shape == (len(data), 3, self.img_wh, self.img_wh)
        # data should be correctly shaped at this point!
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
        data = self.get_dataset()
        for i in data[:10]:
            im, label = i
            im = self.__standardize(im)
            plt.imshow(im.permute(1, 2, 0))
            plt.title(str(label))
            plt.show()
