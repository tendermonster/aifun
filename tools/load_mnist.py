import numpy as np
import os
import torch
from torchvision import transforms, datasets

# from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import download_and_extract_archive

import ssl

ssl._create_default_https_context = ssl._create_unverified_context


class MNIST10:
    dataset_path = "dataset/mnist10"

    shape = (3, 28, 28)
    img_wh = 28
    # (tensor([0.1309, 0.1309, 0.1309]), tensor([0.3018, 0.3018, 0.3018]))
    # this is 3 channel grayscale mean/std of mnist
    # data_mean = [0.1309, 0.1309, 0.1309]
    # data_std = [0.3018, 0.3018, 0.3018]

    # for domain gap we use the mean and std of mnist-m
    data_mean = [0.4412, 0.4674, 0.4427]
    data_std = [0.1876, 0.2215, 0.1952]

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
        self._download_mnistm10()
        self.train, self.test, self.val = self._format_data(split)
        # TODO also return the file names of the images

    def set_logger(self, logger):
        self.logger = logger

    def _download_mnistm10(self) -> bool:
        """Download the MNIST-M data."""
        if os.path.exists(self.dataset_path):
            return

        os.makedirs(self.dataset_path, exist_ok=True)
        # os.makedirs(self.processed_folder, exist_ok=True)

        # download files
        datasets.MNIST(self.dataset_path, train=True, download=True)
        datasets.MNIST(self.dataset_path, train=False, download=True)

    def _get_mnist(self):
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
        all_filenames = []
        return all_data, all_labels, all_filenames

    def _compute_mean_std(self):
        data, _, _ = self._get_mnist()
        data = data.view(-1, 3, self.img_wh, self.img_wh)
        # todo make it work with any shape
        data = data / 255.0
        # print(data.min())
        # print(data.max())
        mean = torch.mean(data, dim=(2, 3))
        mean = torch.mean(mean, dim=0)
        std = torch.std(data, dim=(2, 3))
        std = torch.mean(std, dim=0)

        # print(std)
        # print(mean)
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
        # print(std_new)
        # print(mean_new)
        return mean, std

    def augment_train(self, data):
        # random flipping
        # random cropping
        # random rotation
        # random color jitter
        # gausian noise

        # for now just to play it safe normalization is done after augmentation

        aug_part1 = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomCrop(self.img_wh, padding=4),
            ]
        )
        data = aug_part1(data) / 255.0
        aug_part2 = transforms.Compose(
            [
                transforms.Normalize(
                    mean=self.data_mean, std=self.data_std
                ),  # normalize for values between 0 and 1
            ]
        )
        return aug_part2(data)

    def augment_test(self, data):
        data = data / 255.0
        aug = transforms.Compose(
            [
                transforms.Normalize(
                    mean=self.data_mean, std=self.data_std
                ),  # normalize for values between 0 and 1
            ]
        )
        return aug(data)

    def augment_input(self, data):
        """
        Converts PIL image to proper model input

        Args:
            data (PIL.Image): PIL image

        Returns:
            torch.Tensor: model input
        """
        aug = transforms.Compose(
            [
                transforms.Resize(self.img_wh),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.data_mean, std=self.data_std),
            ]
        )
        return aug(data)

    def normalize_data(self, data):
        # devide by 255
        data = data / 255.0
        # normalize
        std = torch.tensor(self.data_std)
        mean = torch.tensor(self.data_mean)
        data[:, 0, :, :] = data[:, 0, :, :] - mean[0]
        data[:, 1, :, :] = data[:, 1, :, :] - mean[1]
        data[:, 2, :, :] = data[:, 2, :, :] - mean[2]
        data[:, 0, :, :] = data[:, 0, :, :] / std[0]
        data[:, 1, :, :] = data[:, 1, :, :] / std[1]
        data[:, 2, :, :] = data[:, 2, :, :] / std[2]
        return data

    def _split_data(self, data, split=[0.70, 0.15, 0.15]):
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

    def _format_data(self, split):
        data, labels, _ = self._get_mnist()
        # convert to torch tensor
        data = torch.tensor(data, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)
        # reshape data
        print(data.shape)
        data = data.view(-1, 3, self.img_wh, self.img_wh)
        train, test, val = self._split_data(data, split)
        # augment training dataset
        # training dataset is augmented during training process
        # after augmentation make sure that all the datasets have similar mean and std
        # otherwise this would mean that the augmentation is not done properly for training dataset
        # also all min / max values should be similar
        train_labels, test_labels, val_labels = self._split_data(labels, split)
        train = list(zip(train, train_labels))
        test = list(zip(test, test_labels))
        val = list(zip(val, val_labels))
        # train_fn, test_fn, val_fn = self._split_data(filenames, split)
        assert len(train) == len(train_labels)
        assert len(test) == len(test_labels)
        assert len(val) == len(val_labels)
        return train, test, val


if __name__ == "__main__":
    c = MNIST10()

    print(c._compute_mean_std())
