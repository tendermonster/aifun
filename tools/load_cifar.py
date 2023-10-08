import pickle
from typing import Dict, List
import numpy as np
import os
import tarfile
import requests
import torch
from torchvision import transforms


class CIFAR10:
    cifar_tar = "/tmp/cifar-10-python.tar.gz"
    cifar_untar_dir = "dataset/"
    # statistics of training set
    # std tensor([0.0606, 0.0612, 0.0677])
    # mean tensor([0.4914, 0.4822, 0.4465])

    def __init__(self, split=[0.70, 0.15, 0.15]) -> None:
        self._download_cifar10()
        self.train, self.test, self.val = self._format_data(split)
        # TODO also return the file names of the images

    def set_logger(self, logger):
        self.logger = logger

    def _unpickle(self, file):
        with open(file, "rb") as fo:
            dict = pickle.load(fo, encoding="bytes")
        return dict

    def _download_cifar10(self) -> bool:
        if os.path.exists("dataset/cifar-10-batches-py"):
            return
        else:
            # download using requests
            url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
            r = requests.get(url, allow_redirects=True)
            with open(self.cifar_tar, "wb") as f:
                f.write(r.content)
            # extract using tar
            self._untar_cifar10()

    def _untar_cifar10(self):
        # open file
        file = tarfile.open(self.cifar_tar)
        # extract files
        file.extractall(self.cifar_untar_dir)
        # close file
        file.close()

    def _load_data(self) -> List[Dict]:
        all_objects = []
        for i in range(0, 5):
            all_objects.append(
                self._unpickle("dataset/cifar-10-batches-py/data_batch_" + str(i + 1))
            )
        all_objects.append(self._unpickle("dataset/cifar-10-batches-py/test_batch"))
        return all_objects

    def _get_cifar(self):
        # [b'batch_label', b'labels', b'data', b'filenames']
        all_objects = self._load_data()
        all_data = []
        all_labels = []
        all_filenames = []
        for piece in all_objects:
            all_data.append(piece[b"data"])
            all_labels.append(piece[b"labels"])
            all_filenames.append(piece[b"filenames"])
        # to torch
        all_data = [torch.tensor(x, dtype=torch.float32) for x in all_data]
        all_labels = [torch.tensor(x, dtype=torch.int64) for x in all_labels]
        all_data = torch.cat(all_data)
        all_labels = torch.cat(all_labels)
        all_filenames = np.concatenate(all_filenames)
        return all_data, all_labels, all_filenames

    def _compute_mean_std(self):
        data, _, _ = self._get_cifar()
        data = data.view(-1, 3, 32, 32)
        # todo make it work with any shape
        data = data / 255.0
        print(data.min())
        print(data.max())
        mean = torch.mean(data, dim=(2, 3))
        mean = torch.mean(mean, dim=0)
        std = torch.std(data, dim=(2, 3))
        std = torch.mean(std, dim=0)

        print(std)
        print(mean)
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
        print(std_new)
        print(mean_new)
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
                transforms.RandomCrop(32, padding=4),
            ]
        )
        data = aug_part1(data) / 255.0
        aug_part2 = transforms.Compose(
            [
                transforms.Normalize(
                    mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]
                ),  # normalize for values between 0 and 1
            ]
        )
        return aug_part2(data)

    def augment_test(self, data):
        data = data / 255.0
        aug = transforms.Compose(
            [
                transforms.Normalize(
                    mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]
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
                transforms.Resize(32),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]
                ),
            ]
        )
        return aug(data)

    def normalize_data(self, data):
        # devide by 255
        data = data / 255.0
        # normalize
        std = torch.tensor([0.2023, 0.1994, 0.2010])
        mean = torch.tensor([0.4914, 0.4822, 0.4465])
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
        data, labels, filenames = self._get_cifar()
        # convert to torch tensor
        data = torch.tensor(data, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)
        # reshape data
        data = data.view(-1, 3, 32, 32)
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
        train_fn, test_fn, val_fn = self._split_data(filenames, split)
        assert len(train) == len(train_labels) == len(train_fn)
        assert len(test) == len(test_labels) == len(test_fn)
        assert len(val) == len(val_labels) == len(val_fn)
        return train, test, val


if __name__ == "__main__":
    c = CIFAR10()
    print(c._compute_mean_std())
