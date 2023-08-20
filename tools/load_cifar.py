import pickle
from typing import Dict, List
import numpy as np
import os
import tarfile
import requests
import torch
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from typing import Tuple


class CIFAR10:
    cifar_tar = "/tmp/cifar-10-python.tar.gz"
    cifar_untar_dir = "dataset/"
    # statistics of training set
    # std tensor([0.0606, 0.0612, 0.0677])
    # mean tensor([0.4914, 0.4822, 0.4465])

    def __init__(self, split=[0.70, 0.15, 0.15]) -> None:
        self._download_cifar10()
        self.data, self.labels, self.filenames = self._format_data()
        self.train, self.test, self.val = self._split_data(split)

    def set_logger(self, logger):
        self.logger = logger

    def _split_data(self, split) -> Tuple[Subset, Subset, Subset]:
        generator = torch.Generator().manual_seed(42)
        all_data = list(zip(self.data, self.labels))
        data_split = random_split(all_data, split, generator=generator)
        # return train, test, validation datasets
        train = data_split[0]
        test = data_split[1]
        val = data_split[2]
        return train, test, val

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
        for i in range(0, 5):
            all_data.append(all_objects[i][b"data"])
            all_labels.append(all_objects[i][b"labels"])
            all_filenames.append(all_objects[i][b"filenames"])
        all_data = np.concatenate(all_data)
        all_labels = np.concatenate(all_labels)
        all_filenames = np.concatenate(all_filenames)
        return all_data, all_labels, all_filenames

    def _compute_mean_std(self, data):
        # todo make it work with any shape
        data = data / 255
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

        return data

    def normalize_data(self, data):
        # devide by 255
        data = data / 255
        # normalize
        # std tensor([0.2023, 0.1994, 0.2010])
        # mean tensor([0.4914, 0.4822, 0.4465])
        std = torch.tensor([0.2023, 0.1994, 0.2010])
        mean = torch.tensor([0.4914, 0.4822, 0.4465])
        data[:, 0, :, :] = data[:, 0, :, :] - mean[0]
        data[:, 1, :, :] = data[:, 1, :, :] - mean[1]
        data[:, 2, :, :] = data[:, 2, :, :] - mean[2]
        data[:, 0, :, :] = data[:, 0, :, :] / std[0]
        data[:, 1, :, :] = data[:, 1, :, :] / std[1]
        data[:, 2, :, :] = data[:, 2, :, :] / std[2]
        return data

    def _format_data(self):
        data, labels, filenames = self._get_cifar()
        # convert to torch tensor
        data = torch.tensor(data, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)
        # reshape data
        data = data.view(-1, 3, 32, 32)
        data = self.normalize_data(data)
        return data, labels, filenames


if __name__ == "__main__":
    c = CIFAR10()
