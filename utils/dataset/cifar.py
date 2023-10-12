import pickle
from typing_extensions import override
import os
import tarfile
import requests
import torch

from utils.dataset import Dataset


class CIFAR10(Dataset):
    cifar_tar = "/tmp/cifar-10-python.tar.gz"
    cifar_untar_dir = "dataset/"
    # statistics of training set
    # std tensor([0.0606, 0.0612, 0.0677])
    # mean tensor([0.4914, 0.4822, 0.4465])

    def __init__(self, split=[0.70, 0.15, 0.15]) -> None:
        super().__init__(
            img_wh=32,
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.0606, 0.0612, 0.0677],
            split=split,
        )

    def set_logger(self, logger):
        self.logger = logger

    def __unpickle(self, file):
        with open(file, "rb") as fo:
            dict = pickle.load(fo, encoding="bytes")
        return dict

    @override
    def download_dataset(self) -> bool:
        if os.path.exists("dataset/cifar-10-batches-py"):
            return False
        else:
            # download using requests
            url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
            r = requests.get(url, allow_redirects=True)
            with open(self.cifar_tar, "wb") as f:
                f.write(r.content)
            # extract using tar
            self.__untar_cifar10()
            return True

    def __untar_cifar10(self):
        # open file
        file = tarfile.open(self.cifar_tar)
        # extract files
        file.extractall(self.cifar_untar_dir)
        # close file
        file.close()

    @override
    def load_dataset(self):
        all_objects = []
        for i in range(0, 5):
            all_objects.append(
                self.__unpickle("dataset/cifar-10-batches-py/data_batch_" + str(i + 1))
            )
        all_objects.append(self.__unpickle("dataset/cifar-10-batches-py/test_batch"))
        # [b'batch_label', b'labels', b'data', b'filenames']
        all_data = []
        all_labels = []
        for piece in all_objects:
            all_data.append(piece[b"data"])
            all_labels.append(piece[b"labels"])
        # to torch
        all_data = [torch.tensor(x, dtype=torch.float32) for x in all_data]
        all_labels = [torch.tensor(x, dtype=torch.int64) for x in all_labels]
        all_data = torch.cat(all_data)
        all_labels = torch.cat(all_labels)
        return all_data, all_labels


if __name__ == "__main__":
    c = CIFAR10()
    print(c.compute_mean_std())
