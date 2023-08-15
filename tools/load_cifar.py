import pickle
from typing import Dict, List
import numpy as np
import os
import tarfile
import requests
import torch


class CIFAR10:
    cifar_tar = "/tmp/cifar-10-python.tar.gz"
    cifar_untar_dir = "dataset/"

    def __init__(self) -> None:
        self._download_cifar10()
        self.data, self.labels, self.filenames = self._format_data()

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

    def _format_data(self):
        data, labels, filenames = self._get_cifar()
        # convert to torch tensor
        data = torch.tensor(data, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)
        # reshape data
        data = data.view(-1, 3, 32, 32)
        return data, labels, filenames


if __name__ == "__main__":
    c = CIFAR10()
