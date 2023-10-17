from __future__ import annotations
from typing_extensions import override
import os
import torch
import typing


# from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import download_and_extract_archive

import ssl

from utils.dataset.dataset import Dataset

ssl._create_default_https_context = ssl._create_unverified_context

if typing.TYPE_CHECKING:
    from torch import Tensor
    from typing import List, Tuple


class MNISTM10(Dataset):
    cifar_tar: str = "/tmp/mnistm-10-python.tar.gz"
    cifar_untar_dir: str = "dataset/"

    dataset_path: str = "dataset/mnistm10"
    dataset_extract_path: str = dataset_path + "/extract"
    dataset_download_path: str = dataset_path + "/download"

    shape = (3, 28, 28)
    img_wh = 28
    # img wh transform
    img_wh_net: int = 224
    mean: list[float] = [0.4412, 0.4674, 0.4427]
    std: list[float] = [0.1876, 0.2215, 0.1952]

    resources = [
        (
            "https://github.com/liyxi/mnist-m/releases/download/data/mnist_m_train.pt.tar.gz",
            "191ed53db9933bd85cc9700558847391",
        ),
        (
            "https://github.com/liyxi/mnist-m/releases/download/data/mnist_m_test.pt.tar.gz",
            "e11cb4d7fff76d7ec588b1134907db59",
        ),
    ]

    training_file: str = "mnist_m_train.pt"
    test_file: str = "mnist_m_test.pt"
    classes: list[str] = [
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

    def __init__(self, split: List[float] = [0.70, 0.15, 0.15]) -> None:
        super().__init__(
            img_wh=self.img_wh,
            img_wh_net=self.img_wh_net,
            mean=self.mean,
            std=self.std,
            split=split,
        )

    def set_logger(self, logger):
        self.logger = logger

    @override
    def download_dataset(self) -> bool:
        """Download the MNIST-M data."""
        if (
            os.path.exists(self.dataset_path)
            and os.path.exists(self.dataset_extract_path)
            and os.path.exists(self.dataset_download_path)
        ):
            return False

        os.makedirs(self.dataset_path, exist_ok=True)
        os.makedirs(self.dataset_path + "/download", exist_ok=True)
        os.makedirs(self.dataset_path + "/extract", exist_ok=True)
        # os.makedirs(self.processed_folder, exist_ok=True)

        # download files
        for url, md5 in self.resources:
            filename = url.rpartition("/")[2]
            download_and_extract_archive(
                url,
                download_root=self.dataset_path + "/download",
                extract_root=self.dataset_path + "/extract",
                filename=filename,
                md5=md5,
            )
        return True

    @override
    def load_dataset(self) -> Tuple[Tensor, Tensor]:
        train = torch.load(self.dataset_path + "/extract/" + self.training_file)
        test = torch.load(self.dataset_path + "/extract/" + self.test_file)
        all_data = torch.cat((train[0], test[0]))
        all_labels = torch.cat((train[1], test[1]))
        return all_data, all_labels


if __name__ == "__main__":
    c = MNISTM10()
    c.compute_mean_std()
