from __future__ import annotations
from typing import Tuple
from typing_extensions import override
from torch import Tensor
import requests
import tarfile
import pathlib
import shutil
import os
from torchvision import transforms
import torch

# import cv2
import numpy as np
import random
import typing
import PIL.Image as PILImage

# import matplotlib.pyplot as plt
from utils.dataset.dataset import Dataset
from utils.dataset.utils import get_random_patch

if typing.TYPE_CHECKING:
    from PIL.Image import Image
    from pathlib import Path


class BSD500(Dataset):
    resource = "http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz"
    tgz_path = "/tmp"
    dataset_path = "dataset/bsd500"
    img_wh = 300
    img_wh_net = 224

    def __init__(self):
        super().__init__(
            img_wh=300,
            img_wh_net=self.img_wh_net,
            mean=[0.4448, 0.4430, 0.3674],
            std=[0.2153, 0.2018, 0.1913],
            split=[0.70, 0.15, 0.15],
        )

    @override
    def load_dataset(self) -> Tuple[Tensor, Tensor]:
        # output is Tensor: image data Tensor: label
        test = pathlib.Path("dataset/bsd500/images/test")
        train = pathlib.Path("dataset/bsd500/images/train")
        val = pathlib.Path("dataset/bsd500/images/val")

        test_images = list(test.glob("*.jpg"))
        train_images = list(train.glob("*.jpg"))
        val_images = list(val.glob("*.jpg"))

        all_images = test_images + train_images + val_images
        aug = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        all_as_tensors = torch.Tensor()
        all_labels = torch.Tensor()
        for i in all_images:
            patch_PIL: Image = get_random_patch(i)
            patch: Tensor = aug(patch_PIL)
            all_as_tensors = torch.cat((all_as_tensors, patch))
            all_labels = torch.cat((all_labels, torch.Tensor([-1])))
        return all_as_tensors, all_labels

    @override
    def download_dataset(self) -> bool:
        # download dataset
        try:
            local_filename: str = self.resource.split("/")[-1]
            full_path = pathlib.Path(os.path.join(self.tgz_path, local_filename))
            if not full_path.exists():
                # NOTE the stream=True parameter below
                with requests.get(self.resource, stream=True) as r:
                    r.raise_for_status()
                    with open(full_path, "wb") as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            # If you have chunk encoded response uncomment if
                            # and set chunk_size parameter to None.
                            # if chunk:
                            f.write(chunk)
            # extract dataset

            tgz_path = pathlib.Path("/tmp/BSR_bsds500.tgz")
            tar_file = tarfile.open(tgz_path)
            extract_path = pathlib.Path("/tmp/BSR")
            tar_file.extractall("/tmp")
            src = pathlib.Path("/tmp/BSR/BSDS500/data/images")
            dst = pathlib.Path("dataset/bsd500/images")
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src=src, dst=dst)
            # os.remove(pathlib.Path(tgz_path))
            shutil.rmtree(pathlib.Path(extract_path))
            return True
        except Exception:
            return False
