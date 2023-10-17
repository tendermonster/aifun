from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from utils.dataset.utils import get_random_patch, implant_background
from pathlib import Path
import shutil
import glob
import tqdm

# this is a dataset generator for object detection with intent of domain adaptation
# in Domain-Adversarial Neural Networks training paper they used BSDS500 to overlay the background
# the number are just the inverted colors of the background at that particular pixel


if __name__ == "__main__":
    # Prerequisite
    # if not yet done download and unpack the dataset
    # 1.1. get mnist dataset # a: Dataset = MNIST10()
    # 1.2. get bsd500 dataset # c: Dataset = MNIST10()

    # get paths for all images

    bsd_test = glob.glob("dataset/bsd500/images/test/*.jpg")
    bsd_train = glob.glob("dataset/bsd500/images/train/*.jpg")
    bsd_val = glob.glob("dataset/bsd500/images/val/*.jpg")
    bsd_all = bsd_test + bsd_train + bsd_val

    mnist_od_all = glob.glob("dataset/mnist_od/images/*")
    mnist_od_labels = glob.glob("dataset/mnist_od/labels/*")

    mod_images = Path("dataset/mnistm_od/images")
    mod_labels = Path("dataset/mnistm_od/labels")
    if not mod_images.exists():
        mod_images.mkdir()
    if not mod_labels.exists():
        mod_labels.mkdir()
    for i in tqdm.trange(
        len(mnist_od_all), desc="Generating dataset, saving to: dataset/mnistm_od/"
    ):
        m = mnist_od_all[i]
        m_in = Path(m)
        m_out = mod_images.joinpath(m_in.name)
        # get random background
        template = np.random.choice(bsd_all)
        template = Path(template)
        # get random patch
        patch_template = get_random_patch(template)
        patch_template = np.array(patch_template)
        # show patch
        # plt.imshow(patch_template)
        # plt.show()
        implant_background(m_in, patch_template, m_out)

        # copy label
        m_label_in = Path(mnist_od_labels[i])
        m_label_out = mod_labels.joinpath(m_label_in.name)
        shutil.copy(m_label_in, m_label_out)
