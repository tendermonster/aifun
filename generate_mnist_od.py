from utils.dataset.mnist import MNIST10
from utils.dataset.dataset import Dataset
from utils.dataset.generate_mnist_od import generate_dataset
from pathlib import Path


if __name__ == "__main__":
    c: Dataset = MNIST10()
    all_data = c.get_dataset()
    gen_path = Path("dataset/mnist_od")
    img_size = 300
    max_digit_size = 60
    min_digit_size = 20
    # just as in mnist dataset
    num_trian_images = 8000
    num_test_images = 2000
    max_digit_per_image = 20
    num_images = num_trian_images + num_test_images
    # num_images = 10

    generate_dataset(
        gen_path,
        num_images,
        max_digit_size,
        min_digit_size,
        img_size,
        max_digit_per_image,
        all_data,
    )
