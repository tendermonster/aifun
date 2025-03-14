from __future__ import annotations
import random
import typing
import PIL.Image as PILImage
import numpy as np
from typing import Union
from cv2.typing import MatLike
import cv2
from pathlib import Path

if typing.TYPE_CHECKING:
    from PIL.Image import Image


def implant_background(
    input_image_path: Path,
    template_image_path: Union[Path, MatLike],
    output_image_path: Path,
):
    # Load images
    image = cv2.imread(str(input_image_path), cv2.IMREAD_COLOR)

    if isinstance(template_image_path, Path):
        template = template = cv2.imread(
            str(template_image_path), cv2.IMREAD_COLOR
        )  # Assuming the template is a color image
    else:
        template: MatLike = template_image_path
    if image.shape != template.shape:
        print("Image and template should have the same dimensions!")
        return

    # Convert the image to grayscale for thresholding
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Since the background is darker than the object for MNIST
    _, thresholded = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # Separate background and object pixels
    object_pixels = np.where(thresholded == 255)
    background_pixels = np.where(thresholded == 0)

    # Create an output image, initialize as a copy of the original image
    output_image = image.copy()

    # Set background pixels to the template image pixels
    for y, x in zip(*background_pixels):
        output_image[y, x] = template[y, x]

    # Set object pixels to inverted colors of the corresponding template image
    for y, x in zip(*object_pixels):
        output_image[y, x] = [255 - val for val in template[y, x]]

    # Save the processed image
    output_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
    cv2.imwrite(str(output_image_path), output_image)


def get_random_patch(image: Path) -> Image:
    # get random patch of 300x300 from the image
    # image must be >= 300x300
    pillow_image = PILImage.open(image)
    if pillow_image.size[0] < 300 or pillow_image.size[1] < 300:
        raise ValueError("image must be >= 300x300")

    # Get the dimensions of the image
    img_width, img_height = pillow_image.size

    # Define the size of the patch you want to extract
    patch_width = 300
    patch_height = 300

    # Generate random x and y coordinates for the top-left corner of the patch
    x = random.randint(0, img_width - patch_width)
    y = random.randint(0, img_height - patch_height)

    # Crop the randomly selected patch from the Pillow image
    patch_pillow = pillow_image.crop((x, y, x + patch_width, y + patch_height))

    # Convert the Pillow image to a NumPy array (OpenCV format)
    patch_opencv = np.array(patch_pillow)

    # Convert BGR to RGB (if necessary)
    # if patch_opencv.shape[-1] == 3:
    #     patch_opencv = cv2.cvtColor(patch_opencv, cv2.COLOR_BGR2RGB)

    # plt.imshow(patch_opencv)
    # plt.show()

    patch_to_PIL = PILImage.fromarray(patch_opencv)
    return patch_to_PIL
