import os
import sys

import imageio
import imgaug.augmenters as iaa
from keras.src.utils import load_img, img_to_array
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def augment_image(img, n=15):
    """Slightly augment the image to artificially generate more training data."""
    # Ensure image has 3 channels (in case it's grayscale or has alpha)
    if img.ndim == 2:
        image = np.stack([img]*3, axis=-1)
    elif img.shape[2] == 4:  # RGBA -> RGB
        image = img[:, :, :3]
    else:
        image = img
    image = image.astype(np.uint8)

    # Define a set of augmenters
    augmenters = iaa.Sequential([
        iaa.Affine(rotate=(-20, 20), scale=(0.9, 1.1)),     # Rotate & scale
        iaa.AdditiveGaussianNoise(scale=(0, 0.05*255)),     # Gaussian noise
        iaa.GaussianBlur(sigma=(0, 1.5)),                   # Blur
        iaa.LinearContrast((0.5, 1.6)),                     # Adjust contrast
        iaa.Multiply((0.9, 1.1), per_channel=0.3),          # Adjust brightness
        iaa.Fliplr(0.5),                                    # Horizontal flip
        iaa.Add((-5, 5), per_channel=0.3)                 # Color shift
    ])

    # Generate n augmentations
    augmented_images = augmenters(images=[image for _ in range(n)])

    return augmented_images


if __name__ == '__main__':
    for filename in sys.argv[1:]:
        image = img_to_array(load_img(filename))
        augmented_images = augment_image(image)
        for i, aug_img in enumerate(augmented_images):
            Image.fromarray(augmented_images[i]).save(f'aug_{i}_{filename}')
    
