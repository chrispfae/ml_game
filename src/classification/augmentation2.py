import os
import sys

import imageio
import imgaug.augmenters as iaa
from keras.src.utils import load_img, img_to_array
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

if not hasattr(np, 'bool'):
    np.bool = np.bool_


def augment_image(img, n=15):
    """Slightly augment the image to artificially generate more training data."""
    # Example input: img (can be grayscale, RGB, or RGBA)
    # Preprocessing has no effect if it is a rgba image (except for type change)
    if img.ndim == 2:
        # Grayscale -> RGB
        image_rgb = np.stack([img] * 3, axis=-1)
        alpha = np.full(img.shape, 255, dtype=np.uint8)  # opaque
    elif img.shape[2] == 4:
        # RGBA
        image_rgb = img[:, :, :3]
        alpha = img[:, :, 3]
    else:
        # RGB
        image_rgb = img
        alpha = np.full(img.shape[:2], 255, dtype=np.uint8)  # opaque
    
    # Ensure data type
    image_rgb = image_rgb.astype(np.uint8)
    alpha = alpha.astype(np.uint8)

    # Stack RGB + alpha into 4-channel image for geometric sync
    image_rgba_stack = np.concatenate([image_rgb, alpha[..., None]], axis=-1)
    
    # Define a set of augmenters
    # Generate n augmentations
    augmenters = iaa.Sequential([
        iaa.Affine(rotate=(-20, 20), scale=(0.9, 1.1)),     # Rotate & scale
        iaa.Fliplr(0.5),                                    # Horizontal flip (should affect alpha too)
        iaa.SomeOf((1, 4), [                                # Apply some photometric transforms
            iaa.AdditiveGaussianNoise(scale=(0, 0.05 * 255)),
            iaa.GaussianBlur(sigma=(0, 1.5)),
            iaa.LinearContrast((0.5, 1.6)),
            iaa.Multiply((0.9, 1.1), per_channel=0.3),
            iaa.Add((-5, 5), per_channel=0.3)
        ])
    ])
    augmented_images = augmenters(images=[image_rgba_stack for _ in range(n)])

    return augmented_images


if __name__ == '__main__':
    for filename in sys.argv[1:]:
        image = img_to_array(load_img(filename, color_mode='rgba'))
        augmented_images = augment_image(image, 15)
        for i, aug_img in enumerate(augmented_images):
            Image.fromarray(augmented_images[i]).save(f'aug_{i}_{filename}')
    
