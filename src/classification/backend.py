from logging import getLogger
from os import listdir
from os.path import join, isfile, abspath

from PIL import Image as PILImage
from keras import Model
from keras.applications import MobileNetV3Small
from keras.layers import GlobalAveragePooling2D, Dense, Input, Concatenate
from keras.optimizers import Adam
from keras.src.backend import image_data_format
from keras.src.losses import BinaryCrossentropy
from keras.src.metrics import BinaryAccuracy
from keras.src.ops.numpy import expand_dims
from keras.src.optimizers import Adam
from keras.src.saving import load_model as load_keras_model
from keras.src.utils import load_img, img_to_array
from keras.src.utils.image_dataset_utils import paths_and_labels_to_dataset
from numpy import ndarray
import numpy as np
from tensorflow.python.data import AUTOTUNE, Dataset

logger = getLogger(__name__)


class ImageData:
    def __init__(self, img_path: str):
        """
        Initializes an ImageData object with the path to the image file and extracts metadata.
        The metadata includes the name, job, and whether the image is dangerous or clean.
        The metadata is extracted from the image file's info dictionary.

        Args:
            img_path (str): the path to the image file.
        """
        names = ['Sentra', 'Vita', 'Halo', 'C.A.R.E', 'Ava', 'Elan', 'Milo', 'Sam', 'Eir', 'Galatea', 'Aether', 'Soma']
        self.img_path: str = img_path
        with PILImage.open(img_path) as img:
            self.name: str = names[np.random.randint(0, len(names))]
            if 'cute' in img_path:
                self.job: str = 'Kinderbetreuung'
                self.is_clean: bool = True
                self.is_dangerous: bool = False 
            elif 'gef' in img_path:
                self.job: str = 'Aufseher'
                self.is_clean: bool = False
                self.is_dangerous: bool = True
            else:
                self.job: str = 'Assistent'
                self.is_clean: bool = True
                self.is_dangerous: bool = False
                

def load_images_from_path(img_path: str) -> list[ImageData]:
    """
    Loads images from a specified directory and returns a list of ImageData objects.
    Each ImageData object contains the path to the image file and its metadata.

    Args:
        img_path (str): the path to the directory containing the images.

    Returns:
        list[ImageData]: a list of ImageData objects, each representing an image in the directory.
    """
    image_data = [ImageData(img_path=join(img_path, image)) for image in listdir(img_path) if isfile(join(img_path, image))]
    return image_data


def load_image_for_prediction(img_path: str) -> ndarray:
    """
    Loads an image from a specified path and prepares it for prediction.
    The image is resized to 512x384 pixels and converted to a Numpy array, ready for input into a Keras model.

    Args:
        img_path (str): the path to the image file.

    Returns:
        ndarray: the image as a Numpy array, ready for prediction.
    """
    #img = load_img(img_path, target_size=(512, 384))
    #img_array = img_to_array(img)
    #img_array = expand_dims(img_array, axis=0)
    #return img_array
    image_data = [ImageData(img_path=img_path)]
    return image_data


def load_model(model_path: str) -> Model:
    """
    Loads a Keras model from a specified path.
    
    Args:
        model_path (str): the path to the Keras model file.

    Returns:
        Model: the loaded Keras model, ready for use in predictions.
    """
    model = load_keras_model(abspath(model_path), compile=False)
    model.trainable = False
    model.compile(Adam(5e-6), loss=BinaryCrossentropy(from_logits=True), metrics=[BinaryAccuracy()])
    return model


def load_model_mobilenet():
    # Load base model without top classification layer
    base_model = MobileNetV3Small(
        input_shape=(512, 512, 3),
        include_top=False,
        weights='imagenet'
    )
    # Freeze base model initially
    base_model.trainable = False
    
    # Add custom classification head
    inputs = Input(shape=(512, 512, 3))
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    outputs = Dense(1, activation='sigmoid')(x)  # Binary classification
    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['binary_accuracy'])
    return model

def load_model_mobilenet_with_logo():
        
    # Base CNN model
    base_model = MobileNetV3Small(
        input_shape=(512, 512, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False

    # Image input
    image_input = Input(shape=(512, 512, 3), name='image_input')
    # Single extra feature input (e.g., a scalar)
    logo_input = Input(shape=(1,), name='logo_input')

    # Add custom classification head 
    x = base_model(image_input, training=False)
    x = GlobalAveragePooling2D()(x)
    # Combine CNN features with extra scalar input
    x = Concatenate()([x, single_input])
    # Classification head
    x = Dense(128, activation='relu')(x)
    output = Dense(1, activation='sigmoid')(x)
    # Final model
    model = Model(inputs=[image_input, single_input], outputs=output)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['binary_accuracy'])
    return model


def get_possible_models(models_path: str) -> dict[str, str]:
    """
    Retrieves a list of possible Keras models from a specified directory.

    Args:
        models_path (str): the path to the directory containing the Keras model files.
    """
    models = listdir(models_path)
    models = [model for model in models if model.endswith(".keras")]
    models = {model.title().strip(".keras").split("_")[0]: join(models_path, model) for model in models}
    return models


def get_test_data(safe_images: list[ImageData], danger_images: list[ImageData], batch=True) -> Dataset:
    """
    Creates test datasets from safe and dangerous images.

    Args:
        safe_images (list[ImageData]): List of safe images.
        danger_images (list[ImageData]): List of dangerous images.

    Returns:
        Dataset: Dataset containing the images and labels.
    """
    images, labels = [], []
    for label, image_set in {0.: safe_images, 1.: danger_images}.items():
        for img in image_set:
            images.append(img.img_path)
            labels.append(label)
    dataset = paths_and_labels_to_dataset(
        images,
        image_size=(512, 512),
        num_channels=3,
        labels=labels,
        label_mode="binary",
        num_classes=2,
        interpolation="bilinear",
        shuffle=True,
        shuffle_buffer_size=50,
        data_format=image_data_format()
    )
    logo = [1 for img in images if 'gef' in img else 0]
    
    #import matplotlib.pyplot as plt
    ## Take one batch (or one element) from the dataset
    #i = 0
    #for image, label in dataset.take(10):
    #    # If the image is a tensor, convert it to numpy
    #    image = image.numpy()
    #    label = label.numpy()

    #    # Optional: If channels are first (e.g., (3, 512, 384)), transpose it
    #    if image.shape[0] == 3 and len(image.shape) == 3:
    #        image = image.transpose(1, 2, 0)

    #    # Normalize pixel values to [0, 1] if necessary
    #    if image.max() > 1:
    #        image = image / 255.0

    #    plt.imshow(image)
    #    plt.title(f"Label: {label}")
    #    plt.axis('off')
    #    plt.savefig(f'/home/chris/Downloads/out/{i}.png')
    #    i += 1

    if batch:
        dataset = dataset.batch(4)
    else:
        dataset = dataset.batch(1)
    dataset = dataset.prefetch(AUTOTUNE)
   
    return dataset

    



