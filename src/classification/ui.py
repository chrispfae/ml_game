from ipywidgets import Image, Layout, Label, VBox, HTML, Button, HBox, Checkbox, GridBox, IntProgress
from keras.src.callbacks import Callback
from keras import Model
from keras.src.ops import sigmoid

from classification.backend import ImageData, load_image_for_prediction
from logging import getLogger

from utils.colors import colors, color_to_string

# Initialize a logger for the module
logger = getLogger(__name__)


class CustomWidget:
    widget = None

    def __init__(self):
        """
        Base class for all custom widgets.

        This class contains an attribute `widget`, which represents the `ipywidgets` widget.
        """
        pass


class CustomImageWidget(CustomWidget):
    def __init__(self, img_data: ImageData):
        """
        Base class for all image widgets.

        Args:
            img_data (ImageData): Image data object.
        """
        self._img_data = img_data
        super().__init__()


class CustomImageListWidget(CustomWidget):
    def __init__(self, img_data: list[ImageData]):
        """
        Base class for all widgets displaying a list of images.

        Args:
            img_data (list[ImageData]): List of image data objects.
        """
        self._img_data = img_data
        super().__init__()


class CustomButtonWidget(CustomWidget):
    def __init__(self):
        """
        Base class for all button widgets.
        """
        super().__init__()


class ImageBox(CustomImageWidget):
    def __init__(self, img_data: ImageData):
        """
        A widget that displays an image. The image is loaded from the file system and scaled to a height of 150 pixels.

        Args:
            img_data (ImageData): Image data object.
        """
        super().__init__(img_data)
        self.widget = Image(value=open(img_data.img_path, "rb").read(), layout=Layout(height="150px", width="112px"))


class LabeledImageBox(ImageBox):
    def __init__(self, img_data: ImageData):
        """
        A widget that displays an image along with the robot's name and job. The image is loaded from the file system and scaled to a height of 150 pixels.

        Args:
             img_data (ImageData): Image data object.
        """
        super().__init__(img_data)
        name_label = Label(f"Name: {img_data.name}")
        job_label = Label(f"Job: {img_data.job}")
        self.widget = VBox([self.widget, name_label, job_label],
                           layout=Layout(width="150px", min_width="150px", height="fit-content", padding="5px", border="1px solid lightblue", align_items="center"))


class ImageButtonBox(CustomImageWidget, CustomButtonWidget):
    def __init__(self, img_data: ImageData):
        """
        A widget that displays an image and two buttons for classifying the image as dangerous or harmless. The image is loaded from the file system and scaled to a height of 150 pixels.

        Args:
            img_data (ImageData): Image data object.
        """
        super().__init__(img_data)
        self._img = ImageBox(img_data=img_data)
        self._result = None
        self._solution = not img_data.is_dangerous
        self._danger_button = Button(description="Gefährlich", layout=Layout(width="fit-content"))
        self._safe_button = Button(description="Harmlos", layout=Layout(width="fit-content"))
        self.widget = VBox([self._img.widget, HBox([self._safe_button, self._danger_button])])

    def on_safe_click(self, callback):
        """
        Sets the callback for the "Harmless" button click.

        Args:
            callback: Function to be called when the button is clicked.
        """
        self._safe_button.on_click(callback)

    def on_danger_click(self, callback):
        """
        Sets the callback for the "Dangerous" button click.

        Args:
            callback: Callback function.
        """
        self._danger_button.on_click(callback)

    def is_right(self) -> bool:
        """
        Checks if the result is correct.

        Returns:
            bool: True if the result is correct, otherwise False.

        Raises:
            ValueError: If the result is not set.
        """
        if self._result is None:
            raise ValueError("Result is not set")
        return self._result == self._solution

    def get_safe_button_color(self):
        """Return color of safe button."""
        return self._safe_button.style.button_color

    def get_danger_button_color(self):
        """Return color of danger button."""
        return self._danger_button.style.button_color

    def set_safe_button_layout(self, style: str, layout: Layout = Layout()):
        """
        Sets the style and layout of the "Harmless" button.

        Args:
            style (str): Style of the button.
            layout (Layout): Layout parameters.
        """
        self._safe_button.layout = _update_layout(self._safe_button.layout, layout)
        if style == '':
            self._safe_button.style.button_color = None
        else:
            self._safe_button.style.button_color = style

    def set_danger_button_layout(self, style: str, layout: Layout = Layout()):
        """
        Sets the style and layout of the "Dangerous" button.

        Args:
            style (str): Style of the button.
            layout (Layout): Layout parameters.
        """
        self._danger_button.layout = _update_layout(self._danger_button.layout, layout)
        if style == '':
            self._danger_button.style.button_color = None
        else:
            self._danger_button.style.button_color = style

    def disable_buttons(self, disable_safe: bool = False, disable_danger: bool = False):
        """
        Disables the "Harmless" and "Dangerous" buttons.

        Args:
            disable_safe (bool): Disables the safe button.
            disable_danger (bool): Disables the dangerous button.
        """
        self._safe_button.disabled = disable_safe
        self._danger_button.disabled = disable_danger

    @property
    def result(self):
        """
        Returns the result.

        Returns:
            bool: Result.
        """
        return self._result

    @result.setter
    def result(self, value: bool):
        """
        Sets the result.

        Args:
            value (bool): Result.
        """
        self._result = value

    @property
    def solution(self):
        """
        Returns the solution.

        Returns:
            bool: Solution.
        """
        return self._solution


class LabeledImageButtonBox(ImageButtonBox):
    def __init__(self, img_data: ImageData):
        """
        A widget that displays an image along with the robot's name and job.
        Additionally, it provides two buttons for classifying the image as dangerous or harmless.
        The image is loaded from the file system and scaled to a height of 150 pixels.

        Args:
            img_data (ImageData): Image data object.
        """
        super().__init__(img_data)
        self._img = LabeledImageBox(img_data=img_data)
        self.widget = VBox([self._img.widget, HBox([self._safe_button, self._danger_button])], layout=Layout(width="fit-content", align_items="center"))


class TestBox(ImageBox):
    def __init__(self, img: ImageData, img_data, model: Model):
        """
        A widget that displays an image and provides a button to test the model.

        Args:
            img (ImageData): Image data object.
            img_data: The preprocessed image.
            model (Model): Keras model used for predictions.
        """
        super().__init__(img)
        self.model = model
        self.img_data_prepro = img_data
        self.result = HTML("", layout=Layout(width="99%"))
        self.solution = img.is_dangerous
        self.test_button = Button(description="Testen", layout=Layout(width="99%"), style={"color": "darkolivegreen"})
        self.test_button.on_click(self._on_button_click)
        self.widget = VBox([self.widget, self.test_button, self.result], layout=Layout(width="155px", height="fit-content", align_items="center", margin="1px"))

    def _on_button_click(self, b):
        """
        Callback function triggered when the test button is clicked.
        It loads the image, performs a prediction using the model, and updates the result display.

        Args:
            b: The button instance that triggered the callback
        """
        self.test_button.style.button_color = color_to_string(colors['grey'])
        #img_array = load_image_for_prediction(self._img.img_path)

        prediction = self.model.predict(self.img_data_prepro[0])
        print(prediction)
        #score = float(sigmoid(prediction[0]))
        score = prediction[0, 0]
        result = "harmlos" if score < 0.5 else "gefährlich"
        certainty = (1 - score if score < 0.5 else score)
        self.result.value = f"<p>Der Roboter ist <i>{result}</i> mit einer Gewissheit von {certainty:.2%}</p><p>Tatsächliche Kategorie: {'harmlos' if not self.solution else 'gefährlich'}</p>"
        self.result.style.background = color_to_string(colors['red']) if self.solution != (score >= 0.5) else color_to_string(colors['green']) 


class ImageCategory(CustomImageListWidget, CustomButtonWidget):
    def __init__(self, name, images: list[ImageData]):
        """
        A widget that displays a list of images and provides two buttons for classifying the images as dangerous or harmless.

        Args:
            name (str): Name of the category.
            images (list[ImageData]): List of image data objects
        """
        super().__init__(images)
        self.name = name
        self.save = None
        box = VBox(layout=Layout(width="fit-content", height="fit-content", border="2px solid transparent"))
        self._danger_button = Button(description="Gefährlich", layout=Layout(width="auto"), button_style="")
        self._safe_button = Button(description="Harmlos", layout=Layout(width="auto"), button_style="")
        box.children = [HBox([HTML("<b>" + self.name + "</b>"), HBox([self._safe_button, self._danger_button])], layout=Layout(justify_content="space-between")),
                        HBox([LabeledImageBox(img).widget for img in self._img_data])]
        self.widget = box

    def on_safe_click(self, callback):
        """
        Sets the callback for the "Harmless" button click.

        Args:
            callback: Callback function to be executed when the button is clicked
        """
        self._safe_button.on_click(callback)

    def on_danger_click(self, callback):
        """
        Sets the callback for the "Dangerous" button click.

        Args:
            callback: Callback function to be executed when the button is clicked.
        """
        self._danger_button.on_click(callback)

    def set_layout(self, safe_button_layout: Layout = Layout(), danger_button_layout: Layout = Layout(), box_layout: Layout = Layout()):
        """
        Sets the layout for the "Harmless" and "Dangerous" buttons, as well as the entire widget.

        Args:
            safe_button_layout (Layout): Layout for the "Harmless" button.
            danger_button_layout (Layout): Layout for the "Dangerous" button.
            box_layout (Layout): Layout for the widget box.
        """
        self._safe_button.layout = _update_layout(self._safe_button.layout, safe_button_layout)
        self._danger_button.layout = _update_layout(self._danger_button.layout, danger_button_layout)
        self.widget.layout = _update_layout(self.widget.layout, box_layout)

    def set_button_style(self, safe_button_style: str | None = None, danger_button_style: str | None = None):
        """
        Sets the style for the "Harmless" and "Dangerous" buttons.

        Args:
            safe_button_style (str|None): Style for the "Harmless" button.
            danger_button_style (str|None): Style for the "Dangerous" button.
        """
        if safe_button_style is not None:
            self._safe_button.style.button_color = safe_button_style
        if danger_button_style is not None:
            self._danger_button.style.button_color = danger_button_style

    def disable_buttons(self, disable_safe: bool = False, disable_danger: bool = False):
        """
        Disables the "Harmless" and "Dangerous" buttons.

        Args:
            disable_safe (bool): Whether to disable the "Harmless" button.
            disable_danger (bool): Whether to disable the "Dangerous" button.
        """
        self._safe_button.disabled = disable_safe
        self._danger_button.disabled = disable_danger


class SelectableImage(ImageBox):
    def __init__(self, img_data: ImageData, callback):
        """
        A widget that displays an image and a checkbox.
        The checkbox can be used to select or deselect the image.

        Args:
            img_data (ImageData): Image data object.
            callback: Callback function triggered when the checkbox value changes.
        """
        super().__init__(img_data)
        self.checkbox = Checkbox(value=False, indent=False, layout=Layout(width="fit-content"))
        self.checkbox.observe(lambda c: callback(c.new, self._img_data), names="value")
        self.widget = VBox([self.checkbox, self.widget], layout=Layout(width="fit-content", align_items="center", margin="0 2px"))


class SelectionBox(CustomImageListWidget):
    def __init__(self, image_data: list[ImageData]):
        """
        A widget that displays a list of images, each with a checkbox for selection.

        Args:
            image_data (list[ImageData]): List of image data objects.
        """
        super().__init__(img_data=image_data)
        self.selected: list[ImageData] = []
        self.images = [SelectableImage(data, self.on_check_img) for data in image_data]
        self.widget = GridBox([img.widget for img in self.images], layout=Layout(grid_template_columns="repeat(5, 1fr)", margin="10px", padding="5px"))

    def on_check_img(self, change, img_data: ImageData):
        """
        Callback function for selecting an image.

        Args:
            change: Change in the checkbox value.
            img_data: Image data object associated with the checkbox.
        """
        if change:
            self.selected.append(img_data)
        else:
            self.selected.remove(img_data)


class TrainingCallback(Callback):
    def __init__(self, progress: IntProgress, result_text: HTML):
        """
        A callback that displays the training progress.

        Args:
            progress (IntProgress): Progress widget.
            result_text (HTML): Result text widget.
        """
        super().__init__()
        self.progress = progress
        self.result_text = result_text

    def on_train_batch_begin(self, batch, logs=None):
        """
        Callback function triggered at the beginning of a training batch.

        Args:
            batch: Batch number.
            logs: Log data.
        """
        self._update_widgets(batch, logs)

    def on_train_batch_end(self, batch, logs=None):
        """
        Callback function triggered at the end of a training batch.

        Args:
            batch: Batch number.
            logs: Log data.
        """
        self._update_widgets(batch, logs)

    def on_test_batch_end(self, batch, logs=None):
        """
        Callback function triggered at the end of a test batch.

        Args:
            batch: Batch number.
            logs: Log data.
        """
        self._update_widgets(batch, logs)

    def _update_widgets(self, batch, logs=None):
        """
        Updates the progress and result text widgets.

        Args:
            batch: Batch number.
            logs: Log data.
        """
        print(f'Batch:{batch}')
        print(f'log: {logs}')
        self.progress.value = batch + 1
        if logs is not None and len(logs) > 0:
            self.result_text.value = f"Batch: {batch + 1}; Richtig vorhergesagte Bilder: {logs['binary_accuracy']:.2%}"
        else:
            self.result_text.value = f'Batch: {batch + 1}'


def _update_layout(old_layout: Layout, new_layout: Layout) -> Layout:
    """
    Updates the layout of a widget.

    Args:
        old_layout (Layout): Existing layout.
        new_layout (Layout): New layout to be applied.

    Returns:
        Layout: Updated layout.
    """
    old_layout_values = {key: value for key, value in vars(old_layout)["_trait_values"].items() if key not in ("keys", "comm") and not key.startswith("_")}
    new_layout_values = {key: value for key, value in vars(new_layout)["_trait_values"].items() if key not in ("keys", "comm") and not key.startswith("_")}
    for key, value in new_layout_values.items():
        if value is not None:
            old_layout_values[key] = value
    return Layout(**old_layout_values)
