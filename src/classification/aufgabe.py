from logging import getLogger
from math import ceil
from os import getcwd
from os.path import join, abspath
from random import choices
from time import sleep

from IPython.core.display_functions import display, DisplayHandle
from ipywidgets import Button, HTML, VBox, HBox, Layout, GridspecLayout, IntProgress, Dropdown, Tab, Widget
from jupyterlab.browser_check import test_flags
from keras.src.ops import sigmoid

from classification.backend import load_images_from_path, load_model_mobilenet, get_test_data, get_possible_models, load_image_for_prediction, ImageData
from classification.ui import LabeledImageButtonBox, ImageButtonBox, ImageCategory, TestBox, TrainingCallback, LabeledImageBox, SelectionBox, ImageBox
from utils.colors import colors, color_to_string


logger = getLogger(__name__)


class Aufgabe:
    """
    Class containing the main logic for the task.
    The methods represent the different parts of the task.

    Attributes:
        test_images (list[ImageData]): List containing the data for the test images.
        images (list[ImageData]): List containing the data for the normal images.
        unknown_images (list[ImageData]): List containing the data for the unknown images.
        categorized_images (dict[str, list[ImageData]]): Dictionary containing the categorized images.
        model (Model): The model used for predictions.
        sorted_images (dict[str, list[ImageData]]): Dictionary containing the sorted images.
    """

    def __init__(self):
        """
        Initializes the Aufgabe class and loads the images from the specified paths.
        """
        test_folder_path = join(getcwd(), "images/classification", "validation")
        self.validation_images = load_images_from_path(test_folder_path)

        image_folder = join(getcwd(), "images/classification", "training")
        self.images = load_images_from_path(image_folder)

        unknown_folder_path = join(getcwd(), "images/classification", "testing_doors")
        self.unknown_images = load_images_from_path(unknown_folder_path)

        self.sorted_images = None

        self.categorized_images = {}
        for image in self.images:
            if image.job in self.categorized_images:
                self.categorized_images[image.job].append(image)
            else:
                self.categorized_images[image.job] = [image]
        self.model = None

    def part1(self):
        """
        Initializes the UI for the first part of the task.
        The user has to classify the images as "safe" or "dangerous" by clicking the corresponding buttons.
        The images are displayed with all information about them.
        """
        submit_button = Button(description="Bestätigen", disabled=True)
        k = 6
        example_images = []
        counter_cute = 0
        counter_assistent = 0
        counter_aufseher = 0
        discarded = 0
        while counter_cute + counter_assistent + counter_aufseher < k:
            i = counter_cute + counter_assistent + counter_aufseher + discarded
            candidate_image = self.images[i]
            if candidate_image.job == 'Kinderbetreuung':
                if counter_cute == 2:
                    discarded += 1
                    continue
                else:
                    counter_cute += 1
                    example_images.append(candidate_image)
            elif candidate_image.job == 'Assistent':
                if counter_assistent == 2:
                    discarded += 1
                    continue
                else:
                    counter_assistent += 1
                    example_images.append(candidate_image)
            else:
                if counter_aufseher == 2:
                    discarded += 1
                    continue
                else:
                    counter_aufseher += 1
                    example_images.append(candidate_image)
        test_img_widgets = list(map(lambda data: LabeledImageButtonBox(data), example_images))
        instructions = HTML(
            '<p><b>Beep:</b> "Genau! Jetzt können wir das noch für ein paar weitere machen. Versucht mal für jeden zu entscheiden, ob er gefährlich oder harmlos ist."</p>')

        display_box = VBox([instructions, HBox([x.widget for x in test_img_widgets]), submit_button])

        def on_submit_click(b):
            right = [x.is_right() for x in test_img_widgets]
            for button, res in zip(test_img_widgets, right):
                if res:
                    color = color_to_string(colors['green'])
                else:
                    color = color_to_string(colors['red'])
                if button.get_safe_button_color() is not None:
                    button.set_safe_button_layout(color)
                elif button.get_danger_button_color() is not None:
                    button.set_danger_button_layout(color)
            result_text = HTML(f'<p><b>Beep:</b> "Ich habe das mal überprüft: Ihr habt {sum(right)} der {len(test_img_widgets)} Roboter richtig eingeordnet! Die Richtigen sind grün markiert und die Falschen rot."</p>'
                               f'<p><b>Ben:</b> "Kannst du uns vielleicht helfen die Roboter in allen Akten entweder als harmlos oder gefährlich zu gruppieren, Beep? Das sind so viele und du kannst das doch so schnell."</p>'
                               f'<p><b>Beep:</b> "Natürlich, ich mache das kurz für euch."</p>')
            display_box.children += (result_text,)
            submit_button.disabled = True
            #for box in test_img_widgets:
            #    box.disable_buttons(True, True)

        def update_submit_disable():
            submit_button.disabled = not all([v.result is not None for v in test_img_widgets])

        def on_safe_button_click_generator(button_box: ImageButtonBox):
            def on_safe_button_click(b):
                button_box.result = True
                button_box.set_safe_button_layout(style=color_to_string(colors['grey']))
                button_box.set_danger_button_layout(style="")
                update_submit_disable()

            return on_safe_button_click

        def on_danger_button_click_generator(button_box: ImageButtonBox):
            def on_danger_button_click(b):
                button_box.result = False
                button_box.set_danger_button_layout(style=color_to_string(colors['grey']))
                button_box.set_safe_button_layout(style="")
                update_submit_disable()

            return on_danger_button_click

        for box in test_img_widgets:
            box.on_safe_click(on_safe_button_click_generator(box))
            box.on_danger_click(on_danger_button_click_generator(box))

        submit_button.on_click(on_submit_click)
        display(display_box, clear=True)

    def part2(self):
        """
        Initializes the UI for the second part of the task.
        The user has to classify unknown test images as "safe" or "dangerous" by clicking the corresponding buttons.
        """
        self.sorted_images: dict[str, list[ImageData]] = {"safe": [], "danger": []}
        for img in self.images:
            if img.is_dangerous:
                self.sorted_images["danger"].append(img)
            else:
                self.sorted_images["safe"].append(img)

        box_layout = Layout(overflow="scroll", width="100%")
        danger_box = VBox([HTML("<b>Gefährlich</b>"), HBox([LabeledImageBox(x).widget for x in self.sorted_images["danger"]], layout=box_layout)])
        safe_box = VBox([HTML("<b>Harmlos</b>"), HBox([LabeledImageBox(x).widget for x in self.sorted_images["safe"]], layout=box_layout)])

        sort_text = HTML('<p><b>Beep:</b> Fertig! Ich hab für euch alle Roboter sortiert. Hier sind neun Beispiele für jede Kategorie. </p>')

        display_box = VBox([sort_text, danger_box, safe_box])

        display(display_box)

        instruction = HTML(
            '<p><b>Ben:</b> "Danke Beep. Aber wir haben immer noch ein Problem. Bei den Robotern, die die Türe verschließen wissen wir den Beruf nicht, nur wie sie aussehen."'
            '<p><b>Sarah:</b> Stimmt. Die Kinderbetreuer zu identifizieren ist leicht, aber die Assistenz und Aufseher Roboter sehen alle gleich aus.'
            '<p><b>Ben:</b> "Vielleicht können wir einfach raten?"'
            '<p><b>Beep:</b> "Das halte ich für keine gute Idee, aber versucht es ruhig. Hier sind ein paar Roboterakten, die ihr noch nicht gesehen habt. Ich habe diese vorhin zurückgehalten. Ich zeige euch auch direkt an, ob es stimmt. Grün ist wieder richtig und rot falsch."</p>')

        test_img_widgets = [ImageButtonBox(data) for data in self.validation_images]

        def on_safe_button_click_generator(button_box: ImageButtonBox):
            def on_safe_button_click(b):
                if button_box.solution:
                    button_box.set_safe_button_layout(style=color_to_string(colors['green']))
                    button_box.set_danger_button_layout(style="")
                    button_box.disable_buttons(True, True)
                else:
                    button_box.set_safe_button_layout(style=color_to_string(colors['red']))
                    button_box.set_danger_button_layout(style="")
                    button_box.disable_buttons(True, True)

            return on_safe_button_click

        def on_danger_button_click_generator(button_box: ImageButtonBox):
            def on_danger_button_click(b):
                if not button_box.solution:
                    button_box.set_danger_button_layout(style=color_to_string(colors['green']))
                    button_box.set_safe_button_layout(style="")
                    button_box.disable_buttons(True, True)
                else:
                    button_box.set_danger_button_layout(style=color_to_string(colors['red']))
                    button_box.set_safe_button_layout(style="")
                    button_box.disable_buttons(True, True)

            return on_danger_button_click

        for box in test_img_widgets:
            box.on_safe_click(on_safe_button_click_generator(box))
            box.on_danger_click(on_danger_button_click_generator(box))

        display_box.children += (instruction, HBox([box.widget for box in test_img_widgets]))

    def part3(self, dis: DisplayHandle = None):
        """
        Initializes the UI for the third part of the task.
        The user has to select images for training the model.
        Then the user can train the model with the selected images.
        The images are displayed with all information about them.
        """
        if dis is None:
            dis = display(HTML(""), display_id=True)
            #dis = display(VBox([]), display_id=True)

        train_button = Button(description="Trainieren", disabled=True)
        instruction = HTML("")
        categories = [ImageCategory(job, images) for job, images in self.categorized_images.items()]
        image_box = GridspecLayout(len(categories), 1)

        for i, category in enumerate(categories):
            image_box[i, 0] = category.widget
        display_box = VBox([instruction, image_box, train_button])

        def on_safe_button_click_generator(category: ImageCategory):
            def on_safe_button_click(b):
                category.save = True
                category.set_layout(box_layout=Layout(border=f'2px solid {color_to_string(colors["grey"])}'))
                category.set_button_style(safe_button_style=color_to_string(colors['grey']))
                category.disable_buttons(True, True)
                train_button.disabled = not all([cat.save is not None for cat in categories])

            return on_safe_button_click

        def on_danger_button_click_generator(category: ImageCategory):
            def on_danger_button_click(b):
                category.save = False
                category.set_layout(box_layout=Layout(border=f'2px solid {color_to_string(colors["grey"])}'))
                category.set_button_style(danger_button_style=color_to_string(colors['grey']))
                category.disable_buttons(True, True)
                train_button.disabled = not all([cat.save is not None for cat in categories])

            return on_danger_button_click

        for category in categories:
            category.on_safe_click(on_safe_button_click_generator(category))
            category.on_danger_click(on_danger_button_click_generator(category))

        train_button.on_click(lambda b: self._train_model_part3(dis, train_button, display_box, categories))

        dis.update(display_box)

    def _train_model_part3(self, dis: DisplayHandle, train_button: Button, display_box: VBox, categories: list[ImageCategory]):
        """
        Shows the training progress of the model and the results after training.
        It also displays some unknown test images to check the model's performance.
        """
        train_button.disabled = True
        model = load_model_mobilenet()
        safe_images = []
        danger_images = []
        for cat in categories:
            if cat.save:
                safe_images += cat._img_data
            elif not cat.save:
                danger_images += cat._img_data
        train_data = get_test_data([img for img in safe_images], [img for img in danger_images])
        validation_data = get_test_data([img for img in self.validation_images if img.is_clean], [img for img in self.validation_images if img.is_dangerous], batch=False)
        progress = IntProgress(
            value=0,
            min=0,
            max=train_data.cardinality().numpy(),
            description="Training: ",
            style={'bar_color': 'blue'},
            orientation='horizontal',
        )
        result_text = HTML("")
        train_text = HTML(
            '<p><b>Beep:</b> "Das Model wird jetzt trainiert. Hierfür geht das Model mehrmals durch die Daten durch und versucht sich zu verbessern, ähnlich wie wenn man Vokabeln lernt. Das kann ein '
            'bisschen dauern. Wir bekommen aber eine Rückmeldung, wie sich das Model verbessert. Dafür hat das Model die Daten in zwei Teile aufgeteilt. Mit dem einen Teil trainiert es und den '
            'anderen Teil benutzt es um sich selbst zu überprüfen. Nach jedem Durchlauf sehen wir, wie viel Prozent der Daten das Model bei der Überprüfung richtig hat."</p>')
        display_box.children += (train_text, progress, result_text)
        model.fit(train_data, validation_data=validation_data, epochs=5, callbacks=[TrainingCallback(progress, result_text)])
        #model.evaluate(test_data, callbacks=[TrainingCallback(progress, result_text)])

        restart_button = Button(description="Neu trainieren", layout=Layout(width="99%"))
        restart_button.on_click(lambda b: self.part3(dis))
        explanation = HTML('<p><b>Beep:</b> "Ich hab euch hier wieder die gleichen Roboterakten wie bei Aufgabe 1 zum Testen zur Verfügung gestellt."</p>')

        box = HBox(layout=Layout(width="fit-content", height="fit-content"))
        box.children = [TestBox(img, img_data, model).widget for img, img_data in zip(self.validation_images, validation_data)]

        display_box.children += (explanation, box, restart_button)

    def part4(self, dis: DisplayHandle = None):
        if dis is None:
            dis = display(HTML(""), display_id=True)
            #dis = display(VBox([]), display_id=True)

        train_button = Button(description="Trainieren", disabled=True)
        instruction = HTML("")
        categories = [ImageCategory(job, images) for job, images in self.categorized_images.items() if job != 'Kinderbetreuung']
        image_box = GridspecLayout(len(categories), 1)

        for i, category in enumerate(categories):
            image_box[i, 0] = category.widget
        display_box = VBox([instruction, image_box, train_button])

        def on_safe_button_click_generator(category: ImageCategory):
            def on_safe_button_click(b):
                category.save = True
                category.set_layout(box_layout=Layout(border=f'2px solid {color_to_string(colors["grey"])}'))
                category.set_button_style(safe_button_style=color_to_string(colors['grey']))
                category.disable_buttons(True, True)
                train_button.disabled = not all([cat.save is not None for cat in categories])

            return on_safe_button_click

        def on_danger_button_click_generator(category: ImageCategory):
            def on_danger_button_click(b):
                category.save = False
                category.set_layout(box_layout=Layout(border=f'2px solid {color_to_string(colors["grey"])}'))
                category.set_button_style(danger_button_style=color_to_string(colors['grey']))
                category.disable_buttons(True, True)
                train_button.disabled = not all([cat.save is not None for cat in categories])

            return on_danger_button_click

        for category in categories:
            category.on_safe_click(on_safe_button_click_generator(category))
            category.on_danger_click(on_danger_button_click_generator(category))

        train_button.on_click(lambda b: self._train_model_part4(dis, train_button, display_box, categories))

        dis.update(display_box)       
    
    def _train_model_part4(self, dis: DisplayHandle, train_button: Button, display_box: VBox, categories: list[ImageCategory]):
        """
        Shows the training progress of the model and the results after training.
        It also displays some unknown test images to check the model's performance.
        """
        train_button.disabled = True
        self.model = load_model_mobilenet()
        safe_images = []
        danger_images = []
        for cat in categories:
            if cat.save:
                safe_images += cat._img_data
            elif not cat.save:
                danger_images += cat._img_data
        train_data = get_test_data([img for img in safe_images], [img for img in danger_images])
        validation_data = get_test_data([img for img in self.validation_images if (img.is_clean and img.job !='Kinderbetreuung')], [img for img in self.validation_images if img.is_dangerous], batch=False)
        progress = IntProgress(
            value=0,
            min=0,
            max=train_data.cardinality().numpy(),
            description="Training: ",
            style={'bar_color': 'blue'},
            orientation='horizontal',
        )
        result_text = HTML("")
        train_text = HTML("")
        display_box.children += (train_text, progress, result_text)
        self.model.fit(train_data, validation_data=validation_data, epochs=5, callbacks=[TrainingCallback(progress, result_text)])
        #model.evaluate(test_data, callbacks=[TrainingCallback(progress, result_text)])

        restart_button = Button(description="Neu trainieren", layout=Layout(width="99%"))
        restart_button.on_click(lambda b: self.part3(dis))
        explanation = HTML('<p><b>Beep:</b> "Ich hab euch hier wieder die gleichen Roboterakten wie bei Aufgabe 1 zum Testen zur Verfügung gestellt."</p>')

        box = HBox(layout=Layout(width="fit-content", height="fit-content"))
        validation_data_temp = get_test_data([img for img in self.validation_images if img.is_clean], [img for img in self.validation_images if img.is_dangerous], batch=False)  # must include Kinderbetreuung to be consistent with self.validation_images in the next line.
        box.children = [TestBox(img, img_data, self.model).widget for img, img_data in zip(self.validation_images, validation_data_temp) if img.job != 'Kinderbetreuung']

        display_box.children += (explanation, box, restart_button)

#    def part5(self, dis: DisplayHandle = None):
#        """
#        Initializes the UI for the fourth part of the task.
#        Like in part 3, the user can train a model with the selected images.
#        The user can select images to be removed from the training set.
#        He can also select a different model to train.
#        """
#        if dis is None:
#            dis = display(HTML(""), display_id=True)
#        if self.sorted_images is None:
#            dis.update(HTML("<h3>Bitte zuerst Part 2 vollständig bearbeiten, damit wir mit den sortierten Bildern weiterarbeiten können</h3>"))
#            return
#        models = get_possible_models("models")
#        model_dropdown = Dropdown(
#            options=models,
#            index=0,
#            description="Model: ",
#        )
#        if not self.model:
#            self.model = load_model(list(models.values())[0])
#
#        def update_model(change):
#            model_path = change['new']
#            self.model = load_model(model_path)
#
#        model_dropdown.observe(update_model, names="value")
#        train_button = Button(description="Trainieren")
#        instructions = HTML('<p><b>Beep:</b> "Ich hab ein paar Bilder gefunden, die nicht ganz zu den anderen passen. Ihr könnt mal probieren, ob die Ergebnisse besser sind, '
#                            'wenn wir sie aus dem Training entfernen. Außerdem hab ich noch ein paar andere Modelle gefunden, die wir trainieren können."</p>')
#        safe_images = [x for x in self.sorted_images["safe"] if not x.is_clean]
#        danger_images = [x for x in self.sorted_images["danger"] if not x.is_clean]
#        safe_box = SelectionBox(safe_images)
#        danger_box = SelectionBox(danger_images)
#
#        selection_box = Tab(children=[safe_box.widget, danger_box.widget], titles=["Harmlos", "Gefährlich"], layout=Layout(width="fit-content"), box_style="info")
#        train_box = VBox(layout=Layout(width="fit-content"))
#        display_box = VBox([instructions, model_dropdown, selection_box, train_button, train_box])
#
#        train_button.on_click(lambda b: self._train_model_part4(
#            dis, train_box, train_button, model_dropdown,
#            [img for img in self.sorted_images["safe"] if img not in safe_box.selected],
#            [img for img in self.sorted_images["danger"] if img not in danger_box.selected]
#        ))
#
#        dis.update(display_box)
#
#    def _train_model_part5(self, dis: DisplayHandle, train_box: VBox, train_button: Button, model_dropdown: Dropdown, safe_images: list[ImageData], danger_images: list[ImageData]):
#        """
#        Shows the training progress of the model and the results after training.
#        It also displays some unknown test images to check the model's performance.
#        """
#        train_button.disabled = True
#        model_dropdown.disabled = True
#        test_data = get_test_data(safe_images, danger_images)
#        progress = IntProgress(
#            value=0,
#            min=0,
#            max=test_data.cardinality().numpy(),
#            description="Training: ",
#            style={'bar_color': 'blue'},
#            orientation='horizontal'
#        )
#        result_text = HTML("")
#        train_box.children += (progress, result_text)
#        self.model.evaluate(test_data, callbacks=[TrainingCallback(progress, result_text)], verbose=0)
#        restart_button = Button(description="Neu trainieren", layout=Layout(width="99%"))
#        restart_button.on_click(lambda b: self.part4(dis))
#        box = HBox(layout=Layout(width="fit-content", height="fit-content"))
#
#        box.children = [TestBox(img, self.model).widget for img in self.validation_images]
#        instruction = HTML('<p><b>Beep:</b> "Ich hab euch hier wieder die gleichen Roboterakten wie bei Aufgabe 1 zum Testen zur Verfügung gestellt."</p>')
#
#        train_box.children += (instruction, box, restart_button)

    def part6(self, dis: DisplayHandle = None):
        """
        Initializes the UI for the fifth part of the task.
        The user can select a door to open and check if the robot behind it is "safe" or "dangerous".
        To help with the decision, the user can use the model trained in part 4 to predict the robot's affiliation.
        """
        if dis is None:
            dis = display(HTML(""), display_id=True)
        if self.model is None:
            dis.update(HTML("<h3>Bitte zuerst Part 4 vollständig bearbeiten, damit wir ein Model haben, mit dem wir die Vorhersagen durchführen können</h3>"))
            return
        roboters = [
            {
                "data": robot_data,
                "name": robot_data.name,
                "text": HTML("Harmlos: -.-% <br> Gefährlich: -.-%"),
                "box": ImageBox(robot_data),
                "button": Button(description="Vorhersage", disabled=False),
                "title": HTML(robot_data.name),
                "door": Button(description=robot_data.name.replace("Roboter", "Tür"), disabled=False, style=dict(button_color=color_to_string(colors['brown'])),
                               layout=Layout(height="125px", width="75px", align_conten="center", justyfy_content="center", margin="10px", border=f"2px solid {color_to_string(colors['grey'])}"))
            }
            for robot_data in self.unknown_images
        ]
        for robot in roboters:
            robot["widget"] = VBox([robot["title"], robot["box"].widget, robot["text"], robot["button"]], layout=Layout(margin="10px", padding="5px"))
        roboters.sort(key=lambda r: r["name"])
        img_box = HBox([robot["widget"] for robot in roboters])

        instruction = HTML('<p><b>Beep:</b> "Wir können jetzt das Modell nutzen und für die Roboter hinter den Türen eine Vorhersage treffen. '
                           'Das Modell sagt uns dann, mit welcher Wahrscheinlichkeit es den Roboter als gefährlich und mit welcher als harmlos einstuft."')
        door_box_text = HTML("<h3>Tür wählen:</h3>")
        door_box = HBox([door_box_text, HBox([robot["door"] for robot in roboters])])
        result_box = HTML()

        def predict(img_path: str, text_widget: Widget):
            """
            Callback generator function for the button click event to predict the robots' affiliation based on the img_path.
            Args:
                img_path: The path to the image of the robot
                text_widget: The widget to display the prediction result

            Returns:
                The callback function for the button click event.
            """
            def on_click(b):
                img_array = load_image_for_prediction(img_path)
                prediction = self.model.predict(img_array)
                score = float(sigmoid(prediction[0][0]))
                text_widget.value = f"Harmlos: {score:.1%} <br> Gefährlich: {1 - score:.1%}"

            return on_click

        roboter1 = roboters[0]
        roboter2 = roboters[1]
        roboter3 = roboters[2]

        roboter1["button"].on_click(predict(roboter1["data"].img_path, roboter1["text"]))
        roboter2["button"].on_click(predict(roboter2["data"].img_path, roboter2["text"]))
        roboter3["button"].on_click(predict(roboter3["data"].img_path, roboter3["text"]))
        roboter1["door"].on_click(lambda b: self._on_door_select(dis, roboter1["data"], result_box, roboters))
        roboter2["door"].on_click(lambda b: self._on_door_select(dis, roboter2["data"], result_box, roboters))
        roboter3["door"].on_click(lambda b: self._on_door_select(dis, roboter3["data"], result_box, roboters))

        dis.update(VBox([instruction, img_box, door_box, result_box]), clear=True)

    def _on_door_select(self, dis: DisplayHandle, robot_data: ImageData, result: Widget, roboters: list[dict[str, Widget]]):
        """
        Displays the result of the door selection and updates the UI accordingly.
        Restarts the part if the user selects a dangerous robot.
        """
        if not robot_data.is_dangerous:
            result.value = ('<p>Ihr öffnet die Tür und schaut den Roboter erwartungsvoll an. Nach einem kurzen Moment geht er zur Seite und lässt euch durch.</p>'
                            '<p><b>Ben:</b> "Puh, das war die richtige Tür! Jetzt aber nichts wie raus hier!"</p>'
                            '<p><b>Sarah:</b> "Ich bin so froh, dass wir es geschafft haben!"</p>')
            for robot in roboters:
                robot["button"].disabled = True
                robot["door"].disabled = True
        else:
            result.value = ('<p>Ihr öffnet die Tür und schaut den Roboter erwartungsvoll an. Kaum merkt er, dass die Tür offen ist, kommt er bedrohlich auf euch zu. '
                            'Ihr schafft es gerade noch so die Tür wieder zuzuschlagen.</p>'
                            '<p><b>Ben:</b> Puh, das war knapp! Bei der nächsten Tür dürfen wir uns nicht mehr irren.</p>'
                            '<p><b>Sarah:</b> Wenigsten wissen wir jetzt, dass es nicht diese Tür ist.</p>')

            sleep(6)
            self.part5(dis)

