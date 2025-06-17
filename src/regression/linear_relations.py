from ipywidgets import widgets
from IPython.display import display

from utils.colors import colors


def linear_relations():
    # Real world examples that should be categorized into linear or non-linear.
    examples = {
        "1": ("Anzahl der gelesenen Seiten in einem Buch und Lesedauer", True),
        "2": ("Anzahl der Knöpfe auf deinem Rucksack und die Menge an Hausaufgaben", False),
        "3": ("Anzahl der Störche in einer Stadt und Anzahl der geborenen Babys", False),
        "4": ("Anzahl der Schularbeiten und benötigte Zeit zum Korrigieren", True),
        "5": ("Länge des Schulwegs und benötigte Zeit", True),
        "6": ("Anzahl der Personen auf der Welt zwischen 1900 und jetzt (Abhängigkeit von der Zeit)", False),
        "7": ("Länge deiner Haare und die Anzahl der Freunde in der Klasse", False),
        "8": ("Geld für Sammelkarten und maximale Anzahl kaufbarer Artikel (gleich teuer)", True),
        "9": ("Geschwindigkeit auf dem Fahrrad und Bremsweg.", False),
        "10": ("Zeit zum Erledigen der Hausaufgaben (alle Aufgaben sind gleich schwer) "
               "und die Menge der erledigten Aufgaben", True),
    }

    # Solutions and explanations.
    explanations = {"1": "Linear. Wenn du z.B. 50 Seiten in 1 Stunde liest, dann brauchst du für 100 Seiten 2 Stunden "
                         "und für 150 Seiten 3 Stunden. Für 50 Seiten mehr, benötigst du also immer eine Stunde mehr.",
                    "2": "Kein Zusammenhang. Die Anzahl der Knöpfe auf deinem Rucksack hat keinen Einfluss auf die "
                         "Menge der Hausaufgaben. Probiere es gerne in den nächsten Unterrichtsstunden aus.",
                    "3": "Kein Zusammenhang. Die Anzahl der Störche in einer Stadt beeinflusst nicht die Anzahl der "
                         "geborenen Babys, auch wenn alte Geschichte das bestimmt anders sehen.",
                    "4": "Linear. Mehr Schularbeiten bedeuten im gleichen Maß mehr Korrekturzeit."
                         " Wenn dein Lehrer oder Lehrerin 5 Schularbeiten in 1 Stunde korrigiert, "
                         "korrigiert er/sie 10 in 2 Stunden, 15 in 3 Stunden.",
                    "5": "Je länger dein Schulweg ist, desto mehr Zeit brauchst du verhältnismäßig. "
                         "Wenn du für 1 km 10 Minuten brauchst, dann brauchst du für 2 km 20 Minuten und für 3 km 30 Minuten. "
                         "Das stimmt natürlich nur, wenn du immer mit der gleichen Geschwindigkeit unterwegs bist.",
                    "6": "Nicht linear. Die Weltbevölkerung wächst nicht gleichmäßig. "
                         "1950 ist die Weltbevölkerung um 43 Millionen gestiegen, 1990 allerdings um 93 Millionen."
                         "Die Wachstumsrate pro Jahr ist also nicht immer gleich.",
                    "7": "Kein Zusammenhange. "
                         "Die Länge deiner Haare beeinflusst nicht die Anzahl der Freunde in deiner Klasse.",
                    "8": "Linear. Wenn jede Karte 2 Euro kostet, "
                         "kannst du bei 10 Euro 5 Karten kaufen und bei 20 Euro 10 Karten.",
                    "9": "Nicht linear. Der Zusammenhange ist ungefähr quadratisch. "
                         "Bei 10 km/h benötigst du etwa 1.6 Meter zum bremsen, bei 20 km/h schon ca 6.4 Meter."
                         "Bei einem linearen Zusammenhang bräuchtest du bei 20 km/h etwa 3.2 Meter.",
                    "10": "Je mehr Zeit du mit deinen Hausaufgaben verbringst, desto mehr Aufgaben kannst du erledigen. "
                          "Wenn du 1 Stunde Hausaufgaben machst, erledigst du 3 Aufgaben. Wenn du 2 Stunden machst, "
                          "sind es 6 Aufgaben.",
    }

    # Create checkboxes
    # Checkboxes for all examples
    checkboxes = {n: widgets.Checkbox(description=value_[0], value=False, layout=widgets.Layout(width='90%'))
                  for n, value_ in examples.items()}
    # Checkboxes for all explanations
    explanation_widgets = {n: widgets.HTML(value=f"<p>{explanations[n]}</p>",
                                           layout=widgets.Layout(display='none', width='90%')) for n in examples}

    # Evaluate answers and give feedback by colored boxes and explanations.
    # Also the total number of correct answers is given.
    def evaluate(change):
        """Evaluates the given answers by analyzing the values of the checkboxes."""
        correct_count = 0
        for n, (_, is_linear) in examples.items():
            checkbox = checkboxes[n]
            explanation_widget = explanation_widgets[n]
            if checkbox.value == is_linear:
                checkbox.layout.border = f'2px solid rgb({colors["green"][0]*255}, {colors["green"][1]*255}, {colors["green"][2]*255})'
                correct_count += 1
            else:
                checkbox.layout.border = f'2px solid rgb({colors["red"][0]*255}, {colors["red"][1]*255}, {colors["red"][2]*255})'
            # Show explanations
            explanation_widget.layout.display = 'block'
        # Show number of correct answers.
        results_display.value = f"<h2>Du hast {correct_count} von {len(examples)} Beispielen richtig beantwortet.</h2>"
    # Create button 'Auswertung'
    evaluate_button = widgets.Button(description="Auswerten")
    evaluate_button.on_click(evaluate)
    # Create field for number of correct answers
    results_display = widgets.HTML()
    # Create fields for explanations
    checkbox_widgets = [widgets.VBox([checkboxes[key], explanation_widgets[key]], layout=widgets.Layout(margin='10px 0')) for key in examples]
    # Show all boxes in the correct order.
    display(widgets.VBox(checkbox_widgets))
    display(evaluate_button)
    display(results_display)

