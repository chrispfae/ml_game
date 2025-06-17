from ipywidgets import widgets
from IPython.display import display
import matplotlib.pyplot as plt
import numpy as np

from regression.linear_regression import fit_regression_line
from utils.colors import colors

def fmt_de(val):
    return f"{val:.2f}".replace('.', ',')

def get_independent_variable():
    return np.linspace(0, 200, 100)


def get_dependent_variable(independent_variable):
    rng = np.random.default_rng(34)
    offset = 10
    dependent_variable = (0.005 * independent_variable**2 +
                          rng.normal(loc=0, scale=14, size=len(independent_variable)) + offset)
    return dependent_variable


def quadratic_regression():
    # Plot data
    fig, ax = plt.subplots(figsize=(7, 5))
    distances = get_independent_variable()
    times = get_dependent_variable(distances)
    ax.scatter(distances, times, color=colors['dark_blue'], label='Datenpunkte')

    # Plot linear model
    linear_slope, linear_intercept, linear_predictions, linear_mse = fit_regression_line(distances, times)
    linear_line, = ax.plot(distances, linear_predictions, color=colors['green'], linestyle='-', label=f' ', visible=False, lw=2)

    # Plot quadratic model
    quadratic_slope, quadratic_intercept, quadratic_predictions, quadratic_mse = fit_regression_line(distances, times, power=2)
    poly_line, = ax.plot(distances, quadratic_predictions, color=colors['orange'], linestyle='-', label =f' ', visible=False, lw=2)

    # Checkbox linear model
    checkbox_linear = widgets.Checkbox(
        value=False,
        description='Zeige lineares Modell',
        disabled=False,
        indent=False
    )

    # Checkbox quadratic model
    checkbox_poly = widgets.Checkbox(
        value=False,
        description='Zeige quadratisches Modell',
        disabled=False,
        indent=False
    )

    def update_lines(change):
        # Show fitted models depending on checkbox values
        linear_line.set_visible(checkbox_linear.value)
        poly_line.set_visible(checkbox_poly.value)

        # Get current handles and labels of the legend
        handles_lin, labels_lin = ax.get_legend_handles_labels()
        # Modify labels of linear and quadratic line
        if checkbox_linear.value:
            labels_lin[1] = (f'Lineares Modell: y = {fmt_de(linear_slope[0])} * x + {fmt_de(linear_intercept)} (Fehlerwert: {fmt_de(linear_mse)})'
)
        else:
            labels_lin[1] = ''
        if checkbox_poly.value:
            labels_lin[2] = (f'Quadratisches Modell: y = {fmt_de(quadratic_slope[1])} * x^2 + {fmt_de(quadratic_slope[0])} * x + {fmt_de(quadratic_intercept)} (Fehlerwert: {fmt_de(quadratic_mse)})'
)
        else:
            labels_lin[2] = ''
        # Aktualisieren der Legende und Neuzeichnen der Grafik.
        ax.legend(handles_lin, labels_lin)
        fig.canvas.draw()

    # Add function to checkboxes
    checkbox_linear.observe(update_lines, names='value')
    display(checkbox_linear)
    checkbox_poly.observe(update_lines, names='value')
    display(checkbox_poly)

    # Plot annotations
    plt.xlabel('Strecke (m)')
    plt.ylabel('Zeit (s)')
    plt.title('Strecke vs. Zeit')
    plt.legend()
    plt.grid(True)
    plt.show()


def model_selection():
    # Frage anzeigen
    question = widgets.HTML("<h3>Passt das lineare oder das quadratische Modell besser? Woran siehst du das?</h3>")

    # Auswahl-Widget (entweder/oder)
    model_choice = widgets.ToggleButtons(
        options=['Lineares Modell', 'Quadratisches Modell'],
        style={'font_weight': 'normal', 'font_size': '14px', 'text_color': 'black'},
        layout=widgets.Layout(width='90%')
    )

    # Erklärung als Spoiler
    explanation = widgets.HTML(
        "<p>Das quadratische Modell passt besser, weil es einen geringeren Fehlerwert hat als das lineare Modell. "
        "Es beschreibt den Zusammenhang zwischen Strecke und Zeit realistischer.</p>")
    accordion = widgets.Accordion(children=[explanation])
    accordion.set_title(0, 'Erklärung')
    accordion.selected_index = None  # Start geschlossen

    # Ergebnisanzeige
    results_display = widgets.HTML()

    # Auswertungsknopf
    evaluate_button = widgets.Button(description="Auswerten")

    # ✅ Lokale Funktion mit Zugriff auf Widgets
    def evaluate_model(change):
        selected = model_choice.value
        if selected == 'Quadratisches Modell':
            color = colors["green"]
            results_display.value = "<h3 style='color:green;font-size:12px;font-weight:normal;'>Richtig! Der Zusammenhang ist quadratisch.</h3>"
        else:
            color = colors["red"]
            results_display.value = "<h3 style='color:red;font-size:12px;font-weight:normal;'>Leider falsch. Der Zusammenhang ist quadratisch.</h3>"

        border = f'2px solid rgb({color[0]*255}, {color[1]*255}, {color[2]*255})'
        model_choice.layout.border = border

    # Verknüpfen
    evaluate_button.on_click(evaluate_model)

    # Anzeigen
    display(question, model_choice, evaluate_button, results_display, accordion)