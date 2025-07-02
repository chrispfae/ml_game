import threading
import time

from ipywidgets import interactive, widgets, VBox
from IPython.display import display
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

from utils.colors import colors


def get_independent_variable():
    """Get independent variable (x values)."""
    rng = np.random.default_rng(51)
    independent_variable = rng.uniform(110, 170, 20)
    independent_variable[-1] += 5
    return independent_variable


def get_dependent_variable(independent_variable):
    """Get dependent variable (y values)."""
    rng = np.random.default_rng(53)
    dependent_variable = 10 + (independent_variable - 110) * (30 - 10) / (170 - 110) + rng.normal(0, 3, independent_variable.size)
    return dependent_variable


def plot_data():
    independent_variable = get_independent_variable()
    dependent_variable = get_dependent_variable(independent_variable)

    # Plot height vs speed.
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(independent_variable, dependent_variable, color=colors['dark_blue'], label='Daten der Kinder')

    # Set annotations
    ax.set_xlabel('Körpergröße (cm)')
    ax.set_ylabel('Maximale Geschwindigkeit (km/h)')
    ax.set_title('Größe vs. maximale Geschwindigkeit', y=1.2)
    ax.legend(loc='upper left', bbox_to_anchor=(0.0, 1.2))
    ax.grid(True)
    plt.ylim([-1.5, 31.5])
    # Show plot
    plt.tight_layout()
    plt.show()


def mse(ground_truth, measured_values):
    """Compute mean squared error."""
    return np.mean((ground_truth - measured_values) ** 2)


def fit_regression_line(independent_variable, dependent_variable, power=1):
    """
    Fit linear regression line to data."""
    X = independent_variable.reshape(-1, 1)
    X = PolynomialFeatures(degree=power).fit_transform(X)
    model = LinearRegression().fit(X[:, 1:], dependent_variable)
    ml_slope = model.coef_
    ml_intercept = model.intercept_
    ml_predictions = model.predict(X[:, 1:])
    ml_mse = mse(dependent_variable, ml_predictions)
    return ml_slope, ml_intercept, ml_predictions, ml_mse


def lineare_regression():
    independent_variable = get_independent_variable()
    dependent_variable = get_dependent_variable(independent_variable)

    # Plot height vs speed.
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(independent_variable, dependent_variable, color=colors['dark_blue'], label='Daten der Kinder')

    # Plot adjustable regression line.
    default_slope = 0
    default_intercept = 0
    default_predictions = default_slope * independent_variable + default_intercept
    default_mse = mse(dependent_variable, default_predictions)
    line, = ax.plot(independent_variable, default_slope * independent_variable + default_intercept, color=colors['orange'],
                    linestyle='-', label=f'Eure Gerade: Output = {default_slope}'.replace('.', ',') + f'* Input + {default_intercept}'.replace('.', ',') + f'(Fehlerwert: {default_mse:.2f})'.replace('.', ','))

    def update_adjustable_line(slope, intercept):
        """Update the adjustable regression line, which is changed by the sliders."""
        student_predictions = slope * independent_variable + intercept
        student_mse = mse(dependent_variable, student_predictions)
        line.set_ydata(student_predictions)
        line.set_label(f"Eure Gerade: Output = {slope:.2f}".replace('.', ',') + f" * Input + {intercept:.2f}".replace('.', ',') + f" (Fehlerwert: {student_mse:.2f})".replace('.', ','))
        ax.legend()
        fig.canvas.draw_idle()

    w_adj_line = interactive(update_adjustable_line,
                             slope=widgets.FloatSlider(min=-1.0, max=1.0, step=0.01, value=0, description='Steigung'),
                             intercept=widgets.FloatSlider(min=-40, max=5, step=0.01, value=0, description='y-Achse'),)
    display(w_adj_line)

    # Plot the ml regression line (by default invisible, see checkbox).
    ml_slope, ml_intercept, ml_predictions, ml_mse = fit_regression_line(independent_variable, dependent_variable)
    regression_line, = ax.plot(independent_variable, ml_predictions, color=colors['green'], linestyle='-', label='',
                               visible=False)
    # Placeholder for the annotation warning, if the manual line is too far away from a good result.
    annotation_warning = ax.annotate(
        "Die manuelle Gerade ist noch etwas zu schlecht.\n Bitte versuche es erneut.",
        xy=(125, 15),
        xytext=(125, 15.2),
        textcoords="data",
        bbox=dict(boxstyle="round,pad=0.3", fc=colors['brown'], ec=colors['grey'], lw=1),
        visible=False,  # initially hidden
        fontsize=14
    )
    def update_regression_line(change):
        """Function to show or hide the regression line depending on the checkbox."""
        if change['new']:  # Checkbox is activated
            intercept_current = w_adj_line.kwargs['intercept']
            slope_current = w_adj_line.kwargs['slope']
            pred_current = intercept_current + slope_current * independent_variable
            mse_current = mse(dependent_variable, pred_current)
            if abs(mse_current - ml_mse) > 5:
                annotation_warning.set_visible(True)
                regression_line.set_visible(False)
                regression_line.set_label('')
                fig.canvas.draw_idle()  # redraw line
                def hide_warning_after_delay():
                    time.sleep(3)
                    checkbox.value = False
                    annotation_warning.set_visible(False)
                    fig.canvas.draw_idle()
                # Run the delay in a background thread
                threading.Thread(target=hide_warning_after_delay).start()
            else:
                regression_line.set_visible(True)
                regression_line.set_label(
                f'Roboter Gerade: Output = {ml_slope[0]:.2f}'.replace('.', ',') + f' * Input + {ml_intercept:.2f}'.replace('.', ',') + f' (Fehlerwert: {ml_mse:.2f})'.replace('.', ','))
        else:  # Checkbox is deactivated
            regression_line.set_visible(False)
            regression_line.set_label('')
        ax.legend()
        fig.canvas.draw_idle()  # redraw line
    # Create a checkbox, which execute the function update_regression line if it is clicked.
    checkbox = widgets.Checkbox(value=False, description='Zeige Regressionsgerade', disabled=False, indent=False)
    checkbox.observe(update_regression_line, names='value')
    display(checkbox)


    # Set annotations
    ax.set_xlabel('Körpergröße (cm)')
    ax.set_ylabel('Maximale Geschwindigkeit (km/h)')
    ax.set_title('Größe vs. maximale Geschwindigkeit', y=1.2)
    ax.legend(loc='upper left', bbox_to_anchor=(0.0, 1.2))
    ax.grid(True)
    # Show plot
    plt.tight_layout()
    plt.show()


def zeige_lineare_regression():
    independent_variable = get_independent_variable()
    dependent_variable = get_dependent_variable(independent_variable)

    # Plot height vs speed.
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(independent_variable, dependent_variable, color=colors['dark_blue'], label='Daten der Kinder')

    # Plot the ml regression line (by default invisible, see checkbox).
    ml_slope, ml_intercept, ml_predictions, ml_mse = fit_regression_line(independent_variable, dependent_variable)
    regression_line, = ax.plot(independent_variable, ml_predictions, color=colors['green'], linestyle='-', label='',
                               visible=True)
    regression_line.set_label(
        f'Roboter Gerade: Output = {ml_slope[0]:.2f}'.replace('.', ',') + 
        f' * Input + {ml_intercept:.2f}'.replace('.', ',') + 
        f' (Fehlerwert: {ml_mse:.2f})'.replace('.', ','))

    ax.text(145.5, 21, "Sarah", fontsize=9, ha='left', color=colors['purple'])
    ax.scatter(145, 21.3, color=colors['purple'])

    ax.text(152.5, 26.7, "Ben", fontsize=9, ha='right', color=colors['purple'])
    ax.scatter(153, 27, color=colors['purple'])  

    ax.set_xlabel('Körpergröße (cm)')
    ax.set_ylabel('Maximale Geschwindigkeit (km/h)')
    ax.set_title('Größe vs. maximale Geschwindigkeit', y=1.2)
    ax.legend(loc='upper left', bbox_to_anchor=(0.0, 1.2))
    ax.grid(True)
    # Show plot
    plt.tight_layout()
    plt.show()

    person_toggle = widgets.ToggleButtons(
        options=['Ben', 'Sarah'],
        style={'font_weight': 'normal'},
        layout=widgets.Layout(width='50%')
    )

    # Ergebnisanzeige
    results_display = widgets.HTML()

    # Auswerten-Button
    evaluate_button = widgets.Button(description='Auswerten')

    def on_evaluate(change):
        selected = person_toggle.value
        if selected == 'Ben':
            rgb = f'rgb({int(colors["green"][0]*255)}, {int(colors["green"][1]*255)}, {int(colors["green"][2]*255)})'
            person_toggle.style.button_color = rgb
            results_display.value = (
                "<h3 style='color:green;font-size:12px;font-weight:normal;'>"
                "Richtig! Ben sollte laufen, weil er schneller ist, als die Roboter vorhergesagt haben. Im Diagramm liegt sein Punkt deutlich oberhalb der grünen Linie. "
                "</h3>"
            )
        else:
            rgb = f'rgb({int(colors["red"][0]*255)}, {int(colors["red"][1]*255)}, {int(colors["red"][2]*255)})'
            person_toggle.style.button_color = rgb
            results_display.value = (
                "<h3 style='color:red;font-size:12px;font-weight:normal;'>"
                "Leider falsch. Versuche es nochmal.</h3>"
            )

    evaluate_button.on_click(on_evaluate)

    display(VBox([person_toggle, evaluate_button, results_display]))
