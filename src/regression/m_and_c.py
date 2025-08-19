from time import sleep

from ipywidgets import Button, HBox, Output, Text, VBox
import matplotlib.pyplot as plt
import numpy as np

from utils.colors import colors


def m_and_c_plot():
    m = Text(value='', placeholder='Steigung ...', description='m:', disabled=False)
    c = Text(value='', placeholder='Y-Achse ...', description='c:', disabled=False)
    button = Button(description="Zeichne Gerade")
    out = Output()
    line = None
    
    # plot
    x = np.linspace(-1, 11, 100)
    y = x *2.4 -6
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(x, y, linewidth=2, color=colors['dark_blue'])
    ax.set_xlim(0, 10)
    plt.tight_layout()
    plt.show()
   
    # Submit button + effect
    def on_button_click(b):
        nonlocal line
        try:
            y_input = float(m.value) * x + float(c.value)
            if line is not None:
                line.remove()
            line, = ax.plot(x, y_input, color=colors['orange'], linewidth=2)
        except Exception as e:
            with out:
                out.clear_output()
                #display(e)
                display('Bitte Zahlen in die Felder eintragen!')
       
    button.on_click(on_button_click)
    
    # display buttons
    display(out)
    display(VBox([HBox([m, c]), button]))

