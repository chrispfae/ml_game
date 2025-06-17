import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


palette = sns.color_palette("colorblind")
color_names = ['dark_blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'grey', 'yellow', 'light_blue']
colors = {}
for i, color in enumerate(palette):
    colors[color_names[i]] = color


def color_to_string(color):
    """Transforms a tuple of values to rgb. For example, (1, 0, 1) to 'rgb(255, 0, 255)'"""
    return f'rgb({int(color[0] * 255)}, {int(color[1] * 255)}, {int(color[2] * 255)})'

