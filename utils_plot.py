import numpy as np
import matplotlib.pyplot as plt
import utils as ut


def get_standard_color(i):
    return plt.rcParams["axes.prop_cycle"].by_key()["color"][i % 10]


def plot_magnetisation(parameters, iteration):
    data = ut.load_data(parameters, iteration)
    if iteration == 0:
        label = "Starting state"
        color = "black"
    else:
        label = f"Iter = {iteration}"
        color = get_standard_color(iteration - 1)

    plt.plot(np.linspace(data["dt"], data["T"] * data["dt"], data["T"]), data["m"], lw=1.5, alpha=0.5, marker="", label=label, color=color)
