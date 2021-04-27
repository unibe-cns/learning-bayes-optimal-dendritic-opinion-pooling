import pickle

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import figure_config

mpl.rcParams.update(figure_config.mpl_style)

LOG = True


def plot_weights(ax, w, *, ylabel):
    ax.set_ylabel(ylabel, fontsize=figure_config.fontsize_xsmall)
    ax.set_xlim(plot_params["xlim"])
    ax.set_ylim(plot_params["ylim_weights"])
    if LOG:
        ax.set_xscale("log")
    ax.set_xticklabels([])

    ax.axvline(55.0, ls="--", color="0.4")
    ax.plot(times, w[:, 0], color=figure_config.colors["V"], lw=1.5)
    ax.plot(times, w[:, 1], color=figure_config.colors["T"], lw=1.5)


def plot_ratio(ax, wE, wI):
    ax.set_ylabel(
        r"$\frac{W^\mathsf{E}}{W^\mathsf{E} + W^\mathsf{I}}$",
        fontsize=figure_config.fontsize_small,
    )
    ax.set_xlim(plot_params["xlim"])
    ax.set_ylim(plot_params["ylim_ratio"])
    if LOG:
        ax.set_xscale("log")
    ax.set_xticklabels([])

    ax.axvline(55.0, ls="--", color="0.4")
    ax.plot(
        times,
        (wE[:, 0] / (wE[:, 0] + wI[:, 0])),
        color=figure_config.colors["V"],
        lw=1.5,
    )
    ax.plot(
        times,
        (wE[:, 1] / (wE[:, 1] + wI[:, 1])),
        color=figure_config.colors["T"],
        lw=1.5,
    )


def plot_sum(ax, wE, wI):
    ax.set_ylabel(
        r"$W^\mathsf{E} + W^\mathsf{I}$", fontsize=figure_config.fontsize_xsmall
    )
    ax.set_xlim(plot_params["xlim"])
    ax.set_ylim(plot_params["ylim_ratio"])
    ax.set_xlabel("Time (s)", fontsize=figure_config.fontsize_xsmall)
    if LOG:
        ax.set_xscale("log")

    ax.axvline(55.0, ls="--", color="0.4")
    ax.plot(times, (wE[:, 0] + wI[:, 0]), color=figure_config.colors["V"], lw=1.5)
    ax.plot(times, (wE[:, 1] + wI[:, 1]), color=figure_config.colors["T"], lw=1.5)


if __name__ == "__main__":

    with open(f"../data/fig6/params.pkl", "rb") as f:
        params = pickle.load(f)

    plot_params = {
        "xlim": (50e0, 1500.0),
        # "xlim": (0, 150.0),
        "ylim_weights": (-0.02, 0.4),
        "ylim_weights_dot": (-0.02, 0.02),
        "ylim_ratio": (-0.04, 1.0),
    }

    with open(f"../data/fig6/res.pkl", "rb") as f:
        res = pickle.load(f)

    fig = plt.figure(figsize=(3.25, 2.25))

    x_pos = 0.175
    width = 0.78
    height = 0.14

    ax_weightsE = fig.add_axes([x_pos, 0.83, width, height])
    ax_weightsI = fig.add_axes([x_pos, 0.61, width, height])
    ax_ratio_weights = fig.add_axes([x_pos, 0.39, width, height])
    ax_sum_weights = fig.add_axes([x_pos, 0.17, width, height])

    trials = np.arange(0, params["trials"], 1.0 * params["recording_interval"])
    times = trials * 10.0
    times *= 1e-3  # ms to s

    plot_weights(ax_weightsE, res["wEd"], ylabel=r"$W^\mathsf{E}$")
    plot_weights(ax_weightsI, res["wId"], ylabel=r"$W^\mathsf{I}$")
    plot_ratio(ax_ratio_weights, res["wEd"], res["wId"])
    plot_sum(ax_sum_weights, res["wEd"], res["wId"])

    figname = "fig6.pdf"
    print(f"creating {figname}")
    plt.savefig(figname, dpi=300)
