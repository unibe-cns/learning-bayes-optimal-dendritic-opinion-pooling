import json
import pickle
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import figure_config  # noqa: E402

sys.path.insert(0, "../experiments")
import utils  # noqa: E402

mpl.rcParams.update(figure_config.mpl_style)


def adjust_membrane_potential_scale(u):
    return u - 8.0


def _plot_potentials(ax, params, res):
    ax.set_xlabel("Time (ms)", fontsize=figure_config.fontsize_large)
    ax.set_ylabel("Somatic potential (mV)", fontsize=figure_config.fontsize_large)
    ax.tick_params(axis="x", labelsize=figure_config.fontsize_large)
    ax.tick_params(axis="y", labelsize=figure_config.fontsize_large)

    times = (
        np.linspace(0.0, params["sim_time"], int(params["sim_time"] / params["dt"]))
        - params["sim_time"] / 3.0
    )
    for i in range(params["n_noise_realizations"]):
        u0_sample_filtered = utils.moving_average(
            res["u0_sample"][i, :, 0], window_size
        )
        ax.plot(
            times[window_size : len(u0_sample_filtered) + window_size],
            adjust_membrane_potential_scale(u0_sample_filtered),
            lw=2,
        )

    ax.axvline(0.0, lw=1.5, color="k", zorder=-1)


def _plot_average_potentials(ax, params, res):
    ax.set_xlabel("Time (ms)", fontsize=figure_config.fontsize_large)
    ax.set_ylabel("Somatic potential (mV)", fontsize=figure_config.fontsize_large)
    ax.set_xlim(-20.0, 80.0)
    ax.set_xticks([-20.0, 0.0, 20.0, 40.0, 60.0, 80.0])
    ax.set_yticks(np.arange(-84, 60, 4))
    ax.tick_params(axis="x", labelsize=figure_config.fontsize_medium)
    ax.tick_params(axis="y", labelsize=figure_config.fontsize_medium)

    u0_low = []
    u0_middle = []
    u0_high = []
    for i in range(params["n_noise_realizations"]):
        u0_sample_filtered = utils.moving_average(
            res["u0_sample"][i, :, 0], window_size
        )
        u0_sample_filtered = adjust_membrane_potential_scale(u0_sample_filtered)
        u0_sample_filtered_pre = u0_sample_filtered[
            int(len(u0_sample_filtered) * rel_time_stim_onset)
        ]
        mean_pre = np.mean(u0_sample_filtered_pre)

        if mean_pre < -72:
            u0_low.append(u0_sample_filtered)
        elif (-72.0 <= mean_pre) and (mean_pre < -69.0):
            u0_middle.append(u0_sample_filtered)
        else:
            u0_high.append(u0_sample_filtered)

    times = (
        np.linspace(0.0, params["sim_time"], int(params["sim_time"] / params["dt"]))
        - params["sim_time"] / 3.0
    )
    if u0_low:
        ax.plot(
            times[window_size : len(u0_sample_filtered) + window_size],
            np.mean(u0_low, axis=0),
            color=crochet_colors["0"],
            lw=2,
        )
    if u0_middle:
        ax.plot(
            times[window_size : len(u0_sample_filtered) + window_size],
            np.mean(u0_middle, axis=0),
            color=crochet_colors["1"],
            lw=2,
        )
    if u0_high:
        ax.plot(
            times[window_size : len(u0_sample_filtered) + window_size],
            np.mean(u0_high, axis=0),
            color=crochet_colors["2"],
            lw=2,
        )
    ax.axvline(0.0, lw=1.5, color="k", zorder=-1)


def _plot_correlation(ax, params, res):
    ax.set_ylabel(r"$\Delta V_\mathsf{m}$ (mV)", fontsize=figure_config.fontsize_large)
    ax.set_xlabel(r"Prestimulus potential (mV)", fontsize=figure_config.fontsize_large)
    # ax.set_xlim(-70.0, -45.0)
    ax.set_xlim(-77, -62)
    ax.set_ylim(-6, 8)
    ax.set_xticks(np.arange(-76, -60, 4))
    ax.set_yticks(np.arange(-4, 12, 4))
    ax.tick_params(axis="x", labelsize=figure_config.fontsize_medium)
    ax.tick_params(axis="y", labelsize=figure_config.fontsize_medium)

    for i in range(params["n_noise_realizations"]):
        u0_sample_filtered = utils.moving_average(
            res["u0_sample"][i, :, 0], window_size
        )
        u0_sample_filtered = adjust_membrane_potential_scale(u0_sample_filtered)
        pre_u, delta_u = _compute_pre_u_and_delta_u(params, u0_sample_filtered)

        ax.plot(
            pre_u, delta_u, ls="", marker="o", color="0.8", markeredgecolor="k",
        )

    ax.axhline(0.0, ls="-", color="k", zorder=-1)
    v_rev = np.mean(res["u0_sample"][:, -150:, 0])
    v_rev = adjust_membrane_potential_scale(v_rev)
    ax.axvline(v_rev, ls=":", color=crochet_colors["v_rev"], zorder=-1)


def _compute_pre_u_and_delta_u(params, u0_sample_filtered):
    delta_t = 10  # (ms)
    delta_t_steps = int(delta_t / params["dt"])
    pre_u = u0_sample_filtered[int(len(u0_sample_filtered) * rel_time_stim_onset)]
    post_u = u0_sample_filtered[
        int(len(u0_sample_filtered) * rel_time_stim_onset) + delta_t_steps
    ]
    delta_u = post_u - pre_u
    return pre_u, delta_u


if __name__ == "__main__":

    crochet_colors = {
        "0": "#af131a",
        "1": "#b2006a",
        "2": "#d38db7",
        "v_rev": "#af131a",
    }

    rel_time_stim_onset = 1 / 3

    window_size = 20

    with open("../experiments/fig7_params.json", "r") as f:
        params = json.load(f)

    with open("../experiments/fig4_params.json", "r") as f:
        params.update(json.load(f))

    with open(f"../data/fig4/res.pkl", "rb") as f:
        res = pickle.load(f)

    fig = plt.figure(figsize=figure_config.double_figure_horizontal)
    ax_potentials = fig.add_axes([0.1, 0.16, 0.38, 0.8])
    ax_correlation = fig.add_axes([0.59, 0.16, 0.38, 0.8])

    # _plot_potentials(ax_potentials, params, res)
    _plot_average_potentials(ax_potentials, params, res)
    _plot_correlation(ax_correlation, params, res)

    figname = "fig4.pdf"
    print(f"creating {figname}")

    fig.savefig(figname, dpi=300)
