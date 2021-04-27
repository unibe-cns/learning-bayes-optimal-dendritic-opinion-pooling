import pickle

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import figure_config  # noqa: E402

mpl.rcParams.update(figure_config.mpl_style)


def adjust_rate_scale(r):
    return 2.5 * r


if __name__ == "__main__":

    with open(f"../data/fig8/res.pkl", "rb") as f:
        res = pickle.load(f)
    res_VgT = res["res_VgT"]

    xlim = (1e-3, 179.0)
    xticks = np.array([1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2])
    xticks_rate = [1, 10, 100]
    yticks_rate = range(20, 70, 10)

    figsize = (4.0, 3.0)
    fig = plt.figure(figsize=figsize)

    ax = fig.add_axes([0.15, 0.18, 0.8, 0.78])
    ax.set_xlim(xlim)
    ax.set_ylim(0.0, 60.0)
    ax.set_xscale("log")
    ax.set_xlabel("Stimulus intensity", fontsize=figure_config.fontsize_medium)
    ax.set_ylabel("Firing rate (1/s)", fontsize=figure_config.fontsize_medium)
    ax.tick_params(axis="x", labelsize=figure_config.fontsize_small)
    ax.tick_params(axis="y", labelsize=figure_config.fontsize_small)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks)

    ax_inset = fig.add_axes([0.8, 0.33, 0.125, 0.18])
    ax_inset.set_xlim(1.0, 100.0)
    ax_inset.set_ylim(20.0, 60.0)
    ax_inset.set_xscale("log")
    ax_inset.tick_params(axis="x", labelsize=figure_config.fontsize_xxtiny)
    ax_inset.tick_params(axis="y", labelsize=figure_config.fontsize_xxtiny)
    ax_inset.set_xticks(xticks_rate)
    ax_inset.set_xticklabels(xticks_rate)
    ax_inset.set_yticks(yticks_rate)

    SPACING = 3

    ax.plot(
        res_VgT["contrasts"][::SPACING],
        adjust_rate_scale(res_VgT["r0_2"][:, 0][::SPACING]),
        label="T",
        color=figure_config.colors["T"],
        lw=figure_config.lw_medium,
        marker="o",
    )
    ax.plot(
        res_VgT["contrasts"][::SPACING],
        adjust_rate_scale(res_VgT["r0_1"][:, 0][::SPACING]),
        label="V",
        color=figure_config.colors["V"],
        lw=figure_config.lw_medium,
        marker="o",
    )
    ax.plot(
        res_VgT["contrasts"][::SPACING],
        adjust_rate_scale(res_VgT["r0"][:, 0][::SPACING]),
        label="VT",
        color=figure_config.colors["VT"],
        lw=figure_config.lw_medium,
        marker="o",
    )
    ax.axhline(
        adjust_rate_scale(res_VgT["r0_null"][0, 0]),
        color=figure_config.colors["prior"],
        ls="--",
        zorder=-1,
    )
    ax.axhline(
        adjust_rate_scale(res_VgT["r0_1_inf"][0, 0]),
        color=figure_config.colors["V"],
        ls="--",
        zorder=-1,
    )
    ax.axhline(
        adjust_rate_scale(res_VgT["r0_2_inf"][0, 0]),
        color=figure_config.colors["T"],
        ls="--",
        zorder=-1,
    )
    ax.legend(fontsize=figure_config.fontsize_tiny, loc="upper left")

    ax_inset.plot(
        res_VgT["contrasts"][::SPACING],
        adjust_rate_scale(res_VgT["r0_2"][:, 0][::SPACING]),
        label="T",
        color=figure_config.colors["T"],
        lw=0.75 * figure_config.lw_medium,
        marker="o",
        markersize=3.5,
    )
    ax_inset.plot(
        res_VgT["contrasts"][::SPACING],
        adjust_rate_scale(res_VgT["r0_1"][:, 0][::SPACING]),
        label="V",
        color=figure_config.colors["V"],
        lw=0.75 * figure_config.lw_medium,
        marker="o",
        markersize=3.5,
    )
    ax_inset.plot(
        res_VgT["contrasts"][::SPACING],
        adjust_rate_scale(res_VgT["r0"][:, 0][::SPACING]),
        label="VT",
        color=figure_config.colors["VT"],
        lw=0.75 * figure_config.lw_medium,
        marker="o",
        markersize=3.5,
    )

    figname = "fig8.pdf"
    print(f"creating {figname}")

    fig.savefig(figname, dpi=300)
