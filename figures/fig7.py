import json
import pickle

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize
import torch
from matplotlib.ticker import NullFormatter

import figure_config  # noqa: F402
from dopp import FeedForwardCell

mpl.rcParams.update(figure_config.mpl_style)


def _determine_output_rates(res, model, sample):
    if sample:
        return model.f(
            model.sample(torch.DoubleTensor(res["g0"]), torch.DoubleTensor(res["u0"]))
        ).numpy()
    else:
        return model.f(torch.DoubleTensor(res["u0"])).numpy()


def _compute_combined_rate(params, r):
    return (
        r[:, :, 0]
        + (params["min_rate_outputs"] + params["max_rate_outputs"] - r[:, :, 1])
    ) / 2.0


def _compute_class_from_rate(params, r):
    return r >= _rate_decision_boundary(params)


def _rate_decision_boundary(params):
    return (
        params["min_rate_outputs"]
        + (params["max_rate_outputs"] - params["min_rate_outputs"]) / 2.0
    )


def _fit_erfc_curve(angles, fcv, min_value=0.0, max_value=1.0):
    def shifted_erfc(x, gamma, lmd, mu, sigma):
        return 0.5 * scipy.special.erfc(-(x - mu) / (np.sqrt(2) * sigma))

    popt, _ = scipy.optimize.curve_fit(
        shifted_erfc, angles, fcv, p0=(min_value, max_value, 180.0, 10.0)
    )

    x = np.linspace(min(angles), max(angles), 100)
    y = shifted_erfc(x, popt[0], popt[1], popt[2], popt[3])
    assert len(x) == len(y)
    return popt, x, y


def _compute_psychometric_curve(params, res, model, sample):
    angles = np.empty(params["n_trials_test"])
    called_vertical_mean = np.empty(params["n_trials_test"])
    called_vertical_std = np.empty(params["n_trials_test"])

    r = _determine_output_rates(res, model, sample)
    combined_rate = _compute_combined_rate(params, r)
    class_output = _compute_class_from_rate(params, combined_rate)

    for i in range(params["n_trials_test"]):
        assert len(np.unique(res["angle_target"][i]) == 1)
        angles[i] = res["angle_target"][i, 0]
        called_vertical_mean[i] = np.mean(class_output[i])
        called_vertical_std[i] = np.std(class_output[i])

    order = np.argsort(angles)
    return angles[order], called_vertical_mean[order], called_vertical_std[order]


def _compute_class_target(params, res):
    return _compute_class_from_angle(params, res["angle_target"])


def _compute_score_for_random_experiments(score_per_trial):
    # WARNING: score_per_trial is organized in to X experiments for Y
    # fixed orientations; we flatten, shuffle, and reshape the array
    # to mimic experiments with W trials of random orientations
    score_per_trial = score_per_trial.flatten()
    score_per_trial = np.random.permutation(score_per_trial)
    score_per_trial = score_per_trial.reshape(20_000, -1)
    return np.mean(score_per_trial, axis=0)


def _compute_error_MAP(params, res, sample):
    if sample:
        angle_postfix = "sample"
    else:
        angle_postfix = "mu"

    class_output = _compute_class_from_angle(params, res[f"angle_MAP_{angle_postfix}"])

    score_per_trial = _compute_class_target(params, res) ^ class_output
    score_per_experiment = _compute_score_for_random_experiments(score_per_trial)
    mean = np.mean(score_per_experiment)
    sem = np.std(score_per_experiment)
    return mean, sem


def _compute_error_naive(params, res, sample):
    if sample:
        angle_postfix = "sample"
    else:
        angle_postfix = "mu"

    class_output = _compute_class_from_angle(
        params, res[f"angle_naive_{angle_postfix}"]
    )

    score_per_trial = _compute_class_target(params, res) ^ class_output
    score_per_experiment = _compute_score_for_random_experiments(score_per_trial)
    mean = np.mean(score_per_experiment)
    sem = np.std(score_per_experiment)
    return mean, sem


def _compute_class_from_angle(params, angle):
    return angle >= params["decision_boundary"]


def _compute_error_model(params, res, model, sample, by_trial=False):

    r = _determine_output_rates(res, model, sample)
    combined_rate = _compute_combined_rate(params, r)
    class_output = _compute_class_from_rate(params, combined_rate)

    score_per_trial = _compute_class_target(params, res) ^ class_output
    if by_trial:
        return score_per_trial
    else:
        score_per_experiment = _compute_score_for_random_experiments(score_per_trial)
        mean = np.mean(score_per_experiment)
        sem = np.std(score_per_experiment)
        return mean, sem


def plot_loss_bar_plot(ax, params, model, *, sample):

    errors_mean_sem = [
        _compute_error_MAP(params, res_final, sample),
        _compute_error_model(params, res_final, model, sample),
        _compute_error_naive(params, res_final, sample),
        _compute_error_model(params, res_final_1, model, sample),
        _compute_error_model(params, res_final_2, model, sample),
    ]

    ax.set_ylabel(
        r"Average loss in %", fontsize=figure_config.fontsize_medium,
    )
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())

    ax.bar(
        range(len(errors_mean_sem)),
        [mean for mean, _ in errors_mean_sem],
        yerr=[std for _, std in errors_mean_sem],
        color=[
            figure_config.colors["MAP"],
            figure_config.colors["VT"],
            figure_config.colors["sym"],
            figure_config.colors["V"],
            figure_config.colors["T"],
            "b",
        ],
    )
    ax.set_xticks(range(len(errors_mean_sem)))
    ax.set_xticklabels(
        ["MAP", "VT", "unw.\n avg.", "V", "T"], fontsize=figure_config.fontsize_medium,
    )
    ax.set_ylim(2e-2, 0.8e-1)
    ax.set_yticks([])
    ax.set_yticks([0.02, 0.04, 0.06, 0.08])
    ax.set_yticklabels([])
    ax.set_yticklabels([2, 4, 6, 8])


def plot_psychometric_curve(ax, params, model, *, sample):

    angles_final_psycho, mean_final_psycho, _ = _compute_psychometric_curve(
        params, res_final, model, sample
    )
    angles_final_psycho_1, mean_final_psycho_1, _ = _compute_psychometric_curve(
        params, res_final_1, model, sample
    )
    angles_final_psycho_2, mean_final_psycho_2, _ = _compute_psychometric_curve(
        params, res_final_2, model, sample
    )

    # fit to psychometric curves
    do_fit = True
    if do_fit:
        popt_final, x_final_fit, psychometric_curve_final_fit = _fit_erfc_curve(
            angles_final_psycho, mean_final_psycho
        )
        popt_final_1, x_final_fit_1, psychometric_curve_final_fit_1 = _fit_erfc_curve(
            angles_final_psycho_1, mean_final_psycho_1
        )
        popt_final_2, x_final_fit_2, psychometric_curve_final_fit_2 = _fit_erfc_curve(
            angles_final_psycho_2, mean_final_psycho_2
        )

    ax.set_xlabel(
        r"True orientation $\theta^*$", fontsize=figure_config.fontsize_medium
    )
    ax.set_ylabel("Proportion called vertical", fontsize=figure_config.fontsize_medium)
    ax.set_xlim(xlim_angle)
    ax.set_ylim(-0.1, 1.1)
    ax.set_xticks([0.0, 45.0, 90.0])

    # plot psychometric curves
    markersize = 3.0
    linewidth = 0.8

    downsampling_step = 12

    ax.plot(
        angles_final_psycho_1[::downsampling_step],
        mean_final_psycho_1[::downsampling_step],
        color=figure_config.colors["V"],
        marker=".",
        ls="",
        markersize=markersize,
        lw=linewidth,
    )
    if do_fit:
        ax.plot(
            x_final_fit_1,
            psychometric_curve_final_fit_1,
            color=figure_config.colors["V"],
            ls="-",
        )

    ax.plot(
        angles_final_psycho_2[::downsampling_step],
        mean_final_psycho_2[::downsampling_step],
        color=figure_config.colors["T"],
        marker=".",
        ls="",
        markersize=markersize,
        lw=linewidth,
    )
    if do_fit:
        ax.plot(
            x_final_fit_2,
            psychometric_curve_final_fit_2,
            color=figure_config.colors["T"],
            ls="-",
        )

    ax.plot(
        angles_final_psycho[::downsampling_step],
        mean_final_psycho[::downsampling_step],
        color=figure_config.colors["VT"],
        marker=".",
        ls="",
        markersize=markersize,
        lw=linewidth,
    )
    if do_fit:
        ax.plot(
            x_final_fit,
            psychometric_curve_final_fit,
            color=figure_config.colors["VT"],
            ls="-",
        )

    ax.axvline(params["decision_boundary"], lw=2, color="0.7")


if __name__ == "__main__":

    xlim_angle = (-25, 115.0)
    ylim_membrane = (-77, -22)
    ylim_conductance = (0.0, 2.7)

    with open("../experiments/fig7_params.json", "r") as f:
        params = json.load(f)

    np.random.seed(params["seed"])

    if params["current-based"]:
        postfix = "_current"
    else:
        postfix = ""

    with open(f"../data/fig7/res_test_VT{postfix}.pkl", "rb") as f:
        res_final = pickle.load(f)

    with open(f"../data/fig7/res_test_V{postfix}.pkl", "rb") as f:
        res_final_1 = pickle.load(f)

    with open(f"../data/fig7/res_test_T{postfix}.pkl", "rb") as f:
        res_final_2 = pickle.load(f)

    model_final = FeedForwardCell(params["in_features"], params["out_features"])
    model_final.gc = params["gc"]
    model_final.load_state_dict(torch.load(f"../data/fig7/model_final{postfix}.torch"))

    figsize = (7, 3)
    fig = plt.figure(figsize=figsize)

    sample = False

    ax_error = fig.add_axes([0.1, 0.15, 0.35, 0.8])
    plot_loss_bar_plot(ax_error, params, model_final, sample=sample)

    ax_psycho = fig.add_axes([0.55, 0.15, 0.35, 0.8])
    plot_psychometric_curve(ax_psycho, params, model_final, sample=sample)

    figname = "fig7.pdf"
    print(f"creating {figname}")
    fig.savefig(figname, dpi=300)
