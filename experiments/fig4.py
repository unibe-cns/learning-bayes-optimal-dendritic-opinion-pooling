import json
import pickle
import sys

import numpy as np
import torch

import utils  # noqa: E402
from dopp import DynamicFeedForwardCell  # noqa: E402
from sim import (_compute_input_potentials, _compute_input_rates,  # noqa: E402
                 _compute_preferred_angles)

sys.path.insert(0, "../fig5_orientation_estimation/")

sys.path.insert(0, "../include/")


def sim(params, model):

    preferred_angles_input = _compute_preferred_angles(
        params["min_preferred_angles_input"],
        params["max_preferred_angles_input"],
        params["in_features"][0],
    )

    time_steps = int(params["sim_time"] / params["dt"])

    res = {}
    res["g0"] = torch.empty(
        params["n_noise_realizations"], time_steps, params["out_features"]
    )
    res["u0"] = torch.empty(
        params["n_noise_realizations"], time_steps, params["out_features"]
    )
    res["u0_sample"] = torch.empty(
        params["n_noise_realizations"], time_steps, params["out_features"]
    )

    noise_realizations = np.sort(
        np.random.normal(
            params["noise_mu"], params["noise_std"], params["n_noise_realizations"]
        )
    )

    with torch.no_grad():

        for i, angle_noise in enumerate(noise_realizations):

            print(f"trial {i+1}/{params['n_noise_realizations']}", end="\r")

            model.initialize_somatic_potential(-62.5)

            contrast_signal = params["contrast_signal_before"]
            contrast_noise = params["contrast_noise_before"]

            for step in range(time_steps):

                r_1, r_2 = _compute_input_rates(
                    params,
                    torch.Tensor([[params["angle_target"]]]),
                    torch.Tensor([[params["angle_target"]]]),
                    preferred_angles_input,
                    angle_noise,
                    angle_noise,
                    contrast_signal,
                    contrast_noise,
                )

                u_in = _compute_input_potentials(params, model, r_1, r_2)

                g0, u0 = model(u_in)

                res["g0"][i, step] = g0
                res["u0"][i, step] = u0
                res["u0_sample"][i, step] = model.sample(g0, u0)

                if step == time_steps // 3:
                    contrast_signal = params["contrast_signal_after"]
                    contrast_noise = params["contrast_noise_after"]

    print()

    return utils.numpyfy_all_torch_tensors_in_dict(res)


if __name__ == "__main__":

    crochet_colors = {
        "0": "#d38db7",
        "1": "#b2006a",
        "2": "#af131a",
        "v_rev": "#af131a",
    }

    with open("./fig7_params.json", "r") as f:
        params = json.load(f)

    with open("./fig4_params.json", "r") as f:
        params.update(json.load(f))

    np.random.seed(params["seed"] + 1)
    torch.manual_seed(params["seed"] + 1)

    model = DynamicFeedForwardCell(params["in_features"], params["out_features"])
    model.gc = params["gc"]
    model.gL0 = params["gL0"]
    model.gLd = torch.DoubleTensor(params["gLd"])
    model.dt = params["dt"]
    model.cm0 = params["cm0"]
    model.lambda_e = params["lambda_e"]

    model.load_state_dict(torch.load("../data/fig7/model_final.torch"))

    res = sim(params, model)

    with open(f"../data/fig4/res.pkl", "wb") as f:
        pickle.dump(res, f)
