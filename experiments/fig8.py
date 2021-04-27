import json
import pickle

import numpy as np
import torch

import utils  # noqa: E402
from dopp import FeedForwardCell
from sim import (_compute_input_potentials, _compute_input_rates,  # noqa: E402
                 _compute_preferred_angles)


def sim(params, model, *, angle_1, angle_2):

    angle_1 = torch.Tensor([[angle_1]])
    angle_2 = torch.Tensor([[angle_2]])

    preferred_angles_input = _compute_preferred_angles(
        params["min_preferred_angles_input"],
        params["max_preferred_angles_input"],
        params["in_features"][0],
    )

    contrasts = torch.logspace(-3, 3, 50)

    res = {}
    res["contrasts"] = contrasts
    res["g0"] = torch.empty(len(contrasts), params["out_features"])
    res["u0"] = torch.empty(len(contrasts), params["out_features"])
    res["r0"] = torch.empty(len(contrasts), params["out_features"])
    res["g0_1"] = torch.empty(len(contrasts), params["out_features"])
    res["u0_1"] = torch.empty(len(contrasts), params["out_features"])
    res["r0_1"] = torch.empty(len(contrasts), params["out_features"])
    res["g0_2"] = torch.empty(len(contrasts), params["out_features"])
    res["u0_2"] = torch.empty(len(contrasts), params["out_features"])
    res["r0_2"] = torch.empty(len(contrasts), params["out_features"])
    with torch.no_grad():

        r_1, r_2 = _compute_input_rates(
            params,
            angle_1,
            angle_2,
            preferred_angles_input,
            min_angle_noise=None,
            max_angle_noise=None,
            contrast_signal=[1.0, 1.0],
            contrast_noise=[0.0, 0.0],
        )
        u_in = _compute_input_potentials(params, model, r_1, r_2)

        for i, c in enumerate(contrasts):

            model.set_input_scale(0, c)
            model.set_input_scale(1, c)

            g0, u0 = model(u_in)
            res["g0"][i] = g0
            res["u0"][i] = u0
            res["r0"][i] = model.f(u0)

            model.set_input_scale(0, c)
            model.set_input_scale(1, 0.0)
            g0, u0 = model(u_in)
            res["g0_1"][i] = g0
            res["u0_1"][i] = u0
            res["r0_1"][i] = model.f(u0)

            model.set_input_scale(0, 0.0)
            model.set_input_scale(1, c)
            g0, u0 = model(u_in)
            res["g0_2"][i] = g0
            res["u0_2"][i] = u0
            res["r0_2"][i] = model.f(u0)

    # determine limit potentials for infinitely weak input
    model.set_input_scale(0, 0.0)
    model.set_input_scale(1, 0.0)
    _, u0 = model(u_in)
    res["u0_null"] = u0
    res["r0_null"] = model.f(u0)

    # determine limit potentials for infinitely strong input
    model.set_input_scale(0, 1e5)
    model.set_input_scale(1, 0.0)
    _, u0 = model(u_in)
    res["u0_1_inf"] = u0
    res["r0_1_inf"] = model.f(u0)

    model.set_input_scale(0, 0.0)
    model.set_input_scale(1, 1e5)
    _, u0 = model(u_in)
    res["u0_2_inf"] = u0
    res["r0_2_inf"] = model.f(u0)

    return utils.numpyfy_all_torch_tensors_in_dict(res)


if __name__ == "__main__":

    with open("fig7_params.json", "r") as f:
        params = json.load(f)

    np.random.seed(params["seed"])
    torch.manual_seed(params["seed"])

    model = FeedForwardCell(params["in_features"], params["out_features"])
    model.gc = params["gc"]
    model.gL0 = params["gL0"]
    model.gLd = torch.DoubleTensor(params["gLd"])

    model.load_state_dict(torch.load("../data/fig7/model_final.torch"))

    res_VgT = sim(params, model, angle_1=65, angle_2=50)

    res = {
        "res_VgT": res_VgT,
    }

    with open(f"../data/fig8/res.pkl", "wb") as f:
        pickle.dump(res, f)
