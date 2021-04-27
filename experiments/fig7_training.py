import json
import pickle

import numpy as np
import torch

from dopp import FeedForwardCell, FeedForwardCurrentCell
from sim import sim  # noqa: F402

if __name__ == "__main__":

    with open("fig7_params.json", "r") as f:
        params = json.load(f)

    assert params["in_features"][0] == params["in_features"][1]

    torch.manual_seed(params["seed"])
    np.random.seed(params["seed"])

    if params["current-based"]:
        model = FeedForwardCurrentCell(params["in_features"], params["out_features"])
        postfix = "_current"
        params["lr"] *= 2.5
    else:
        model = FeedForwardCell(params["in_features"], params["out_features"])
        postfix = ""

    torch.save(
        model.state_dict(), f"../data/fig7/model_initial{postfix}.torch",
    )

    torch.manual_seed(params["seed"])
    np.random.seed(params["seed"])
    res_train = sim(
        params,
        model,
        n_trials=params["n_trials_train"],
        min_angle=params["min_angle_train"],
        max_angle=params["max_angle_train"],
        contrast_signal=params["contrast_signal"],
        contrast_noise=params["contrast_noise"],
        ratio_VT_V_T_trials=params["ratio_VT_V_T_trials"],
        apply_grad=True,
    )

    torch.save(model.state_dict(), f"../data/fig7/model_final{postfix}.torch")

    with open(f"../data/fig7/res_train{postfix}.pkl", "wb") as f:
        pickle.dump(res_train, f)
