import json
import pickle

import numpy as np
import torch

from dopp import FeedForwardCell, FeedForwardCurrentCell
from sim import sim  # noqa: F402

if __name__ == "__main__":

    with open("fig7_params.json", "r") as f:
        params = json.load(f)

    torch.manual_seed(params["seed"])
    np.random.seed(params["seed"])
    if params["current-based"]:
        model = FeedForwardCurrentCell(params["in_features"], params["out_features"])
        postfix = "_current"
    else:
        model = FeedForwardCell(params["in_features"], params["out_features"])
        postfix = ""

    torch.manual_seed(params["seed"])
    np.random.seed(params["seed"])

    model.load_state_dict(torch.load(f"../data/fig7/model_final{postfix}.torch"))

    # bimodal
    res_final = sim(
        params,
        model,
        n_trials=params["n_trials_test"],
        n_subtrials=params["n_subtrials_test"],
        min_angle=params["min_angle_test"],
        max_angle=params["max_angle_test"],
        min_angle_noise=params["min_angle_noise_test"],
        max_angle_noise=params["max_angle_noise_test"],
        contrast_signal=params["contrast_signal"],
        contrast_noise=params["contrast_noise"],
    )
    with open(f"../data/fig7/res_test_VT{postfix}.pkl", "wb") as f:
        pickle.dump(res_final, f)

    # only first modality
    res_final_1 = sim(
        params,
        model,
        contrast=[1.0, params["contrast_structured_noise"]],
        n_trials=params["n_trials_test"],
        n_subtrials=params["n_subtrials_test"],
        min_angle=params["min_angle_test"],
        max_angle=params["max_angle_test"],
        min_angle_noise=params["min_angle_noise_test"],
        max_angle_noise=params["max_angle_noise_test"],
        contrast_signal=params["contrast_signal"],
        contrast_noise=params["contrast_noise"],
    )
    with open(f"../data/fig7/res_test_V{postfix}.pkl", "wb") as f:
        pickle.dump(res_final_1, f)

    # only second modality
    res_final_2 = sim(
        params,
        model,
        contrast=[params["contrast_structured_noise"], 1.0],
        n_trials=params["n_trials_test"],
        n_subtrials=params["n_subtrials_test"],
        min_angle=params["min_angle_test"],
        max_angle=params["max_angle_test"],
        min_angle_noise=params["min_angle_noise_test"],
        max_angle_noise=params["max_angle_noise_test"],
        contrast_signal=params["contrast_signal"],
        contrast_noise=params["contrast_noise"],
    )
    with open(f"../data/fig7/res_test_T{postfix}.pkl", "wb") as f:
        pickle.dump(res_final_2, f)
