import contextlib
import sys

import numpy as np
import torch

import utils  # noqa: E402

sys.path.insert(0, "../include/")


# def feature_detector(angle, preferred_angle, min_rate, max_rate, concentration):
#     return min_rate + (max_rate - min_rate) * np.exp(
#         concentration * (np.cos(np.radians(angle - preferred_angle)) - 1.0)
#     )
def feature_detector(angle, preferred_angle, min_rate, max_rate, concentration):
    return min_rate + (max_rate - min_rate) * np.exp(
        -concentration
        / 2.0
        * (np.radians(angle.reshape(-1, 1) - preferred_angle.reshape(1, -1))) ** 2
    )


def orientation_generator(
    n_samples, min_angle, max_angle, *, batch_size=1,
):

    angles = torch.empty(n_samples, dtype=torch.double).uniform_(min_angle, max_angle)

    n_batches = n_samples // batch_size

    # yield batches of same size
    for i in range(n_batches):

        angle_target = angles[i * batch_size : (i + 1) * batch_size]
        yield angle_target

    # yield remaining samples that are not a full batch any more
    if n_batches * batch_size < n_samples:
        angle_target = angles[(i + 1) * batch_size :]
        yield angle_target


def subtrial_generator(n_subtrials, angle_target, sigma_1, sigma_2):

    for _ in range(n_subtrials):
        angle_1 = angle_target + torch.empty(
            len(angle_target), dtype=torch.double
        ).normal_(0.0, sigma_1)
        angle_2 = angle_target + torch.empty(
            len(angle_target), dtype=torch.double
        ).normal_(0.0, sigma_2)

        yield angle_1, angle_2


def _compute_preferred_angles(min_angle, max_angle, n):
    return torch.linspace(min_angle, max_angle, n, dtype=torch.double)


@contextlib.contextmanager
def possibly_no_grad(use_backprop):
    if use_backprop:
        yield
    else:
        with torch.no_grad():
            yield


def _compute_target_rates(params, angle_target):
    r_target = torch.empty(
        len(angle_target), params["out_features"], dtype=torch.double
    )
    r_target[angle_target >= params["decision_boundary"], 0] = params[
        "max_rate_outputs"
    ]
    r_target[angle_target >= params["decision_boundary"], 1] = params[
        "min_rate_outputs"
    ]
    r_target[angle_target < params["decision_boundary"], 0] = params["min_rate_outputs"]
    r_target[angle_target < params["decision_boundary"], 1] = params["max_rate_outputs"]

    return r_target


def _randomly_disable_one_modality(params, model, contrast, ratio_VT_V_T_trials):

    ratio_VT_V_T_trials = np.array(ratio_VT_V_T_trials) / np.sum(ratio_VT_V_T_trials)

    # by default, activate both modalities
    model.set_input_scale(0, contrast[0])
    model.set_input_scale(1, contrast[1])

    # disable either visual or tactile input
    x = np.random.rand()
    if x <= ratio_VT_V_T_trials[0]:  # VT trial
        pass
    elif ratio_VT_V_T_trials[0] < x and x <= sum(ratio_VT_V_T_trials[:2]):  # V trial
        model.set_input_scale(1, 0.0)
    else:
        model.set_input_scale(0, 0.0)  # T trial


def _generate_angle_noise(params, n, min_angle_noise, max_angle_noise):
    angle_1_noise = torch.empty(n, dtype=torch.double).uniform_(
        min_angle_noise, max_angle_noise
    )
    angle_2_noise = torch.empty(n, dtype=torch.double).uniform_(
        min_angle_noise, max_angle_noise
    )
    return angle_1_noise, angle_2_noise


def _compute_input_rates(
    params,
    angle_1,
    angle_2,
    preferred_angles,
    min_angle_noise,
    max_angle_noise,
    contrast_signal,
    contrast_noise,
):
    assert contrast_signal[0] >= 0.0
    assert contrast_signal[1] >= 0.0
    assert contrast_noise[0] >= 0.0
    assert contrast_noise[1] >= 0.0

    assert abs(contrast_signal[0] + contrast_noise[0] - 1.0) < 1e-9
    assert abs(contrast_signal[1] + contrast_noise[1] - 1.0) < 1e-9

    r_1 = feature_detector(
        angle_1,
        preferred_angles,
        params["min_rate_feature_detectors"],
        params["max_rate_feature_detectors"],
        params["concentration_feature_detectors"],
    )
    r_2 = feature_detector(
        angle_2,
        preferred_angles,
        params["min_rate_feature_detectors"],
        params["max_rate_feature_detectors"],
        params["concentration_feature_detectors"],
    )

    if min_angle_noise is not None:

        assert max_angle_noise is not None

        # generate distractors
        angle_1_noise, angle_2_noise = _generate_angle_noise(
            params, len(angle_1), min_angle_noise, max_angle_noise
        )

        r_1_noise = feature_detector(
            angle_1_noise,
            preferred_angles,
            params["min_rate_feature_detectors"],
            params["max_rate_feature_detectors"],
            params["concentration_feature_detectors"],
        )
        assert torch.all(r_1_noise >= params["min_rate_feature_detectors"])
        assert torch.all(r_1_noise <= params["max_rate_feature_detectors"])

        r_1 = contrast_signal[0] * r_1 + contrast_noise[0] * r_1_noise
        assert torch.all(r_1 >= params["min_rate_feature_detectors"])
        assert torch.all(r_1 <= params["max_rate_feature_detectors"])

        r_2_noise = feature_detector(
            angle_2_noise,
            preferred_angles,
            params["min_rate_feature_detectors"],
            params["max_rate_feature_detectors"],
            params["concentration_feature_detectors"],
        )
        assert torch.all(r_2_noise >= params["min_rate_feature_detectors"])
        assert torch.all(r_2_noise <= params["max_rate_feature_detectors"])
        assert torch.all(r_2 >= params["min_rate_feature_detectors"])
        assert torch.all(r_2 <= params["max_rate_feature_detectors"])

        r_2 = contrast_signal[1] * r_2 + contrast_noise[1] * r_2_noise

    return r_1, r_2


def _compute_input_potentials(params, model, r_1, r_2):
    if params["include_prior"]:
        r_prior = torch.ones(len(r_1), 1, dtype=torch.double)
        u_in = model.f_inv(torch.cat([r_1, r_2, r_prior], dim=1))
    else:
        u_in = model.f_inv(torch.cat([r_1, r_2], dim=1))

    assert torch.all(u_in <= model.EE)
    assert torch.all(u_in >= model.EI)

    return u_in


def sim(
    params,
    model,
    *,
    n_trials,
    n_subtrials=1,
    min_angle,
    max_angle,
    min_angle_noise=None,
    max_angle_noise=None,
    contrast=[1.0, 1.0],
    contrast_signal=[1.0, 1.0],
    contrast_noise=[0.0, 0.0],
    ratio_VT_V_T_trials=[1.0, 0.0, 0.0],
    apply_grad=False,
    use_backprop=False,
):

    assert len(contrast) == 2
    assert len(contrast_signal) == 2

    if params["include_prior"]:
        assert len(params["in_features"]) == 3
    else:
        assert len(params["in_features"]) == 2
    assert params["in_features"][0] == params["in_features"][1]

    if params["gL0"] is not None:
        model.gL0 = params["gL0"]
    else:
        model.gL0 = 0.0

    if params["gLd"] is not None:
        model.gLd = torch.DoubleTensor(params["gLd"])
    else:
        model.gLd = torch.zeros(model.n_dendrites)

    if params["gc"] is not None:
        model.gc = torch.DoubleTensor(params["gc"])
    else:
        model.gc = None

    model.lambda_e = params["lambda_e"]

    res = {}
    res["angle_target"] = torch.empty(n_trials, n_subtrials)
    res["angle_1"] = torch.empty(n_trials, n_subtrials)
    res["angle_2"] = torch.empty(n_trials, n_subtrials)
    res["angle_1_noise"] = torch.empty(n_trials, n_subtrials)
    res["angle_2_noise"] = torch.empty(n_trials, n_subtrials)
    res["angle_out"] = torch.empty(n_trials, n_subtrials)

    res["angle_MAP_mu"] = torch.empty(n_trials, n_subtrials)
    res["angle_MAP_sigma"] = torch.empty(n_trials, n_subtrials)
    res["angle_MAP_sample"] = torch.empty(n_trials, n_subtrials)

    res["angle_naive_mu"] = torch.empty(n_trials, n_subtrials)
    res["angle_naive_sample"] = torch.empty(n_trials, n_subtrials)

    res["g0"] = torch.empty(n_trials, n_subtrials, params["out_features"])
    res["gE"] = torch.empty(n_trials, n_subtrials, params["out_features"])
    res["gI"] = torch.empty(n_trials, n_subtrials, params["out_features"])
    res["u0"] = torch.empty(n_trials, n_subtrials, params["out_features"])

    res["energy"] = torch.empty(n_trials, n_subtrials, params["out_features"])
    res["loss"] = torch.empty(n_trials, n_subtrials, params["out_features"])

    preferred_angles_input = _compute_preferred_angles(
        params["min_preferred_angles_input"],
        params["max_preferred_angles_input"],
        params["in_features"][0],
    )

    if use_backprop:
        optimizer = torch.optim.SGD(
            model.parameters(), lr=model.lambda_e * params["lr"]
        )

    with possibly_no_grad(use_backprop):
        n_samples = 0
        for angle_target in orientation_generator(
            n_trials, min_angle, max_angle, batch_size=params["batch_size"],
        ):

            if n_samples % 1000 == 0:
                print(f"trial {n_samples}/{n_trials}", end="\r")

            for subtrial, (angle_1, angle_2) in enumerate(
                subtrial_generator(
                    n_subtrials, angle_target, params["sigma_1"], params["sigma_2"]
                )
            ):

                # compute target
                r_target = _compute_target_rates(params, angle_target)
                u0_target = model.f_inv(r_target)

                # compute input
                r_1, r_2 = _compute_input_rates(
                    params,
                    angle_1,
                    angle_2,
                    preferred_angles_input,
                    min_angle_noise,
                    max_angle_noise,
                    contrast_signal,
                    contrast_noise,
                )
                _randomly_disable_one_modality(
                    params, model, contrast, ratio_VT_V_T_trials
                )

                u_in = _compute_input_potentials(params, model, r_1, r_2)

                # forward pass
                g0, u0 = model(u_in)

                # update weights
                if apply_grad:
                    model.zero_grad()
                    if use_backprop:
                        model.energy().sum().backward()
                        optimizer.step()
                    else:
                        model.compute_grad_manual_target(u0_target, g0, u0, u_in)
                        model.apply_grad_weights(params["lambda_e"] * params["lr"])

                # stimulus data
                batch_slice = slice(n_samples, n_samples + len(angle_target))
                res["angle_target"][batch_slice, subtrial] = angle_target
                res["angle_1"][batch_slice, subtrial] = angle_1
                res["angle_2"][batch_slice, subtrial] = angle_2

                # Bayes-optimal results
                res["angle_MAP_mu"][batch_slice, subtrial] = (
                    1.0
                    / (1.0 / params["sigma_1"] ** 2 + 1.0 / params["sigma_2"] ** 2)
                    * (
                        angle_1 / params["sigma_1"] ** 2
                        + angle_2 / params["sigma_2"] ** 2
                    )
                )
                res["angle_MAP_sigma"][batch_slice, subtrial] = 1.0 / np.sqrt(
                    1.0 / params["sigma_1"] ** 2 + 1.0 / params["sigma_2"] ** 2
                )
                res["angle_MAP_sample"][batch_slice, subtrial] = (
                    res["angle_MAP_mu"][batch_slice, subtrial]
                    + res["angle_MAP_sigma"][batch_slice, subtrial]
                    * torch.empty(len(angle_target)).normal_()
                )

                # naive results
                res["angle_naive_mu"][batch_slice, subtrial] = (
                    1.0 / 2 * (angle_1 + angle_2)
                )
                res["angle_naive_sample"][batch_slice, subtrial] = (
                    res["angle_naive_mu"][batch_slice, subtrial]
                    + res["angle_MAP_sigma"][batch_slice, subtrial]
                    * torch.empty(len(angle_target)).normal_()
                )

                # model results
                if not params["current-based"]:
                    res["g0"][batch_slice, subtrial] = g0.clone()
                    gE = torch.empty(
                        len(angle_target), params["out_features"], model.n_dendrites
                    )
                    gI = torch.empty(
                        len(angle_target), params["out_features"], model.n_dendrites
                    )
                    for d in range(model.n_dendrites):
                        gE[:, :, d], gI[:, :, d] = model.compute_gEd_gId(u_in, d)
                    res["gE"][batch_slice, subtrial] = torch.mean(gE, dim=2)
                    res["gI"][batch_slice, subtrial] = torch.mean(gI, dim=2)
                res["u0"][batch_slice, subtrial] = u0.clone()
                res["energy"][batch_slice, subtrial] = model.energy_target(
                    u0_target, g0, u0
                )
                res["loss"][batch_slice, subtrial] = model.loss_target(
                    u0_target, g0, u0
                )

            n_samples += len(angle_target)

    print()

    return utils.numpyfy_all_torch_tensors_in_dict(res)
