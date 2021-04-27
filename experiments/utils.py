import numpy as np


def low_pass_filter(x, tau, dt):

    assert tau > dt
    P = np.exp(-dt / tau)
    x_filtered = np.empty_like(x)
    x_filtered[0] = x[0]
    for i in range(1, len(x)):
        x_filtered[i] = P * x_filtered[i - 1] + (1.0 - P) * x[i]

    return x_filtered


def moving_average(a, window_size):
    assert len(a) > window_size
    new_a = np.empty_like(a, dtype=np.double)
    for i in range(len(a) - window_size):
        new_a[i] = np.mean(a[i : i + window_size], axis=0)
    return new_a[:i]


def gaussian_density(x, mu, sigma):
    return (
        1.0
        / np.sqrt(2.0 * np.pi * sigma ** 2)
        * np.exp(-((x - mu) ** 2) / (2.0 * sigma ** 2))
    )


def numpyfy_all_torch_tensors_in_dict(d):
    for key in d:
        try:
            d[key] = d[key].detach().numpy()
        except AttributeError:
            pass
    return d
