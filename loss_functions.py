from numba import njit
import numpy as np


@njit(fastmath=True)
def nu(h, h_0):
    return (h - h_0) ** 2 / 2


@njit(fastmath=True)
def d_nu(h, h_0):
    return h - h_0


@njit(fastmath=True)
def dd_nu(h, h_0):
    return 1


@njit(fastmath=True)
def make_batch(parameters):
    T = int(parameters["T"])
    n_samples = int(parameters["n_samples"])
    b = parameters["b"]

    batch = (np.random.rand(T, n_samples)) - b
    for t in range(T):
        for sample in range(n_samples):
            if batch[t, sample] < 0:
                batch[t, sample] = 1 / b
            else:
                batch[t, sample] = 0
    return batch
