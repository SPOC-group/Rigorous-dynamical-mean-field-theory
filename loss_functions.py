from numba import njit
import numpy as np


@njit(fastmath=True)
def nu(h: float, h_0: float) -> float:
    """Loss function

    Args:
        h (float): preactivation
        h_0 (float): teacher preactivation

    Returns:
        float: loss value
    """

    return (h - h_0) ** 2 / 2


@njit(fastmath=True)
def d_nu(h: float, h_0: float) -> float:
    """Derivative of loss function

    Args:
        h (float): preactivation
        h_0 (float): teacher preactivation

    Returns:
        float: value of derivative of loss
    """

    return h - h_0


@njit(fastmath=True)
def dd_nu(h: float, h_0: float) -> float:
    """Second erivative of loss function

    Args:
        h (float): preactivation
        h_0 (float): teacher preactivation

    Returns:
        float: value of second derivative of loss
    """

    return 1


@njit(fastmath=True)
def make_batch(parameters):
    """Makes batch selection variable for SGD

    Args:
        parameters (dict): parameter dictionary

    Returns:
        np.ndarray: [T, n_samples] matrix, sample selection variable
    """

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
