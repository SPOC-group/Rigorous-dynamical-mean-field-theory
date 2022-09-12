import warnings
from numba import njit
import numpy as np
import loss_functions as lf
import math_tools as mt

warnings.filterwarnings("ignore")


@njit(fastmath=True)
def make_h(tilde_nu, m, batch, M_R, sq_mat, parameters):
    """Makes n_samples stochastic processes for the preactivation h up to time step T

    Args:
        tilde_nu (np.ndarray): [T] vector, effective regularisation
        m (np.ndarray): [T] vector, magnetisation
        batch (np.ndarray): [T, n_samples] matrix, samples selection variable
        M_R (np.ndarray): [T, T] matrix, memory kernel
        sq_mat (np.ndarray): [T, T] matrix, square root of covariance of the noise
        parameters (np.ndarray): parameter dictionary

    Returns:
        np.ndarray: [T, n_samples] matrix, stochastic processes for h
        np.ndarray: [n_samples] vector, optimal preactivations h_0
    """

    dt = parameters["dt"]
    T = int(parameters["T"])
    n_samples = int(parameters["n_samples"])

    h = np.ones((T, n_samples))

    h_start = np.random.normal(0, 1, n_samples)
    h_0 = np.random.normal(0, 1, n_samples)

    noise = mt.make_noise(sq_mat, parameters)

    h[0] = h_start
    for t in range(1, T):
        for sample in range(n_samples):
            h[t, sample] = (
                h[t - 1, sample]
                - dt * tilde_nu[t - 1] * h[t - 1, sample]
                - dt * lf.d_nu(h[t - 1, sample] + h_0[sample] * m[t - 1], h_0[sample]) * batch[t - 1, sample]
                + dt**2 * M_R[t - 1, : t - 1] @ h[: t - 1, sample]
                + dt * noise[t - 1, sample]
            )
    return h, h_0


@njit(fastmath=True)
def make_tilde_nu(hat_nu, delta_nu):
    """Make the effective regularisation

    Args:
        hat_nu (np.ndarray): [T] vector, explicit regularisation
        delta_nu (np.ndarray): [T] vector, effective correction to the regularisation

    Returns:
        np.ndarray: [T] vector, effective regularisation
    """

    return hat_nu + delta_nu


@njit(fastmath=True)
def make_m(hat_nu, mu, parameters):
    """Make the magnetisation

    Args:
        hat_nu (np.ndarray): [T] vector, explicit regularisation
        mu (np.ndarray): [T] vector, magnetisation drift
        parameters (dict): parameters dictionary

    Returns:
        np.ndarray: [T] vector, magnetisation
    """

    m_0 = parameters["m_0"]
    T = int(parameters["T"])
    dt = parameters["dt"]

    m = np.empty(T)

    m[0] = m_0
    for t in range(1, T):
        m[t] = m[t - 1] - dt * (hat_nu[t - 1] * m[t - 1] + mu[t - 1])
    return m


@njit(fastmath=True)
def make_h_tilde(h, h_0, m, parameters):
    """Compute "n_samples" samples of h_tilde, the preactivation shifted by the normalisation

    Args:
        h (np.ndarray): [T, n_samples] matrix, preactivation
        h_0 (np.ndarray): [n_samples] vector, optimal preactivation
        m (np.ndarray): [T] vector, magnetisation
        parameters (dict): parameters dictionary

    Returns:
        np.ndarray: shifted preactivation
    """

    n_samples = parameters["n_samples"]

    h_tilde = np.zeros_like(h)
    for sample in range(n_samples):
        h_tilde[:, sample] = h[:, sample] + h_0[sample] * m
    return h_tilde


@njit(fastmath=True)
def make_kernels(h_tilde, h_0, tilde_nu, batch, parameters):
    """Make the kernels delta_nu, hat_nu, mu, M_C, M_R

    Args:
        h_tilde (np.ndarray): [T, n_samples] matrix, shifted preactivation
        h_0 (np.ndarray): [n_samples] vector, optimal preactivation
        tilde_nu (np.ndarray): [T] vector, effective regularisation
        batch (np.ndarray): [T, n_samples] matrix, sample selection variable
        parameters (dict): parameters dictionary

    Returns:
        np.ndarray: [T] vector, effective correction to the regularisation delta_nu
        np.ndarray: [T] vector, explicit regularisation hat_nu
        np.ndarray: [T] vector, magnetisation drift mu
        np.ndarray: [T, T] matrix, covariance of the noise
        np.ndarray: [T, T] matrix, memory kernel
    """

    T = int(parameters["T"])
    dt = parameters["dt"]
    n_samples = int(parameters["n_samples"])
    alpha = parameters["alpha"]
    lambd = parameters["lambd"]
    b = parameters["b"]

    delta_nu = np.zeros(T)
    hat_nu = np.zeros(T)
    mu = np.zeros(T)
    M_C = np.zeros((T, T))
    M_R = np.zeros((T, T))

    for t1 in range(T):
        # Compute delta_nu
        delta_nu[t1] = alpha * np.mean(batch[t1] * lf.dd_nu(h_tilde[t1], h_0))

        # # SPHERICAL: Compute hat_nu
        # hat_nu[t1] = -alpha * np.mean(batch[t1] * h_tilde[t1] * d_nu(h_tilde[t1], h_0))

        # REGULARISED: Compute hat_nu
        hat_nu = np.zeros_like(hat_nu) + lambd

        # Compute mu
        mu[t1] = alpha * np.mean(batch[t1] * h_0 * lf.d_nu(h_tilde[t1], h_0))

        T_eq = np.ones((T - t1, n_samples))
        for t2 in range(t1, T):
            # Compute M_C
            M_C[t1, t2] = alpha * np.mean(lf.d_nu(h_tilde[t1], h_0) * lf.d_nu(h_tilde[t2], h_0) * batch[t1] * batch[t2])
            M_C[t2, t1] = M_C[t1, t2]

            # Compute M_R
            if t2 > t1:
                for sample in range(n_samples):
                    T_eq[t2 - t1, sample] = (
                        T_eq[t2 - t1 - 1, sample]
                        - dt * tilde_nu[t2 - 1] * T_eq[t2 - t1 - 1, sample]
                        - dt * batch[t2 - 1, sample] * lf.dd_nu(h_tilde[t2 - 1, sample], h_0[sample]) * (T_eq[t2 - t1 - 1, sample] - mt.delta(t2 - 1, t1))
                        + dt**2 * M_R[t2 - 1, t1 : t2 - 1] @ T_eq[: t2 - t1 - 1, sample]
                    )
                M_R[t2, t1] = alpha / b * np.mean(lf.dd_nu(h_tilde[t2], h_0) * T_eq[t2 - t1] * batch[t2])

    return delta_nu, hat_nu, mu, M_C, M_R
