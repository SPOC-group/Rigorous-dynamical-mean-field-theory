import warnings
from numba import njit
import numpy as np
import loss_functions as lf
import math_tools as mt

warnings.filterwarnings("ignore")


@njit(fastmath=True)
def make_h(tilde_nu, m, batch, M_R, sq_mat, parameters):
    dt = parameters["dt"]
    T = int(parameters["T"])
    n_samples = int(parameters["n_samples"])

    h = np.ones((T, n_samples))

    h_start = np.random.normal(0, 1, n_samples)
    h_0 = np.random.normal(0, 1, n_samples)

    noise = mt.make_noise(sq_mat, T, n_samples)

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
    return hat_nu + delta_nu


@njit(fastmath=True)
def make_m(hat_nu, mu, parameters):
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
    n_samples = parameters["n_samples"]

    h_tilde = np.zeros_like(h)
    for sample in range(n_samples):
        h_tilde[:, sample] = h[:, sample] + h_0[sample] * m
    return h_tilde


@njit(fastmath=True)
def make_kernels(h_tilde, h_0, tilde_nu, batch, parameters):
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
