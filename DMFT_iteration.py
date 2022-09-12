from numba import njit
import numpy as np
import loss_functions as lf
import math_tools as mt
import DMFT_equations as eq
import utils as ut


@njit(fastmath=True)
def step(parameters, m, M_R, tilde_nu, sq_mat):
    # Make batch selection variable
    batch = lf.make_batch(parameters)

    # Integrate h
    h, h_0 = eq.make_h(tilde_nu, m, batch, M_R, sq_mat, parameters)

    # Compute h_tilde
    h_tilde = eq.make_h_tilde(h, h_0, m, parameters)

    # Compute kernels
    delta_nu, hat_nu, mu, M_C, M_R = eq.make_kernels(h_tilde, h_0, tilde_nu, batch, parameters)
    return delta_nu, hat_nu, mu, M_C, M_R


def init(parameters, init_samples=256):
    m_0 = parameters["m_0"]
    T = int(parameters["T"])
    alpha = parameters["alpha"]
    lambd = parameters["lambd"]

    m = np.ones(T) * m_0

    M_R = np.zeros((T, T))
    M_C = np.ones((T, T)) * 0.01
    np.fill_diagonal(M_C, 0.1)

    h_0 = np.random.normal(0, 1, init_samples)
    h_tilde = np.random.normal(0, 1, init_samples) + m_0 * h_0

    delta_nu = alpha * np.mean(lf.dd_nu(h_tilde, h_0)) * np.ones(T)

    # # SPHERICAL: Compute hat_nu
    # hat_nu = -alpha * np.mean(h_tilde * d_nu(h_tilde, h_0)) * np.ones(T)

    # REGULARISED: Compute hat_nu
    hat_nu = lambd * np.ones(T)

    mu = alpha * np.mean(h_0 * lf.d_nu(h_tilde, h_0)) * np.ones(T)

    tilde_nu = hat_nu + delta_nu

    # Save the data
    store_data = m, M_R, M_C, tilde_nu, hat_nu, delta_nu, mu
    ut.save_data(store_data, parameters, iteration=0)


def iterate(comm, iteration, parameters):
    T = int(parameters["T"])
    damping = parameters["damping"]

    # Load data
    data = ut.load_data(parameters, iteration)

    # Unpack data
    hat_nu = data["hat_nu"]
    delta_nu = data["delta_nu"]
    tilde_nu = data["tilde_nu"]
    mu = data["mu"]
    dt = data["dt"]
    m = data["m"]
    T = data["T"]
    M_R = data["M_R"]
    M_C = data["M_C"]

    # Find sqrt of matrix
    sq_MC = mt.sq_matrix(M_C)

    # Compute new guess of the kernels
    delta_nu_new, hat_nu_new, mu_new, M_C_new, M_R_new = step(parameters, m, M_R, tilde_nu, sq_MC)

    # Setup comunication
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Create comunication buffer
    if rank == 0:
        delta_nu_comm = np.zeros((size, T))
        hat_nu_comm = np.zeros((size, T))
        mu_comm = np.zeros((size, T))
        M_C_comm = np.zeros((size, T, T))
        M_R_comm = np.zeros((size, T, T))

    # Send the new kernels
    comm.Gather(delta_nu_new, delta_nu_comm, root=0)
    comm.Gather(hat_nu_new, hat_nu_comm, root=0)
    comm.Gather(mu_new, mu_comm, root=0)
    comm.Gather(M_C_new, M_C_comm, root=0)
    comm.Gather(M_R_new, M_R_comm, root=0)

    # Average over the samples
    if rank == 0:
        delta_nu_new = delta_nu_comm.mean(axis=0)
        hat_nu_new = hat_nu_comm.mean(axis=0)
        mu_new = mu_comm.mean(axis=0)
        M_C_new = M_C_comm.mean(axis=0)
        M_R_new = M_R_comm.mean(axis=0)

        # Propose update to the kernels using damping
        delta_nu = delta_nu_new * damping + delta_nu * (1 - damping)
        hat_nu = hat_nu_new * damping + hat_nu * (1 - damping)
        M_C = M_C_new * damping + M_C * (1 - damping)
        M_R = M_R_new * damping + M_R * (1 - damping)
        mu = mu_new * damping + mu * (1 - damping)

        # Compute tilde_nu
        tilde_nu = eq.make_tilde_nu(hat_nu, delta_nu)

        # Compute m
        m = eq.make_m(hat_nu, mu, parameters)

        # Save the data
        store_data = m, M_R, M_C, tilde_nu, hat_nu, delta_nu, mu
        ut.save_data(store_data, parameters, iteration + 1)
