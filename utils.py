from numba.typed import Dict
from numba import types
import numpy as np
import pickle


def set_seed(comm):
    # Comment on why and how
    rank = comm.Get_rank()
    np.random.seed(rank)


def printMPI(text, comm):
    rank = comm.Get_rank()
    if rank == 0:
        print(text)


def init_parameters():
    parameters = Dict.empty(
        key_type=types.unicode_type,
        value_type=types.float64,
    )
    return parameters


def load_data(parameters, iteration):
    m_0 = parameters["m_0"]
    alpha = parameters["alpha"]
    b = parameters["b"]
    lambd = parameters["lambd"]

    if iteration == 0:
        return pickle.load(open(f"data/initM{m_0}A{alpha}B{b}L{lambd}.pkl", "rb"))
    return pickle.load(open(f"data/iterM{m_0}A{alpha}B{b}L{lambd}I{iteration}.pkl", "rb"))


def save_data(store_data, parameters, iteration):
    m_0 = parameters["m_0"]
    alpha = parameters["alpha"]
    b = parameters["b"]
    lambd = parameters["lambd"]
    T = int(parameters["T"])
    dt = parameters["dt"]

    m, M_R, M_C, tilde_nu, hat_nu, delta_nu, mu = store_data
    dataSave = {
        "T": T,
        "dt": dt,
        "tilde_nu": tilde_nu,
        "hat_nu": hat_nu,
        "delta_nu": delta_nu,
        "mu": mu,
        "M_C": M_C,
        "M_R": M_R,
        "m": m,
    }

    if iteration == 0:
        pickle.dump(dataSave, open(f"data/initM{m_0}A{alpha}B{b}L{lambd}.pkl", "wb"))
    else:
        pickle.dump(dataSave, open(f"data/iterM{m_0}A{alpha}B{b}L{lambd}I{iteration}.pkl", "wb"))
    dataSave.clear()
