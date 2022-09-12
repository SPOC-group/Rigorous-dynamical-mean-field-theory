from numba.typed import Dict
from numba import types
import numpy as np
import pickle


def set_seed(comm):
    """Set the seed for the random number generator. Each process has seed equal to its rank.

    Args:
        comm : communication util
    """
    # Comment on why and how
    rank = comm.Get_rank()
    np.random.seed(rank)


def printMPI(text: str, comm):
    """Printing util. The string in text will only be printed by the process with rank 0

    Args:
        text (str): text to be printed
        comm : communication util
    """
    rank = comm.Get_rank()
    if rank == 0:
        print(text)


def init_parameters():
    """Initializes the parameters dictionary

    Returns:
        dict: parameter dictionary with explicit type
    """
    parameters = Dict.empty(
        key_type=types.unicode_type,
        value_type=types.float64,
    )
    return parameters


def load_data(parameters, iteration: int):
    """Load the data from file

    Args:
        parameters (dict): parameter dictionary
        iteration (int): iteration in the DMFT fixed point scheme. Iteration 0 is the starting ansatz.

    Returns:
        _type_: _description_
    """
    m_0 = parameters["m_0"]
    alpha = parameters["alpha"]
    b = parameters["b"]
    lambd = parameters["lambd"]

    if iteration == 0:
        return pickle.load(open(f"data/initM{m_0}A{alpha}B{b}L{lambd}.pkl", "rb"))
    return pickle.load(open(f"data/iterM{m_0}A{alpha}B{b}L{lambd}I{iteration}.pkl", "rb"))


def save_data(store_data: list, parameters, iteration: int):
    """Save the data to file

    Args:
        store_data (list): data to be saved in the file
        parameters (dict): parameter dictionary
        iteration (int): iteration in the DMFT fixed point scheme. Iteration 0 is the starting ansatz.
    """

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
