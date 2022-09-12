from numba import njit
import numpy as np

REG = 1e-8


@njit(fastmath=True)
def delta(x: int, y: int) -> int:
    """Kronecker delta function: returns 1 if the inputs are the same, 0 if otherwise

    Args:
        x (int): first input
        y (int): second input

    Returns:
        int: output
    """

    if x == y:
        return 1
    return 0


def sq_matrix(matrix: np.ndarray) -> np.ndarray:
    """Compute the square root of a matrix

    Args:
        matrix (np.ndarray): input matrix

    Returns:
        np.ndarray: output matrix
    """

    val, vec = np.linalg.eigh(matrix + REG * np.eye(matrix.shape[0]))
    return vec @ np.diag(np.sqrt(val))


@njit(fastmath=True)
def make_noise(sq_mat: np.ndarray, parameters) -> np.ndarray:
    """Generate n_samples Gaussian stochastic processes of lenght T

    Args:
        sq_mat (np.ndarray): [T, T] matrix, square root of the covariance
        parameters (dict): parameter dictionary

    Returns:
        np.ndarray: [T, n_samples] matrix, stochastic processes
    """

    T = int(parameters["T"])
    n_samples = int(parameters["n_samples"])

    return sq_mat @ np.random.normal(0, 1, (T, n_samples))
