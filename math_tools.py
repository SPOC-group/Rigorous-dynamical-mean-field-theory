from numba import njit
import numpy as np

REG = 1e-8


@njit(fastmath=True)
def delta(x, y):
    if x == y:
        return 1
    return 0


def sq_matrix(matrix):
    val, vec = np.linalg.eigh(matrix + REG * np.eye(matrix.shape[0]))
    return vec @ np.diag(np.sqrt(val))


@njit(fastmath=True)
def make_noise(sq_mat, T, n_samples):
    return sq_mat @ np.random.normal(0, 1, (T, n_samples))
