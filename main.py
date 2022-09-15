from mpi4py import MPI
import utils as ut
import DMFT_iteration as it

if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    ut.set_seed(comm)

    parameters = ut.init_parameters()
    parameters["m_0"] = 0.2
    parameters["alpha"] = 3
    parameters["b"] = 1.0
    parameters["lambd"] = 0.5
    parameters["dt"] = 0.1
    parameters["T"] = 50
    parameters["n_samples"] = 2000
    parameters["damping"] = 0.7
    parameters["n_iterations"] = 30

    ut.printMPI(f"Initialising...", comm)
    if rank == 0:
        it.init(parameters)

    n_iterations = int(parameters["n_iterations"])
    for iteration in range(n_iterations):
        ut.printMPI(f"Iteration {iteration+1} of {n_iterations}", comm)
        it.iterate(comm, iteration, parameters)
        comm.Barrier()
