import utils_plot as utp
import matplotlib.pyplot as plt
import utils as ut


parameters = ut.init_parameters()
parameters["m_0"] = 0.2
parameters["alpha"] = 3
parameters["b"] = 0.5
parameters["lambd"] = 0.5

iter_start = 20
iter_end = 30

for iteration in range(iter_start, iter_end):
    utp.plot_magnetisation(parameters, iteration)


plt.legend()
plt.xlabel("Time")
plt.ylabel("Magnetisation")
plt.xscale("log")
plt.show()
