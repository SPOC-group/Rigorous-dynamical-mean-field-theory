# Rigorous-dynamical-mean-field-theory
Code for integrating the DMFT equations in "Rigorous dynamical mean field theory for stochastic gradient descent methods"

# Dependencies
- Numpy version 1.22.2
- Scipy version 1.7.1
- MPI4PY version 3.0.3
- Matplotlib version 3.5.1
- Pandas version 1.3.2

Note that MPI4PY also requires a MPI implementation. We refer to the MPI4PY documentation for further information.
A convinient way to setup the enviroment is by using conda and environment.yml

# How to integrate the equations
You can compute the DMFT fixed point iteration by running main.py. It's necessary to have a folder caller 'data'

# How view the results
A convenient plotting script is provided. Running plot.py will visualise the last steps of the iteration. We also provided a numerical simulation, which can be used to compare the results of the DMFT procedure. Any personal computer should reproduce comparison.png in approximately 1 minute using a single core.
