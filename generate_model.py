import numpy as np
import sys
import os
import time
from grid import grid
from disk import disk


model_name = 'demo'

# generate the spatial grid
sim_grid = grid(model_name)


# generate a model structure on that grid
sim_disk = disk(model_name, sim_grid)



#mdir = 'testmodel_me/'

# recover spatial grid
#Rwalls = np.loadtxt(mdir+'amr_grid.inp', skiprows=6, max_rows=257)
#Twalls = np.loadtxt(mdir+'amr_grid.inp', skiprows=263, max_rows=513)
#Rgrid  = 0.5*(Rwalls[:-1] + Rwalls[1:])
#Tgrid  = 0.5*(Twalls[:-1] + Twalls[1:])


# recover density structure
#rho_in = np.loadtxt(mdir+'gas_density.inp', skiprows=2)
#i_rho_gd = np.reshape(rho_in, (512, 256))
