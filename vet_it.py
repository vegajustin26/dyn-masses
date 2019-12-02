from scipy.interpolate import griddata
from grid import grid
from disk import disk
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import yaml
import scipy.constants as sc
from scipy import integrate



# generate the spherical coordinate grid and associated control files
sim_grid = grid('testrich_hyd', writegrid=True)
grid_R = sim_grid.r_centers
grid_T = sim_grid.t_centers
grid_P = sim_grid.p_centers
grid_nR = sim_grid.nr
grid_nT = sim_grid.nt
grid_nP = sim_grid.np

print(grid_nR, grid_nT, grid_nP)




# generate a structure model on a cylindrical polar grid
hyd_disk = disk('testrich_hyd')

# those structure grid points in spherical polar coordinates
str_R = hyd_disk.polar_r
str_THETA = hyd_disk.polar_t

# grab density, temperature, abundance, velocity structures
rho_g = hyd_disk.density_g
T = hyd_disk.temperature

