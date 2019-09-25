import numpy as np
import sys
import os
import yaml
import build_structure as model
import matplotlib.pyplot as plt


# load setup file and parameters
setup_file = 'testmodel.yaml'
conf = open(setup_file)
config = yaml.load(conf)
conf.close()

# set up the spatial grid
grid = config["grid"]
nr = grid["nr"]
ntheta = grid["ntheta"]
nphi = grid["nphi"]
r_in = grid["r_in"]
r_out = grid["r_out"]
grid = model.Grid(nr, ntheta, nphi, r_in, r_out)

# create a model structure and write the RADMC3D input files
diskmodel = model.DiskModel(setup_file)
diskmodel.write_Model(grid)


AU = 1.49597871e13
Msun = 1.98847542e33
mu_gas = 2.37
m_H = 1.67353284e-24
G = 6.67408e-8
kB = 1.38064852e-16
PI = np.pi


rcyl = grid.r_centers * np.sin(grid.theta_centers)
z = grid.r_centers * np.cos(grid.theta_centers)

dgrid = diskmodel.rho_g(rcyl, z)

print(np.shape(dgrid))
