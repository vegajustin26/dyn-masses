import numpy as np
import sys
import os
import yaml
import build_structure as model


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



