import numpy as np
import sys
import os
import time
import yaml
from grid import grid
from disk import disk
from plotutils import plotutils
from fitsconversion import convert_to_fits
from raytrace_maps import raytrace_maps


model_name = 'demo'


# generate the spatial grid
sim_grid = grid(model_name)

# generate a model structure on that grid
print('making structure')
t0 = time.time()
sim_disk = disk(model_name, sim_grid)
print(time.time()-t0)

#print('plotting structures')
_ = plotutils(model_name)
sys.exit()

# raytrace out a set of channel maps
print('raytracing')
t0 = time.time()
ch_maps = raytrace_maps(model_name)
print(time.time()-t0)
