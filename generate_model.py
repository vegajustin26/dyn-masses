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
gr = grid(model_name)

# generate a model structure on that grid
print('making structure')
t0 = time.time()
di = disk(model_name, gr, writestruct=False)
print(time.time()-t0)


#T = di.temperature(r=1.496e15, z=np.array([0., 1.496e14]), **sim_disk.T_args)


#print(di.scaleheight(r=1.496e15, T=T) / 1.496e15)

#print('plotting structures')
_ = plotutils(model_name, struct=di)
sys.exit()

# raytrace out a set of channel maps
print('raytracing')
t0 = time.time()
ch_maps = raytrace_maps(model_name)
print(time.time()-t0)
