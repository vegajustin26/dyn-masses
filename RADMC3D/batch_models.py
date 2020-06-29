import numpy as np
import sys
import os
import time
import yaml
from sim_grid import sim_grid
from sim_disk import sim_disk
from plotutils import plotutils
from fitsconversion import convert_to_fits
from raytrace_maps import raytrace_maps


model_name = ['grid1_A', 'grid1_B', 'grid1_C', 'grid1_D', 'grid1_E', 'grid1_F']
model_name = ['grid1_F']

#for i in range(len(model_name)):
#    t0 = time.time()
#    di = sim_disk(model_name[i])
#    print(model_name[i], time.time()-t0)

#    _ = plotutils(model_name[i])	

#sys.exit()

# raytrace out a set of channel maps
for i in range(len(model_name)):
    t0 = time.time()
    ch_maps = raytrace_maps(model_name[i])
    print(time.time()-t0)
