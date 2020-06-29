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


#model_name = ['grid1_A', 'grid1_B', 'grid1_C', 'grid1_D', 'grid1_E', 'grid1_F']
model_name = ['phys4_i40']

for i in range(len(model_name)):
    # generate disk structure
    di = sim_disk(model_name[i])
    
    # make spherical and cylindrical plots
    _ = plotutils(model_name[i])	
    di = sim_disk(model_name[i], writestruct=False, cyl=True)
    _ = plotutils(model_name[i], struct=di, cyl=True)

    # raytrace out a set of channel maps
    t0 = time.time()
    ch_maps = raytrace_maps(model_name[i])
    print(time.time()-t0)
