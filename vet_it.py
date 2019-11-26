from scipy.interpolate import griddata
from grid import grid
from disk import disk
import numpy as np
import sys
import os


# generate the spherical coordinate grid and associated control files
sim_grid = grid('testrich', writegrid=True)





sys.exit()
temp = disk('testrich')


_ = temp.plot_rotation(contourf_kwargs=dict(levels=np.arange(0, 10, .1)))
_.savefig('test.png')
