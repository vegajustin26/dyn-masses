from scipy.interpolate import griddata
from grid import grid
from disk import disk
import numpy as np
import sys
import os


# generate the spherical coordinate grid and associated control files
sim_grid = grid('testrich', writegrid=True)

# generate a structure model on a cylindrical grid
sim_disk = disk('testrich')


sys.exit()


_ = temp.plot_rotation(contourf_kwargs=dict(levels=np.arange(0, 10, .1)))
_.savefig('test.png')
