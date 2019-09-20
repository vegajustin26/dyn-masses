import numpy as np
import constants as const

"""


"""

def grid_setup(nr, nt, na, rin, rout):


    ### RADIAL GRID

    # define grid boundaries (cgs units)
    r_in  = rin * const.au.cgs
    r_out = rout * const.au.cgs
 
    # define cell walls and centers
    """ not sure about the centers defined in this way... """
    r_walls = np.logspace(np.log10(r_in), np.log10(r_out), nr+1)
    r_centers = 0.5 * (r_walls[:-1] + r_walls[1:])

    
    ### THETA GRID (= zenith angle)
    
    # set a slight offset bound at pole
    ped = 0.1

    # define cell walls and centers
    """ not sure about the centers defined in this way... """
    t_walls = 0.5*np.pi + ped - \
              np.logspace(np.log10(ped), np.log10(0.5*np.pi+ped), nt+1)[::-1]
    t_centers = 0.5 * (t_walls[:-1] + t_walls[1:])


    ### PHI GRID (= azimuth)

    # define cell walls and centers
    az_walls = np.linspace(0, 2*np.pi, na+1)
    az_centers = 0.5 * (az_walls[:-1] + az_walls[1:])


    ### OUTPUT

    # open file
    f = open('amr_grid.inp', 'w')

    # file header
    f.write('1\n')	# format code
    f.write('0\n')	# regular grid
    f.write('100\n')	# spherical coordinate system
    f.write('0\n')	# no grid info written to file (as recommended)
    f.write('1 1 1\n')	# 
    f.write('%d %d %d\n' % (nr, nt, na))

    # write wall coordinates to file
    for r in r_walls: f.write('%.9e\n' % r)
    for t in t_walls: f.write('%.9e\n' % t)
    for az in az_walls: f.write('%.9e\n' % az)

    # close file
    f.close()

    # return the cell centers to user
    return r_centers, t_centers, az_centers
