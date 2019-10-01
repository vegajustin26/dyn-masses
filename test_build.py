import numpy as np
import sys
import os
import yaml
import build_structure as model
import matplotlib.pyplot as plt


def print_radmc_inp(incl_dust=0, incl_lines=1, incl_freefree=0, 
                    scattering='Isotropic', binary=False, 
                    camera_tracemode='image', lines_mode='LTE'):

    filename = 'radmc3d.inp'
    f = open(filename, 'w')
    f.write('incl_dust=%d\n' % incl_dust)
    f.write('incl_lines=%d\n' % incl_lines)
    f.write('incl_freefree=%d\n' % incl_freefree)
    if scattering == 'None':
        f.write('scattering_mode_max=%d\n' % 0)
    elif scattering == 'Isotropic':
        f.write('scattering_mode_max=%d\n' % 1)
        f.write('nphot_scat=2000000\n')
    elif scattering == 'HG':
        f.write('scattering_mode_max=%d\n' % 2)
        f.write('nphot_scat=10000000\n')
    elif scattering == 'Mueller':
        f.write('scattering_mode_max=%d\n' % 3)
        f.write('nphot_scat=100000000\n')
    if binary:
        f.write('writeimage_unformatted = 1\n')
        f.write('rto_single = 1\n')
        f.write('rto_style = 3\n')
    else:
        f.write('rto_style = 1\n')

    if camera_tracemode=='image':
        f.write('camera_tracemode = 1\n')
    elif camera_tracemode=='tau':
        f.write('camera_tracemode = -2\n')
    if lines_mode=='LTE':
        f.write('lines_mode = 1\n')

    f.close()



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


