import numpy as np
import sys
import os
import yaml
import build_structure as model
import matplotlib.pyplot as plt
from fitsconversion import convert_to_fits
from look_utils import read_Tgas, read_nmol


# parameter file and bookkeeping setup
modelname = 'testabund'
if not os.path.exists(modelname):
    os.makedirs(modelname)


# set up the spatial grid and RADMC3D config files
grid = model.Grid(modelname)


# build a model structure 
diskmodel = model.DiskModel(modelname)
diskmodel.write_Model(grid)


# extract additional parameters for synthetic observations
conf = open(modelname+'.yaml')
pars = yaml.load(conf)
conf.close()


# look at the gas temperature structure
AU = 1.496e13
Tgas = (read_Tgas(grid, fname=modelname+'/gas_temperature.inp'))
levs = np.linspace(5, 100, 24)
#plt.contour(grid.r_centers/1.496e13, 0.5*np.pi-grid.theta_centers, Tgas,
#            levs, colors='k')
plt.contourf(0.5*np.pi-grid.theta_centers, grid.r_centers/AU, Tgas, levs)
plt.yscale('log')
plt.show()


sys.exit()

# look at the gas temperature structure
lognco = np.log10(read_nmol(grid, fname=modelname+'/numberdens_co.inp'))
levs = np.linspace(-8, 12, 24)

contours = plt.contour(grid.r_centers/1.496e13, 0.5*np.pi-grid.theta_centers, 
                       lognco, levs, colors='k')
plt.clabel(contours, colors = 'k', fmt = '%2.1f', fontsize=12)
plt.contourf(grid.r_centers/1.496e13, 0.5*np.pi-grid.theta_centers, lognco,
             levs)
plt.xscale('log')
plt.show()


sys.exit()


# raytracing
os.chdir(modelname)
os.system('rm -rf image.out')
os.system("radmc3d image incl %.2f posang %.2f " % \
          (pars["outputs"]["incl"], pars["outputs"]["PA"]) + \
          "npix %d iline %d " % \
          (pars["outputs"]["npix"], pars["setup"]["transition"]) + \
          "widthkms %.2f linenlam %d setthreads 4" % \
          (pars["outputs"]["widthkms"], pars["outputs"]["nchan"]))


# make a FITS file
outfile = modelname+'_'+pars["setup"]["molecule"]+'.fits'
os.system('rm '+outfile)
convert_to_fits('image.out', outfile, pars["outputs"]["dpc"], 
                RA=pars["outputs"]["RA"], DEC=pars["outputs"]["DEC"])
