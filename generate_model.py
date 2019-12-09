import numpy as np
import sys
import os
import time
import yaml
from grid import grid
from disk import disk
from plotutils import plotutils
from fitsconversion import convert_to_fits


model_name = 'demo'


_ = plotutils(model_name)







sys.exit()



# generate the spatial grid
sim_grid = grid(model_name)


# generate a model structure on that grid
print('making structure')
sim_disk = disk(model_name, sim_grid)


# raytracing
print('raytracing')
conf = open(model_name + ".yaml")
config = yaml.load(conf, Loader=yaml.FullLoader)
outpars = config["outputs"]
setpars = config["setup"]
conf.close()
os.chdir(model_name)
os.system('rm -rf image.out')
os.system("radmc3d image incl %.2f posang %.2f " % \
          (outpars["incl"], outpars["PA"]) + \
          "npix %d iline %d " % \
          (outpars["npix"], setpars["transition"]) + \
          "widthkms %.2f linenlam %d setthreads 4" % \
          (outpars["widthkms"], outpars["nchan"]))

# make a FITS file
outfile = model_name+'_'+setpars["molecule"]+'.fits'
os.system('rm '+outfile)
convert_to_fits('image.out', outfile, outpars["dpc"],
                RA=outpars["RA"], DEC=outpars["DEC"])





#mdir = 'testmodel_me/'

# recover spatial grid
#Rwalls = np.loadtxt(mdir+'amr_grid.inp', skiprows=6, max_rows=257)
#Twalls = np.loadtxt(mdir+'amr_grid.inp', skiprows=263, max_rows=513)
#Rgrid  = 0.5*(Rwalls[:-1] + Rwalls[1:])
#Tgrid  = 0.5*(Twalls[:-1] + Twalls[1:])


# recover density structure
#rho_in = np.loadtxt(mdir+'gas_density.inp', skiprows=2)
#i_rho_gd = np.reshape(rho_in, (512, 256))
