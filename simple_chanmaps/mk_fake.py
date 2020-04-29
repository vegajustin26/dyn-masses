import os
import sys
import numpy as np
from mk_FITScube import mk_FITScube

# outfile
outfile = 'rich_io.fits'

# parameters: theta = [inc, PA, mstar, Tb0, Tbq, r_max, vsys]
theta = np.array([30., 40., 1.75, 150., -0.5, 300., 0.0])

### - parse and package the DATA
data_file = 'testrich2.uvfits'		# template file

### - compute 1 model to set up GCF, corr caches
foo = mk_FITScube(inc=theta[0], PA=theta[1], mstar=theta[2], 
                  FOV=8.0, dist=150., Npix=256,
                  Tb0=theta[3], Tbq=theta[4], r_max=theta[5],
                  vsys=theta[6], Tbmax=500., restfreq=230.538e9,
                  datafile=data_file, outfile='fake_data/'+outfile)
