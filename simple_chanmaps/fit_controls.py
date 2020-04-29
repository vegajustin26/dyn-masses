import os
import sys
import time
import numpy as np
from mk_FITScube import mk_FITScube
from vis_sample import vis_sample
from vis_sample.file_handling import import_data_uvfits
from lnprob import lnprob


### - parse and package the DATA
# load file
data_file = 'fake_data/rich_io.sim.config5.30min.noisy.specbin.uvfits'
dvis = import_data_uvfits(data_file)

### - compute 1 model to set up GCF, corr caches
theta = np.array([30., 40., 2.0, 150., -0.5, 400., 0.0, 0.0, 0.0])
foo = mk_FITScube(inc=theta[0], PA=theta[1], mstar=theta[2], 
                  FOV=12.0, dist=150., Npix=512,
                  Tb0=theta[3], Tbq=theta[4], r_max=theta[5],
                  vsys=theta[6], Tbmax=500., restfreq=230.538e9,
                  datafile=data_file, outfile='model.fits')

test_vis, gcf, corr = vis_sample(imagefile='model.fits', uvfile=data_file, 
                                 return_gcf=True, return_corr_cache=True, 
                                 mod_interp=False)

### - package data and supplementary information
args = dvis, gcf, corr
