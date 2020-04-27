import os
import sys
import time
import numpy as np
from astropy.io import fits
from mk_FITScube import mk_FITScube
from vis_sample import vis_sample
from vis_sample.file_handling import import_data_uvfits
from lnprob import lnprob


### - parse and package the DATA
# load file
data_file = 'testrich2.uvfits'
data_vis = import_data_uvfits(data_file)
print(np.shape(data_vis.VV.T))

### - package the parameters
theta = np.array([30., 40., 2.0, 150., -0.5, 400., 0.0])


### - compute 1 model to set up GCF, corr caches
foo = mk_FITScube(inc=theta[0], PA=theta[1], mstar=theta[2], 
                  FOV=12.0, dist=150., Npix=512,
                  Tb0=theta[3], Tbq=theta[4], r_max=theta[5],
                  vsys=theta[6], Tbmax=500., restfreq=230.538e9,
                  datafile=data_file, outfile='model.fits')

t0 = time.time()
test_vis, GCF, corr = vis_sample(imagefile='model.fits', uvfile=data_file, 
                                 return_gcf=True, return_corr_cache=True)
t1 = time.time()
rep_vis = vis_sample(imagefile='model.fits', gcf_holder=GCF, corr_cache=corr)

sys.exit()


# package the arguments
args = data_vis, GCF, corr
t0 = time.time()
for i in range(100):
    test_lnprob = lnprob(theta, args)

print((time.time-t0) / 100.)




sys.exit()

# convert to appropriate units


# parameters
inc = 30.
PA = 45.
mstar = 2.0
FOV = 12.
dist = 150.
Npix = 512
Tb0 = 150.
Tbq = -0.5
r_max = 400.
vsys = 4.


foo = mk_FITScube(inc=inc, PA=PA, mstar=mstar, FOV=FOV, dist=dist, Npix=Npix,
                  Tb0=Tb0, Tbq=Tbq, r_max=r_max, vsys=vsys, Tbmax=300., 
                  datafile='testrich2.uvfits', outfile='testmodel.fits')


vis_sample(imagefile='testmodel.fits', uvfile='testrich2.uvfits', 
           outfile='testmodel1.uvfits')


