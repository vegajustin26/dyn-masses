import os
import sys
import numpy as np
from mk_FITScube import mk_FITScube
from vis_sample import vis_sample
from vis_sample.file_handling import import_data_uvfits
from astropy.io import fits

# template uvfits file
template = 'fake_data/rich_io.sim.config5.30min.noisy.specbin.uvfits'
dvis = import_data_uvfits(template)

# parameters: theta = [inc, PA, mstar, Tb0, Tbq, r_max, vsys, dx, dy]
theta = np.array([30., 40., 1.75, 150., -0.5, 300., 0.0, 0.0, 0.0])
FOV = 8.0
dist = 150.
Npix = 256
Tbmax = 500.
restfreq = 230.538e9
fixed = FOV, dist, Npix, Tbmax, restfreq

### - extract the velocities from the template file
dat = fits.open(template)
hdr = dat[0].header
freq0 = hdr['CRVAL4']
indx0 = hdr['CRPIX4']
nchan = hdr['NAXIS4']
dfreq = hdr['CDELT4']
freqs = freq0 + (np.arange(nchan) - indx0 + 1) * dfreq
vel = 2.9979245800000e10 * (1. - freqs / restfreq) / 100.

### - compute a model cube (SkyImage object)
foo = mk_FITScube(inc=theta[0], PA=theta[1], mstar=theta[2], FOV=FOV, 
                  dist=dist, Npix=Npix, Tb0=theta[3], Tbq=theta[4], 
                  r_max=theta[5], vsys=theta[6], Tbmax=Tbmax, 
                  restfreq=restfreq, vel=vel)

### - sample it on the (u,v) spacings of your choice: NOISE FREE
### save it as a NEW "datafile"
vis_sample(imagefile=foo, uvfile=template, mu_RA=theta[7], 
           mu_DEC=theta[8], mod_interp=False, 
           outfile='fake_data/rich_io_nf.uvfits')

