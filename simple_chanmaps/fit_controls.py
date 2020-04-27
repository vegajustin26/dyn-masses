import os
import sys
import numpy as np
from astropy.io import fits
from mk_FITScube import mk_FITScube
from vis_sample import vis_sample


### - parse and package the DATA
# load file
data_file = 'testrich2.uvfits'
dat = fits.open(data_file)
data = dat[0].data
dhd = dat[0].header

# extract frequencies
freq0, indx0, nchan = dhd['CRVAL4'], dhd['CRPIX4'], dhd['NAXIS4']
dfreq = dhd['CDELT4']
freq = freq0 + (np.arange(nchan) - indx0 + 1) * dfreq
rfreq = freq[np.int(indx0)]	# approximate mean frequency of cube

# extract (u,v) spacings

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


