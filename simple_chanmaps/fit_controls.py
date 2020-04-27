import os
import sys
import numpy as np
from mk_FITScube import mk_FITScube
from vis_sample import vis_sample


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


