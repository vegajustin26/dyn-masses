import numpy as np
from astropy.io import fits
from mk_FITScube import mk_FITScube
from vis_sample import vis_sample

def lnprob(theta, args):

    # constants
    CC = 2.9979245800000e10
    KK = 1.3807e-16
    restfreq = 230.538e9

    # parse input arguments
    dvis, gcf, corr = args

    # get out velocities
    freqs = dvis.freqs
    vel = CC * (1. - freqs / restfreq) / 100.

    # generate a model FITS cube
    # presumes theta = [inc, PA, mstar, Tb0, Tbq, r_max, V_sys]
    foo = mk_FITScube(inc=theta[0], PA=theta[1], mstar=theta[2], 
                       FOV=12.0, dist=150., Npix=512, 
                       Tb0=theta[3], Tbq=theta[4], r_max=theta[5], 
                       vsys=theta[6], Tbmax=500., restfreq=restfreq, 
                       vel=vel, outfile='model.fits')

    # now sample the FT of the model onto the observed (u,v) points
    modl_vis = vis_sample(imagefile='model.fits', 
                          gcf_holder=gcf, corr_cache=corr)

    return 0 
