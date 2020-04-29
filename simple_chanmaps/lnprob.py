import numpy as np
from astropy.io import fits
from mk_FITScube import mk_FITScube
from vis_sample import vis_sample

def lnprob(theta, args):

    # constants
    CC = 2.9979245800000e10
    restfreq = 230.538e9

    # parse input arguments
    dvis, gcf, corr = args

    # convert to velocities
    vel = CC * (1. - dvis.freqs / restfreq) / 100.

    # generate a model FITS cube
    # presumes theta = [inc, PA, mstar, Tb0, Tbq, r_max, V_sys, dx, dy]
    foo = mk_FITScube(inc=theta[0], PA=theta[1], mstar=theta[2], 
                       FOV=12.0, dist=150., Npix=512, 
                       Tb0=theta[3], Tbq=theta[4], r_max=theta[5], 
                       vsys=theta[6], Tbmax=500., restfreq=restfreq, 
                       vel=vel, outfile='model.fits')

    # now sample the FT of the model onto the observed (u,v) points
    modl_vis = vis_sample(imagefile='model.fits', mu_ra=theta[7], 
                          mu_dec=theta[8], gcf_holder=gcf, 
                          corr_cache=corr, mod_interp=False)

    # compute the log-likelihood
    logL = -0.5 * np.sum(dvis.wgts * np.absolute(dvis.VV.T - modl_vis)**2)

    return logL
