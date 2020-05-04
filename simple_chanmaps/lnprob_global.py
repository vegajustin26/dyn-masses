import os
import sys
import time
import numpy as np
from mk_FITScube import mk_FITScube
from vis_sample import vis_sample
from vis_sample.file_handling import import_data_uvfits

# posterior sample function
def lnprob_global(theta, data):

    # parse input arguments
    dvis, fixed = data
    FOV, dist, Npix, Tbmax, restfreq = fixed

    ### - PRIORS
    ptheta = np.empty_like(theta)

    # inc: p(i) = sin(i)
    if ((theta[0] > 0.) and (theta[0] < 90.)):
        #ptheta[0] = np.sin(np.radians(theta[0]))
        ptheta[0] = -0.5 * (theta[0] - 30.)**2 / 5.**2
    else: return -np.inf

    # PA: p(PA) = uniform(0, 360)
    if ((theta[1] > 0.) and (theta[1] < 360.)): 
        #ptheta[1] = 1./360.
        ptheta[1] = -0.5 * (theta[1] - 40.)**2 / 5.**2
    else: return -np.inf

    # Mstar: p(Mstar) = uniform(0, 5)
    if ((theta[2] > 0.) and (theta[2] < 5.)):
        ptheta[2] = 0.
    else: return -np.inf

    # Tb0: p(Tb0) = uniform(0, Tbmax)		
    if ((theta[3] > 5.) and (theta[3] < Tbmax)):
        ptheta[3] = 0.
    else: return -np.inf

    # Tbq: p(Tbq) = uniform(-2, 0)
    if ((theta[4] > -2) and (theta[4] < 0)):
        ptheta[4] = 0.
    else: return -np.inf

    # r_max: p(r_max) = uniform(0, dist * FOV / 2)		
    if ((theta[5] > 10.) and (theta[5] < (dist * FOV / 2))):
        ptheta[5] = 0.
    else: return -np.inf

    # V_sys: p(V_sys) = normal(0.0, 0.1)	# adjusted for each case
    if ((theta[6] > -0.2) & (theta[6] < 0.2)):
        ptheta[6] = 0.
    else: return -np.inf

    # dx: p(dx) = normal(0.0, 0.1)		# adjusted for each case
    #if ((theta[7] > -0.2) & (theta[7] < 0.2)):
    #    ptheta[7] = 0.
    #else: return -np.inf

    # dy: p(dy) = normal(0.0, 0.1)		# adjusted for each case
    #if ((theta[8] > -0.2) & (theta[8] < 0.2)):
    #    ptheta[8] = 0.
    #else: return -np.inf
    
    # constants
    CC = 2.9979245800000e10

    # convert to velocities
    vel = CC * (1. - dvis.freqs / restfreq) / 100.

    # generate a model FITS cube
    # presumes theta = [inc, PA, mstar, Tb0, Tbq, r_max, V_sys, dx, dy]
    model = mk_FITScube(inc=theta[0], PA=theta[1], mstar=theta[2], 
                        FOV=FOV, dist=dist, Npix=Npix, 
                        Tb0=theta[3], Tbq=theta[4], r_max=theta[5], 
                        vsys=theta[6], Tbmax=Tbmax, restfreq=restfreq, vel=vel)


    # now sample the FT of the model onto the observed (u,v) points
    #modl_vis = vis_sample(imagefile=model, mu_RA=theta[7], 
    #                      mu_DEC=theta[8], gcf_holder=gcf, 
    #                      corr_cache=corr, mod_interp=False)
    modl_vis = vis_sample(imagefile=model, 
                          uvfile='fake_data/rich_io_nf.uvfits',
                          mod_interp=False)

    # compute the log-likelihood
    logL = -0.5 * np.sum(dvis.wgts * np.absolute(dvis.VV.T - modl_vis)**2)
    print(logL)

    # return the posterior
    return logL + np.sum(ptheta)
