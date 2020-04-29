import numpy as np
from astropy.io import fits
from mk_FITScube import mk_FITScube
from vis_sample import vis_sample

def lnprob(theta, args):

    ### - PRIORS
    ptheta = np.empty_like(theta)

    # inc: p(i) = sin(i)
    if ((theta[0] > 0.) and (theta[0] < 90.)):
        ptheta[0] = np.sin(np.radians(theta[0]))
    else: return -np.inf

    # PA: p(PA) = uniform(0, 360)
    if ((theta[1] > 0.) and (theta[1] < 360.)): 
        ptheta[1] = 1./360.
    else: return -np.inf

    # Mstar: p(Mstar) = uniform(0, 5)
    if ((theta[2] > 0.) and (theta[2] < 5.)):
        ptheta[2] = 1./5.
    else: return -np.inf

    # Tb0: p(Tb0) = uniform(0, 500)		# should be tied to Tbmax
    if ((theta[3] > 0.) and (theta[3] < 500.)):
        ptheta[3] = 1./500.
    else: return -np.inf

    # Tbq: p(Tbq) = uniform(-2, 0)
    if ((theta[4] > 0.) and (theta[4] < 2.)):
        ptheta[4] = 1./2.
    else: return -np.inf

    # r_max: p(r_max) = uniform(0, 500)		# should be tied to FOV
    if ((theta[5] > 0.) and (theta[5] < 500.)):
        ptheta[5] = 1./500.
    else: return -np.inf

    # V_sys: p(V_sys) = normal(0.0, 0.1)	# adjusted for each case
    ptheta[6] = np.exp(-0.5*(theta[6] - 0.0)**2 / 0.1**2) / \
                (0.1 * np.sqrt(2 * np.pi))

    # dx: p(dx) = normal(0.0, 0.1)		# adjusted for each case
    ptheta[7] = np.exp(-0.5*(theta[7] - 0.0)**2 / 0.1**2) / \
                (0.1 * np.sqrt(2 * np.pi))

    # dy: p(dy) = normal(0.0, 0.1)		# adjusted for each case
    ptheta[8] = np.exp(-0.5*(theta[8] - 0.0)**2 / 0.1**2) / \
                (0.1 * np.sqrt(2 * np.pi))
    

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

    # return the posterior
    return logL + np.sum(np.log(ptheta))
