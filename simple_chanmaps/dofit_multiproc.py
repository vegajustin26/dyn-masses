import os
import sys
import time
import numpy as np
from mk_FITScube import mk_FITScube
from vis_sample import vis_sample
from vis_sample.file_handling import import_data_uvfits
from lnprob import lnprob
from lnprob_globdata import lnprob_globdata
import emcee
from multiprocessing import Pool
os.environ["OMP_NUM_THREADS"] = "1"

# parse and package the DATA
data_set = 'rich_io.sim.config5.30min.noisy.specbin'
data_file = 'fake_data/'+data_set+'.uvfits'
dvis = import_data_uvfits(data_file)

# fixed parameters
FOV = 8.0
dist = 150.
Npix = 256
Tbmax = 500.
restfreq = 230.538e9
fixed = FOV, dist, Npix, Tbmax, restfreq

# initialize walkers
p_lo = np.array([20., 30., 1.50, 100., -1.0, 100., -0.1, -0.1, -0.1])
p_hi = np.array([40., 50., 2.00, 200., -0.2, 400.,  0.1,  0.1,  0.1])
ndim, nwalk = len(p_lo), 4 * len(p_lo)
p0 = [np.random.uniform(p_lo, p_hi, ndim) for i in range(nwalk)]

# compute 1 model to set up GCF, corr caches
theta = p0[0]
foo = mk_FITScube(inc=theta[0], PA=theta[1], mstar=theta[2], FOV=FOV, 
                  dist=dist, Npix=Npix, Tb0=theta[3], Tbq=theta[4], 
                  r_max=theta[5], vsys=theta[6], Tbmax=Tbmax, 
                  restfreq=restfreq, datafile=data_file)

tvis, gcf, corr = vis_sample(imagefile=foo, uvfile=data_file, return_gcf=True, 
                             return_corr_cache=True, mod_interp=False)


# package data and supplementary information
global dpassit
dpassit = dvis, gcf, corr, fixed

# posterior sample function
def lnprob_globdata(theta):

    # parse input arguments
    dvis, gcf, corr, fixed = dpassit
    FOV, dist, Npix, Tbmax, restfreq = fixed

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

    # Tb0: p(Tb0) = uniform(0, Tbmax)		
    if ((theta[3] > 0.) and (theta[3] < Tbmax)):
        ptheta[3] = 1./Tbmax
    else: return -np.inf

    # Tbq: p(Tbq) = uniform(-2, 0)
    if ((theta[4] > -2) and (theta[4] < 0)):
        ptheta[4] = 1./2.
    else: return -np.inf

    # r_max: p(r_max) = uniform(0, dist * FOV / 2)		
    if ((theta[5] > 0.) and (theta[5] < (dist * FOV / 2))):
        ptheta[5] = 1./(dist * FOV / 2)
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

    # convert to velocities
    vel = CC * (1. - dvis.freqs / restfreq) / 100.

    # generate a model FITS cube
    # presumes theta = [inc, PA, mstar, Tb0, Tbq, r_max, V_sys, dx, dy]
    model = mk_FITScube(inc=theta[0], PA=theta[1], mstar=theta[2], 
                        FOV=FOV, dist=dist, Npix=Npix, 
                        Tb0=theta[3], Tbq=theta[4], r_max=theta[5], 
                        vsys=theta[6], Tbmax=Tbmax, restfreq=restfreq, vel=vel)

    # now sample the FT of the model onto the observed (u,v) points
    modl_vis = vis_sample(imagefile=model, mu_RA=theta[7], 
                          mu_DEC=theta[8], gcf_holder=gcf, 
                          corr_cache=corr, mod_interp=False)

    # compute the log-likelihood
    logL = -0.5 * np.sum(dvis.wgts * np.absolute(dvis.VV.T - modl_vis)**2)

    # return the posterior
    return logL + np.sum(np.log(ptheta))


nsteps = 10
# perform the inference
with Pool() as pool:
    sampler = emcee.EnsembleSampler(nwalk, ndim, lnprob_globdata, pool=pool)
    start = time.time()
    sampler.run_mcmc(p0, nsteps)
    end = time.time()
    print("multiproc took {0:.1f} seconds".format(end-start))
