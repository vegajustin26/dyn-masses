import os
import sys
import time
import numpy as np
from mk_FITScube import mk_FITScube
from vis_sample import vis_sample
from vis_sample.file_handling import import_data_uvfits
from lnprob import lnprob
import emcee

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
model = mk_FITScube(inc=theta[0], PA=theta[1], mstar=theta[2], FOV=FOV, 
                    dist=dist, Npix=Npix, Tb0=theta[3], Tbq=theta[4], 
                    r_max=theta[5], vsys=theta[6], Tbmax=Tbmax, 
                    restfreq=restfreq, datafile=data_file)

tvis, gcf, corr = vis_sample(imagefile=model, uvfile=data_file, 
                             return_gcf=True, return_corr_cache=True, 
                             mod_interp=False)

# package data and supplementary information
args = dvis, gcf, corr, fixed

# test serialized run
nsteps = 10
sampler = emcee.EnsembleSampler(nwalk, ndim, lnprob, args=[args])
start = time.time()
sampler.run_mcmc(p0, nsteps)
end = time.time()
print(end-start)
#np.savez('serial_test.chain', chain=sampler.chain)
