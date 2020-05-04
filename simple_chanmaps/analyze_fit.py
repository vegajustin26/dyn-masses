import numpy as np
import matplotlib.pyplot as plt
import emcee
import corner
import os
import sys


# emcee backend file
fname = 'rich_io.sim.config5.30min.noisy.specbin'

# load the backend
reader = emcee.backends.HDFBackend(fname+'.h5')

# parse the samples
samples = reader.get_chain(discard=300, flat=False)
nsteps, nwalk, ndim = samples.shape[0], samples.shape[1],  samples.shape[2]

# parse the probabilities
lnprob = reader.get_log_prob(discard=0, flat=True)

# parameter labels
lbls = ['i', 'PA', 'M', 'Tb0', 'Tbq', 'r_max', 'vsys', 'dx', 'dy']

# truths
theta = [30., 40., 1.75, 150., -0.5, 300., 0.0, 0.0, 0.0]

# plot the traces
fig = plt.figure()
for idim in np.arange(ndim):
    plt.subplot(3, 3, idim+1)
    for iw in np.arange(nwalk):
        plt.plot(np.arange(nsteps), samples[:, iw, idim], alpha=0.3)
        plt.ylabel(lbls[idim])
fig.savefig(fname + '.traces.png')
fig.clf()

# corner plot to view covariances
levs = 1. - np.exp(-0.5*(np.arange(3)+1)**2)
flat_chain = samples.reshape(-1, ndim)
fig, axes = plt.subplots(ndim, ndim, figsize=(6.5, 6.5))
corner.corner(flat_chain, plot_datapoints=False, levels=levs, 
              labels=lbls, fig=fig, truths=theta)
fig.savefig(fname + '.corner.png')
fig.clf()
