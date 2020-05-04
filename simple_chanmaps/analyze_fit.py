import numpy as np
import matplotlib.pyplot as plt
import emcee
import corner
import os
import sys


# emcee backend file
fname = 'rich_io_nf'

# load the backend
reader = emcee.backends.HDFBackend(fname+'.h5')

# set burn-in
burnin = 100

# set parameter labels, truths
lbls = ['i', 'PA', 'M', 'Tb0', 'Tbq', 'r_max', 'vsys', 'dx', 'dy']
theta = [30., 40., 1.75, 150., -0.5, 300., 0., 0., 0.]

# parse the samples
all_samples = reader.get_chain(discard=0, flat=False)
samples = reader.get_chain(discard=burnin, flat=False)
fsamples = reader.get_chain(discard=burnin, flat=True)
nsteps, nwalk, ndim = samples.shape[0], samples.shape[1],  samples.shape[2]

# parse the probabilities
lnprob = reader.get_log_prob(discard=burnin, flat=False)
flnprob = reader.get_log_prob(discard=burnin, flat=True)

# plot the integrated autocorrelation time convergence every Ntau steps
Ntau = 50
Nmax = all_samples.shape[0]
if (Nmax > Ntau):
    tau_ix = np.empty(np.int(Nmax / Ntau))
    ix = np.empty(np.int(Nmax / Ntau))
    for i in range(len(tau_ix)):
        nn = (i + 1) * Ntau
        ix[i] = nn
        tau = emcee.autocorr.integrated_time(all_samples[:nn,:,:], tol=0)
        tau_ix[i] = np.mean(tau)

    fig = plt.figure()
    plt.plot(ix, tau_ix, '-o')
    plt.xlabel('steps')
    plt.ylabel('autocorr time')
    plt.xlim([0, Nmax])
    plt.ylim([0, tau_ix.max() + 0.1 * (tau_ix.max() - tau_ix.min())])
    fig.savefig(fname+'.autocorr.png')
    fig.clf()

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
