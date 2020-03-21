import numpy as np
import os
import sys
from post_summary import post_summary
from astropy.io import ascii
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
plt.rc('font', size=9)

bs = 100
fspot = ['f000', 'f017', 'f034', 'f051', 'f068', 'f085']

inp_dat = ascii.read('Tau_done.txt')
sname, Mdyn, eMdyn = inp_dat['name'], inp_dat['Mdyn'], inp_dat['eMdyn']

fig = plt.figure(figsize=(4, 3))
gs = gridspec.GridSpec(1, 1)
ax0 = fig.add_subplot(gs[0, 0])
tlims = [5.5, 7.7]
ax0.set_xlim(tlims)
ax0.set_ylabel('probability')
ax0.set_xlabel('$\log{\,\, t_\\ast \,\, / \,\, {\\rm yr}}$')

std_logAGE = []
bf_logAGE = []
bf_wgts = []
for i in range(len(sname)):
    if ((sname[i] != 'HN_Tau_A') & (sname[i] != 'CY_Tau')):
        # cycle through each spot model for a given target
        allf_logM = []
        allf_logAGE = []
        for j in range(len(fspot)):
            dat = np.load('posteriors/' + sname[i] + '_' + fspot[j] + \
                          '.age-mass.posterior.npz')
            logM, logAGE = dat['logM'], dat['logAGE']
            allf_logM = np.append(allf_logM, logM)
            allf_logAGE = np.append(allf_logAGE, logAGE)
 
            # record standard age posteriors (spot-free model)
            if j == 0: std_logAGE = np.append(std_logAGE, logAGE)

        # compute weights based on dynamical mass
        weights = np.exp(-0.5*((10**allf_logM - Mdyn[i]) / eMdyn[i])**2)
        crit = (allf_logAGE <= 8.5)
        weights[crit] *= np.exp(-0.5*((allf_logAGE[crit] - 6.8) / 0.5)**2)

        # record all age posteriors and weights
        bf_logAGE = np.append(bf_logAGE, allf_logAGE)
        bf_wgts = np.append(bf_wgts, weights)


# compute / plot the standard (spot-free) age distribution
n, bins, patches = ax0.hist(std_logAGE, bs, density=True, range=tlims, 
                            histtype='stepfilled', color='deepskyblue',
                            alpha=0.5)

n, bins, patches = ax0.hist(bf_logAGE, bs, weights=bf_wgts, density=True, 
                            range=tlims, histtype='stepfilled', 
                            color='crimson', alpha=0.5)

ax0.text(7.6, 1.18, 'standard', color='deepskyblue', ha='right')
ax0.text(7.6, 1.10, 'magnetic', color='crimson', ha='right')


fig.subplots_adjust(wspace=0.0, hspace=0.15)
fig.subplots_adjust(left=0.15, right=0.85, bottom=0.15, top=0.99)
fig.savefig('age_histograms.png')
fig.clf()
