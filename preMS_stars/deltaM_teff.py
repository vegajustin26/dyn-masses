import numpy as np
import os
import sys
from post_summary import post_summary
from astropy.io import ascii
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
plt.rc('font', size=9)

# load data
inp_dat = ascii.read('Tau_done.txt')
sname, Mdyn, eMdyn = inp_dat['name'], inp_dat['Mdyn'], inp_dat['eMdyn']
Teff, eTeff = inp_dat['Teff'], inp_dat['eTeff']

# set up plot
fig = plt.figure(figsize=(3.5, 5.0))
gs = gridspec.GridSpec(2, 1)
dMlims = [0.2, 5]
Tlims = [3100, 4700]
ax0 = fig.add_subplot(gs[0, 0])
ax0.set_xlim(Tlims)
ax0.set_ylim(dMlims)
ax0.set_yscale('log')
ax1 = fig.add_subplot(gs[1, 0])
ax1.set_xlim(Tlims)
ax1.set_ylim(dMlims)
ax1.set_yscale('log')
ax1.set_xlabel('$T_{\\rm eff} \,\, / \,\, {\\rm K}}$')
ax0.set_ylabel('$M_{\\rm dyn} \,\, / \,\, M_{\\rm pms}}$')
ax1.set_ylabel('$t_{\\rm dyn} \,\, / \,\, t_{\\rm pms}}$')

# top plot: mass ratios
# loop through sources to compute posteriors on the mass ratio (= Mdyn / Mpms)
mu_dM = np.empty_like(Mdyn)
hi_dM, lo_dM = np.empty_like(Mdyn), np.empty_like(Mdyn)
for i in range(len(sname)):

    # load Mpms posteriors
    dat = np.load('posteriors/'+sname[i]+'_f000.age-mass.posterior.npz')
    Mpms = 10.**dat['logM']

    # randomly sample a Gaussian pdf for Mdyn posteriors
    Malma = np.random.normal(loc=Mdyn[i], scale=eMdyn[i], size=len(Mpms))

    # mass ratio posteriors
    mass_ratio = Malma / Mpms
    log_dM = np.log10(mass_ratio)

    # mass ratio posterior summaries
    mu_dM[i], hi_dM[i], lo_dM[i] = post_summary(mass_ratio, prec=0.01)

ax0.errorbar(Teff, mu_dM, xerr=eTeff, yerr=[lo_dM, hi_dM], fmt='.')
ax0.plot(Tlims, [1, 1], ':')
ax0.set_yticks([1])
ax0.set_yticklabels([' 1 '])
ax0.text(2998., 1.96, '2', color='k', ha='center', va='center')
ax0.text(2998., 2.94, '3', color='k', ha='center', va='center')
ax0.text(2998., 3.92, '4', color='k', ha='center', va='center')
ax0.text(2998., 0.69, '0.7', color='k', ha='center', va='center')
ax0.text(2998., 0.49, '0.5', color='k', ha='center', va='center')
ax0.text(2998., 0.30, '0.3', color='k', ha='center', va='center')


# bottom plot: age ratios
inp = ascii.read('inference_summaries.txt')
t00, tbf = inp['col2'], inp['col5']
ax1.errorbar(Teff, 10**(tbf-t00), xerr=eTeff, fmt='.')
ax1.plot(Tlims, [1, 1], ':')
ax1.set_yticks([1])
ax1.set_yticklabels([' 1 '])
ax1.text(2998., 1.96, '2', color='k', ha='center', va='center')
ax1.text(2998., 2.94, '3', color='k', ha='center', va='center')
ax1.text(2998., 3.92, '4', color='k', ha='center', va='center')
ax1.text(2998., 0.69, '0.7', color='k', ha='center', va='center')
ax1.text(2998., 0.49, '0.5', color='k', ha='center', va='center')
ax1.text(2998., 0.30, '0.3', color='k', ha='center', va='center')

#ax0.grid(which='both')

xspt = [3.782, 3.740, 3.702, 3.679, 3.638, 3.609, 3.585, 3.551, 3.515, 3.476]
lspt = ['G0', 'G5', 'K0', 'K3', 'K5', 'K7', 'M0', 'M2', 'M4', 'M6']
for il in range(len(xspt)):
    ax0.annotate(lspt[il], xy=(10**xspt[il], 0.22), ha='center', size=9,
                 color='gray', zorder=0)
    ax1.annotate(lspt[il], xy=(10**xspt[il], 0.22), ha='center', size=9,
                 color='gray', zorder=0)

fig.subplots_adjust(wspace=0.0, hspace=0.15)
fig.subplots_adjust(left=0.15, right=0.85, bottom=0.09, top=0.99)
fig.savefig('deltaM_teff.png')
fig.clf()
