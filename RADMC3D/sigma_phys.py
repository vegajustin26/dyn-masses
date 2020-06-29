import os, sys
import yaml
import numpy as np
import scipy.constants as sc
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable

plt.rc('font', size=9)


# set up the figure
fig, ax = plt.subplots(figsize=(3.5, 2.5))
ax.set_xlim([1., 1000.])
ax.set_xscale('log')
ax.set_ylim([1.25e-1, 8e2])
ax.set_yscale('log')
ax.set_xlabel(r'$r \,\,\, [{\rm au}]$')
ax.set_xticks([1, 10, 100, 1000])
ax.set_xticklabels(['1', '10', '100', '1000'])
ax.set_ylabel(r'$\Sigma \,\,\, [{\rm g \,\, cm}^{-2}]$')
ax.set_yticks([1, 10, 100])
ax.set_yticklabels(['1', '10', '100'])

# model colors
cols = ['r', 'g', 'b', 'c', 'm']

for im in np.arange(1,6,1):
    
    # model ID
    mname = 'phys'+str(im)+'_i40'

    # fetch surface density profile
    r, sig, H = np.loadtxt(mname+'/gas_profiles.txt', skiprows=1).T

    # overplot the density profile
    ax.plot(r, sig, cols[im-1], lw=2)


fig.subplots_adjust(left=0.15, right=0.85, bottom=0.15, top=0.99)
fig.savefig('sigma_phys.pdf')
fig.clf()
