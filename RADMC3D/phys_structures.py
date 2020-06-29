import os, sys
import numpy as np
import yaml
from sim_disk import sim_disk
import scipy.constants as sc
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colorbar import Colorbar


# constants
mu = 2.37
m_p = sc.m_p * 1e3
AU = 1.496e13

# contouring
Tlevs = np.logspace(np.log10(3), np.log10(300), 50)
nlevs = np.linspace(1, 14, 39)
Tmap = 'coolwarm_r'
nmap = 'pink_r'

# set up the plot grid
plt.rc('font', size=9)
fig = plt.figure(figsize=(7., 7.5))
gs = gridspec.GridSpec(5, 5, width_ratios=(1, 0.05, 0.5, 1, 0.05), 
                             height_ratios=(1, 1, 1, 1, 1))
rlims, zlims = [0.125, 800], [0.5, 200]

mnames = ['phys1_i40', 'phys2_i40', 'phys3_i40', 'phys4_i40', 'phys5_i40']

for i in range(len(mnames)):

    # generate structure
    di = sim_disk(mnames[i], writestruct=False, cyl=True)

    # retrieve abundance layer heights
    

    # spatial coordinates
    r = di.rvals / AU
    z = di.zvals / AU

    # temperatures
    T = di.temp

    # densities
    ngas = di.rhogas / mu / m_p

    # temperature plot
    ax = fig.add_subplot(gs[i, 0])
    contourf_kwargs = {}
    cmap = contourf_kwargs.pop("cmap", Tmap)
    imT = ax.contourf(r, z, T, levels=Tlevs, cmap=cmap, **contourf_kwargs)
    ax.set_xlim(rlims)
    ax.set_ylim(zlims)
    ax.set_xscale('log')
    ax.set_yscale('log')
    if i == 4:
        ax.set_xticks([1, 10, 100])
        ax.set_xticklabels(['1', '10', '100'])
        ax.set_xlabel(r'$r \,\,\, [{\rm au}]$')
        ax.set_yticks([1, 10, 100])
        ax.set_yticklabels(['1', '10', '100'])
        ax.set_ylabel(r'$z \,\,\, [{\rm au}]$')
    else:
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    # density plot
    ax = fig.add_subplot(gs[i, 3])
    contourf_kwargs = {}
    cmap = contourf_kwargs.pop("cmap", nmap)
    imn = ax.contourf(r, z, np.log10(ngas), levels=nlevs, cmap=cmap, 
                     **contourf_kwargs)
    ax.plot(r, 0.15*r, '--k')
    ax.set_xlim(rlims)
    ax.set_ylim(zlims)
    ax.set_xscale('log')
    ax.set_yscale('log')
    if i == 4:
        ax.set_xticks([1, 10, 100])
        ax.set_xticklabels(['1', '10', '100'])
        ax.set_xlabel(r'$r \,\,\, [{\rm au}]$')
        ax.set_yticks([1, 10, 100])
        ax.set_yticklabels(['1', '10', '100'])
        ax.set_ylabel(r'$z \,\,\, [{\rm au}]$')
    else:
        ax.set_xticklabels([])
        ax.set_yticklabels([])


# temperature colorbar
cbaxT = fig.add_subplot(gs[4:5, 1])
cb = Colorbar(ax=cbaxT, mappable=imT, orientation='vertical')
cbaxT.set_ylabel(r'$T \,\,\, [{\rm K}]$', labelpad=5)

# density colorbar
cbaxn = fig.add_subplot(gs[4:5, 4])
cb = Colorbar(ax=cbaxn, mappable=imn, orientation='vertical')
cbaxn.set_ylabel(r'$\log{n} \,\,\, [{\rm cm}^{-3}]$', labelpad=5)
    

fig.subplots_adjust(hspace=0.0, wspace=0.05)
fig.subplots_adjust(left=0.07, right=0.93, bottom=0.05, top=0.99)
fig.savefig('phys_structures.pdf')
