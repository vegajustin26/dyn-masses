import numpy as np
import os
import sys
from post_summary import post_summary
import corner
from astropy.io import ascii
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
plt.rc('font', size=9)

nbins = 30
fspot = ['f000', 'f017', 'f034', 'f051', 'f068', 'f085']

inp_dat = ascii.read('Tau_done.txt')
sname, Mdyn, eMdyn = inp_dat['name'], inp_dat['Mdyn'], inp_dat['eMdyn']

f = open('inference_summaries.txt', 'w')

for i in range(len(sname)):

    # bookkeeping
    name = sname[i]
    oname = name.replace('_', '')
    lname = name.replace('_', ' ')

    # set up plot
    fig = plt.figure(figsize=(3.5, 5.0))
    gs = gridspec.GridSpec(2, 1)
    Mlims = [-0.5, 0.1]
    Alims = [5.5, 7.7]
    prange = [Mlims, Alims]
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[1,0])
    ax1.set_xlabel('$\log{\,\, M_\\ast / M_\odot}$')
    ax1.set_ylabel('$\log{\,\, t_\\ast / yr}$')

    # defaults
    levs = 1.-np.exp(-0.5*(np.arange(2)+1)**2)
    quants = np.array([0.5, 0.16, 0.84])
    lcol = ['lightskyblue', 'deepskyblue', 'dodgerblue', 
            'royalblue', 'b', 'navy']

    # top plot: individual fspot posteriors
    all_logM = []
    all_logAGE = []
    for j in range(len(fspot)):

        # load posteriors
        dat = np.load('posteriors/'+name+'_'+fspot[j]+'.age-mass.posterior.npz')
        logM, logAGE = dat['logM'], dat['logAGE']
        all_logM = np.append(all_logM, logM)
        all_logAGE = np.append(all_logAGE, logAGE)

        # corner plot
        if j == 0:
            corner.hist2d(logM, logAGE, plot_datapoints=False, bins=nbins, 
                          ax=ax0, levels=[levs[0]], range=prange, 
                          no_fill_contours=True, plot_density=False, 
                          color=lcol[j])
            qf00 = corner.quantile(logAGE, quants)
        else:
            corner.hist2d(logM, logAGE, plot_datapoints=False, bins=nbins, 
                          levels=[levs[0]], range=prange, 
                          no_fill_contours=True, plot_density=False, 
                          color=lcol[j], ax=ax0)

    # annotations
    ax0.fill_between([np.log10(Mdyn[i]-eMdyn[i]), np.log10(Mdyn[i]+eMdyn[i])], 
                     [Alims[0], Alims[0]], [Alims[1], Alims[1]], 
                     color='mistyrose')
    ax0.text(0.04, 0.91, lname, ha='left', transform=ax0.transAxes)


    # bottom plot: combined posteriors + joint with dynamical mass

    # weights based on dynamical mass
    weights = np.exp(-0.5*((10**all_logM - Mdyn[i]) / eMdyn[i])**2)

    corner.hist2d(all_logM, all_logAGE, plot_datapoints=False, bins=nbins, 
                  ax=ax1, levels=levs, range=prange, no_fill_contours=True, 
                  plot_density=False, color='lightsteelblue')

    corner.hist2d(all_logM, all_logAGE, weights=weights, plot_datapoints=False, 
                  bins=nbins, levels=levs, range=prange, no_fill_contours=True,
                  plot_density=False, color='r', ax=ax1)

    qfall = corner.quantile(all_logAGE, quants, weights=weights)

    # annotations
    ax1.fill_between([np.log10(Mdyn[i]-eMdyn[i]), np.log10(Mdyn[i]+eMdyn[i])], 
                     [Alims[0], Alims[0]], [Alims[1], Alims[1]], 
                     color='mistyrose')


    # outputs
    f.write(f'{lname:15s}  {qf00[0]:.2f}  {qf00[2]-qf00[0]:.2f}  {qf00[0]-qf00[1]:.2f}  {qfall[0]:.2f}  {qfall[2]-qfall[0]:.2f}  {qfall[0]-qfall[1]:.2f}\n')

    fig.subplots_adjust(wspace=0.0, hspace=0.15)
    fig.subplots_adjust(left=0.15, right=0.85, bottom=0.09, top=0.99)
    fig.savefig('corner_plots/corner_'+oname+'_joint.png')
    fig.clf()

f.close()
