import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.interpolate import interp1d
from astropy.io import ascii

# plot layout
xinch = 5.5
yinch = 4.3
lbound, rbound, bbound, tbound = 0.10, 0.90, 0.09, 0.98
fig = plt.figure(figsize=(xinch, yinch))
gs = gridspec.GridSpec(1, 1)

# plotting stuff for each parameter
Llims = [-1.0, 3.0]     # log Lstar / Lsun
Tlims = [4.15, 3.6]      # log Teff / K


ax = fig.add_subplot(gs[0,0])

# MIST isochrones 
iso_dir = '/pool/asha1/COMPLETED/Lupus_sizes/analysis/MIST_isochrones/MIST_v1.1_vvcrit0.4_full_isos/'
iso_dat = ascii.read(iso_dir+'MIST_v1.1_feh_p0.00_afe_p0.0_vvcrit0.4_full.iso')
iso_AGE, iso_L, iso_T = iso_dat['col2'], iso_dat['col9'], iso_dat['col14']

# MIST isochrones @ 0.5, 1, 2, 4, 8, 15 Myr
age = [5.7, 6.0, 6.300000000000001, 6.6000000000000005, 6.9, 7.15]
for i in range(len(age)):
    ax.plot(iso_T[iso_AGE == age[i]], iso_L[iso_AGE == age[i]], 'C0', alpha=0.5)

# MIST mass tracks @ 0.8, 1.2, 1.6, 2.0, 2.4, 2.8, 3.2 Msun
ms = ['00360', '00320', '00280', '00240', '00200', '00160', '00120', '00080']
mdir = '/pool/asha0/SCIENCE/ARAA/SP/ScottiePippen/data/MIST/MIST_v1.0_feh_p0.00_afe_p0.0_vvcrit0.4_EEPS/'
for i in range(len(ms)):
    mt_dat = ascii.read(mdir+ms[i]+'M.track.eep')
    mt_AGE, mt_L, mt_T = mt_dat['col1'], mt_dat['col7'], mt_dat['col12']
    ax.plot(mt_T[mt_AGE <= 5e7], mt_L[mt_AGE <= 5e7], 'r', alpha=0.5)


# datapoints
os.system('cp -r /pool/asha0/SCIENCE/ARAA/DISKS.csv temp.csv')
db = ascii.read('temp.csv', format='csv', fast_reader=True)
base = ((db['SFR'] != 'Usco') & (db['SFR'] != 'IC348') & (db['FL_MULT'] != 'J') & (db['FL_MULT'] != 'CB') & (db['FL_Teff'] == 0))

print(np.unique(db['SFR'][base]))

ax.errorbar(db['logTeff'][base], db['logLs'][base], xerr=db['elogTeff'][base],
            yerr=[db['elogLs_L'][base], db['elogLs_H'][base]], fmt='.k',
            ecolor='gray', alpha=0.3)
ax.plot(db['logTeff'][base], db['logLs'][base], '.k')



# spectral type labels
xspt = [3.862, 3.823, 3.782, 3.740, 3.702, 3.658, 3.617]
lspt = ['F0', 'F3', 'G0', 'G5', 'K0', 'K3', 'K5']
for il in range(len(xspt)):
    ax.annotate(lspt[il], xy=(xspt[il], 2.8), ha='center', size=10, 
                color='gray')


# plot appearances
ax.set_ylim(Llims)
ax.set_xlim(Tlims)
ax.set_xlabel(r'${\rm log}$'+' '+r'$T_{\rm eff}$'+' / '+r'${\rm K}$',
              fontsize=10)
ax.set_ylabel(r'${\rm log}$'+' '+r'$L_\ast$'+' / '+r'$L_\odot$', fontsize=10)
ax.yaxis.labelpad = 2

ax.tick_params(axis='both', which='major', labelsize=9)
fig.subplots_adjust(left=lbound, right=rbound, bottom=bbound, top=tbound)
fig.savefig('HRD_mist.pdf')
fig.clf()
