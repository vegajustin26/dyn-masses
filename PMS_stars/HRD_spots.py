import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.interpolate import interp1d
from astropy.io import ascii


fspt = 'f000'


# plot layout
xinch = 5.5
yinch = 4.3
lbound, rbound, bbound, tbound = 0.10, 0.90, 0.09, 0.98
fig = plt.figure(figsize=(xinch, yinch))
gs = gridspec.GridSpec(1, 1)

# plotting stuff for each parameter
Llims = [-1.8, 1.0]     # log Lstar / Lsun
Tlims = [4500, 3100]


ax = fig.add_subplot(gs[0,0])

# SPOTS isochrones 
iso_dat = ascii.read('SPOTS/Isochrones/'+fspt+'.isoc')
iso_AGE, iso_L, iso_T = iso_dat['col1'], iso_dat['col5'], iso_dat['col8']

# SPOTS isochrones @ 1, 2, 4, 8, 15 Myr
age = [6.00, 6.30, 6.60, 6.90, 7.15]
for i in range(len(age)):
    ax.plot(10.**iso_T[iso_AGE == age[i]], iso_L[iso_AGE == age[i]], 'c')

iso_dat = ascii.read('SPOTS/Isochrones/f051.isoc')
iso_AGE, iso_L, iso_T = iso_dat['col1'], iso_dat['col5'], iso_dat['col8']


# MIST mass tracks @ 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2 Msun
ms = ['m020', 'm040', 'm060', 'm080', 'm100']
for i in range(len(ms)):
    mt_dat = ascii.read('SPOTS/Tracks/'+ms[i]+'_'+fspt+'.track')
    mt_AGE, mt_L, mt_T = mt_dat['Age(Gyr)'], mt_dat['log(L/Lsun)'], mt_dat['log(Teff)']
    ax.plot(10.**mt_T[mt_AGE <= 0.05], mt_L[mt_AGE <= 0.05], 'r')


# datapoints
os.system('cp -r /pool/asha0/SCIENCE/ARAA/DISKS.csv temp.csv')
db = ascii.read('temp.csv', format='csv', fast_reader=True)
base = ((db['SFR'] != 'Usco') & (db['SFR'] != 'IC348') & (db['FL_MULT'] != 'J') & (db['FL_MULT'] != 'B') & (db['FL_MULT'] != 'HJ') & (db['FL_MULT'] != 'HJB') & (db['FL_MULT'] != 'HC') & (db['FL_MULT'] != 'HCB') & (db['FL_MULT'] != 'CB') & (db['FL_Teff'] == 0) & (db['SED'] != 'I'))

base = ((db['SFR'] == 'Tau') & (db['FL_MULT'] != 'J') & (db['FL_MULT'] != 'B') & (db['FL_MULT'] != 'HJ') & (db['FL_MULT'] != 'HJB') & (db['FL_MULT'] != 'HC') & (db['FL_MULT'] != 'HCB') & (db['FL_MULT'] != 'CB') & (db['FL_Teff'] == 0) & (db['SED'] != 'I'))


#ax.errorbar(db['logTeff'][base], db['logLs'][base], xerr=db['elogTeff'][base],
#            yerr=[db['elogLs_L'][base], db['elogLs_H'][base]], fmt='.k',
#            ecolor='gray', alpha=0.3)
ax.plot(10.**db['logTeff'][base], db['logLs'][base], '.k', zorder=0)

K5 = (base & (db['logTeff'] >= 3.62) & (db['logTeff'] < 3.64))
K7 = (base & (db['logTeff'] >= 3.59) & (db['logTeff'] < 3.62))
M0 = (base & (db['logTeff'] >= 3.57) & (db['logTeff'] < 3.59))
M1 = (base & (db['logTeff'] >= 3.56) & (db['logTeff'] < 3.57))
M2 = (base & (db['logTeff'] >= 3.54) & (db['logTeff'] < 3.56))
M3 = (base & (db['logTeff'] >= 3.52) & (db['logTeff'] < 3.54))
M4 = (base & (db['logTeff'] >= 3.50) & (db['logTeff'] < 3.52))
early = (base & (db['logTeff'] >= 3.64))
for i in range(len(db['NAME'][early])): print(db['NAME'][early][i])



# spectral type labels
xspt = [3.782, 3.740, 3.702, 3.679, 3.638, 3.609, 3.585, 3.551, 3.515, 3.476]
lspt = ['G0', 'G5', 'K0', 'K3', 'K5', 'K7', 'M0', 'M2', 'M4', 'M6']
for il in range(len(xspt)):
    ax.annotate(lspt[il], xy=(10**xspt[il], -1.75), ha='center', size=10, 
                color='gray', zorder=0)


# plot appearances
ax.set_ylim(Llims)
ax.set_xlim(Tlims)
ax.set_xlabel(r'$T_{\rm eff}$'+' / '+r'${\rm K}$', fontsize=10)
ax.set_ylabel(r'${\rm log}$'+' '+r'$L_\ast$'+' / '+r'$L_\odot$', fontsize=10)
ax.yaxis.labelpad = 2

ax.tick_params(axis='both', which='major', labelsize=9)
fig.subplots_adjust(left=lbound, right=rbound, bottom=bbound, top=tbound)
fig.savefig('HRD_spots_'+fspt+'.pdf')
fig.clf()
