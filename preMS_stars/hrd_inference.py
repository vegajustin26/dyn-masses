import numpy as np
import os
import sys
import time
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from astropy.io import ascii
from scipy.interpolate import interp1d
from post_summary import post_summary


# inputs
sname = 'IPTau'
in_logT, in_elogT = 3.583, 0.014
in_logL, in_elogL = -0.328, 0.10
fspot = 'f085'


# bookkeeping
name = sname + '_' + fspot
ndim = 6

# plot layout
xinch = 7.5
yinch = 6.5
dhspace, dwspace = 0.25, 0.25
lbound, rbound, bbound, tbound = 0.08, 0.92, 0.08, 0.99
fig = plt.figure(figsize=(xinch, yinch))
gs = gridspec.GridSpec(2, 2)
ax = fig.add_subplot(gs[0,0])

# plot ranges for each parameter
Llims = [-1.5, 0.3]     # log Lstar / Lsun
Tlims = [3.62, 3.50]      # log Teff / K
Mlims = [-0.7, -0.2]     # log Mstar / Msun
Alims = [5., 7.5]	

# HRD
ntest = 10000
data_logT = np.random.normal(in_logT, in_elogT, ntest)
data_logL = np.random.normal(in_logL, in_elogL, ntest)

# overplot guesses at mean age, mass models
iso_dat = ascii.read('SPOTS/Isochrones/'+fspot+'.isoc')
iso_AGE, iso_M = iso_dat['col1'], iso_dat['col2']
iso_L, iso_T = iso_dat['col5'], iso_dat['col8']
ax.plot(iso_T[iso_AGE == 6.45], iso_L[iso_AGE == 6.45], 'b')

mt_dat = ascii.read('SPOTS/Tracks/m075_'+fspot+'.track')
mt_AGE = np.log10(mt_dat['Age(Gyr)'] * 1e9)
mt_L, mt_T = mt_dat['log(L/Lsun)'], mt_dat['log(Teff)']
ax.plot(mt_T[mt_AGE <= 7.5], mt_L[mt_AGE <= 7.5], 'r')

ax.plot(data_logT, data_logL, ',k', zorder=0)
ax.errorbar(in_logT, in_logL, xerr=in_elogT, yerr=in_elogL, fmt='ok')

ax.set_ylim(Llims)
ax.set_xlim(Tlims)



# load isochrone data
iso_dat = ascii.read('SPOTS/Isochrones/'+fspot+'.isoc')
init_AGE = iso_dat['col1']
ok = (init_AGE <= 8.0)
iso_AGE, iso_M = (iso_dat['col1'])[ok], (iso_dat['col2'])[ok]
iso_L, iso_T = (iso_dat['col5'])[ok], (iso_dat['col8'])[ok]
ages = np.unique(iso_AGE)

# load list of mass track filenames
files = np.loadtxt('SPOTS/Tracks/filenames.txt', dtype=str)

# loop through a collection of datapoints
axbot = fig.add_subplot(gs[1,0])
axtop = fig.add_subplot(gs[0,1])
axcor = fig.add_subplot(gs[1,1])

agestar  = np.zeros(ntest)
massstar = np.zeros(ntest)

t0 = time.time()
for i in range(ntest):
    if (i % 1000) == 0: print(i)
    # mass as function of temperature at the given luminosity
    interp_M = []
    interp_T = []
    for im in range(len(files)):

        # load the mass track file
        mt_dat = ascii.read('SPOTS/Tracks/'+files[im]+'_'+fspot+'.track', 
                            fast_reader=True)
        mt_age = np.log10(mt_dat['Age(Gyr)'] * 1e9)
        mt_logM = np.log10((mt_dat['Mass'])[mt_age <= 8.0])
        mt_logL = (mt_dat['log(L/Lsun)'])[mt_age <= 8.0]
        mt_logT = (mt_dat['log(Teff)'])[mt_age <= 8.0]

        # check and see if the mass track surrounds the input log L value
        ok = (np.max(mt_logL) >= data_logL[i]) & \
             (np.min(mt_logL) <= data_logL[i])

        # if so, interpolate the track to the input log L value
        if ok:
            ltint = interp1d(mt_logL, mt_logT)
            temp = ltint(data_logL[i])
            interp_T = np.append(interp_T, temp)
            interp_M = np.append(interp_M, mt_logM[0])

    # plot the interpolated temperature, mass points
    axbot.plot(interp_T, interp_M)

    # infer the corresponding stellar mass for that datapoint
    mtint = interp1d(interp_T, interp_M, fill_value='extrapolate')
    massstar[i] = mtint(data_logT[i])


    # age as function of luminosity at the given temperature
    interp_age = []
    interp_L = []
    for ia in range(len(ages)):
        il_logL = (iso_L)[iso_AGE == ages[ia]]
        il_logT = (iso_T)[iso_AGE == ages[ia]]

        # check and see if the isochrone surrounds the input log T value
        ok = (np.max(il_logT) >= data_logT[i]) & \
             (np.min(il_logT) <= data_logT[i])

        if ok:
            ilint = interp1d(il_logT, il_logL)
            lum = ilint(data_logT[i])
            interp_L = np.append(interp_L, lum)
            interp_age = np.append(interp_age, ages[ia])

    # plotting stuff for each parameter
    axtop.plot(interp_age, interp_L)

    ageint = interp1d(interp_L, interp_age, fill_value='extrapolate')
    agestar[i] = ageint(data_logL[i])



axcor.plot(agestar, massstar, ',k')

axbot.set_ylim(Mlims)
axbot.set_xlim(Tlims)
axtop.set_ylim(Llims)
axtop.set_xlim(Alims)
axcor.set_xlim(Alims)
axcor.set_ylim(Mlims)

tf = time.time()
print(tf-t0)

os.system('rm -rf '+name+'.age-mass.posterior.npz')
np.savez(name+'.age-mass.posterior.npz', logAGE=agestar, logM=massstar)

# plot appearances
fig.subplots_adjust(left=lbound, right=rbound, bottom=bbound, top=tbound)
fig.subplots_adjust(hspace=dhspace, wspace=dwspace)
fig.savefig(name+'_hrd_inferences.png')
fig.clf()
