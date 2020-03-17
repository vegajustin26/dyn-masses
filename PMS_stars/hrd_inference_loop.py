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
#inp_dat = ascii.read('Tau_done.txt')
inp_dat = ascii.read('Tau_targets.txt')
in_logT = np.log10(inp_dat['Teff'])
in_elogT = 0.4343*inp_dat['eTeff']/inp_dat['Teff']
in_logL = inp_dat['L']
in_elogL = 0.15 * np.ones_like(in_logL)
sname = inp_dat['name']
 
#fspot = ['f000', 'f017', 'f034', 'f051', 'f068', 'f085']
fspot = ['f000']

for i in range(len(fspot)):
    for j in np.arange(9, len(sname), 1):	#range(len(sname)):

        # bookkeeping
        print(sname[j])
        name = sname[j] + '_' + fspot[i]

        # HRD
        ntest = 7500
        data_logT = np.random.normal(in_logT[j], in_elogT[j], ntest)
        data_logL = np.random.normal(in_logL[j], in_elogL[j], ntest)

        # load isochrone data
        iso_dat = ascii.read('SPOTS/Isochrones/'+fspot[i]+'.isoc')
        init_AGE = iso_dat['col1']
        ok = (init_AGE <= 8.0)
        iso_AGE, iso_M = (iso_dat['col1'])[ok], (iso_dat['col2'])[ok]
        iso_L, iso_T = (iso_dat['col5'])[ok], (iso_dat['col8'])[ok]
        ages = np.unique(iso_AGE)

        # load list of mass track filenames
        files = np.loadtxt('SPOTS/Tracks/filenames.txt', dtype=str)

        agestar  = np.zeros(ntest)
        massstar = np.zeros(ntest)

        for ix in range(ntest):
            # mass as function of temperature at the given luminosity
            interp_M = []
            interp_T = []

            for im in range(len(files)):
                # load the mass track file
                mdir = 'SPOTS/Tracks/'
                mt_dat = ascii.read(mdir+files[im]+'_'+fspot[i]+'.track', 
                                    fast_reader=True)
                mt_age = np.log10(mt_dat['Age(Gyr)'] * 1e9)
                mt_logM = np.log10((mt_dat['Mass'])[mt_age <= 8.0])
                mt_logL = (mt_dat['log(L/Lsun)'])[mt_age <= 8.0]
                mt_logT = (mt_dat['log(Teff)'])[mt_age <= 8.0]

                # see if the mass track surrounds the input log L value
                ok = (np.max(mt_logL) >= data_logL[ix]) & \
                     (np.min(mt_logL) <= data_logL[ix])

                # if so, interpolate the track to the input log L value
                if ok:
                    ltint = interp1d(mt_logL, mt_logT)
                    temp = ltint(data_logL[ix])
                    interp_T = np.append(interp_T, temp)
                    interp_M = np.append(interp_M, mt_logM[0])

            # infer the corresponding stellar mass for that datapoint
            mtint = interp1d(interp_T, interp_M, fill_value='extrapolate')
            massstar[ix] = mtint(data_logT[ix])

            # age as function of luminosity at the given temperature
            interp_age = []
            interp_L = []
            for ia in range(len(ages)):
                il_logL = (iso_L)[iso_AGE == ages[ia]]
                il_logT = (iso_T)[iso_AGE == ages[ia]]

                # see if the isochrone surrounds the input log T value
                ok = (np.max(il_logT) >= data_logT[ix]) & \
                     (np.min(il_logT) <= data_logT[ix])

                if ok:
                    ilint = interp1d(il_logT, il_logL)
                    lum = ilint(data_logT[ix])
                    interp_L = np.append(interp_L, lum)
                    interp_age = np.append(interp_age, ages[ia])

            ageint = interp1d(interp_L, interp_age, fill_value='extrapolate')
            agestar[ix] = ageint(data_logL[ix])

        # outputs
        os.system('rm -rf posteriors/'+name+'.age-mass.posterior.npz')
        np.savez('posteriors/'+name+'.age-mass.posterior.npz', logAGE=agestar, 
                 logM=massstar)
