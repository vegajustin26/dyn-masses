import os
import sys
import numpy as np
import matplotlib.pyplot as plt


# model
modelname = 'demo_wrt'


# constants
pc = 3.0857e18
cc = 2.9979e10
Lsun = 3.826e33


# load spectrum
wl, f1pc = np.loadtxt(modelname+'/spectrum.out', skiprows=3).T
nu = cc * 1e4 / wl
nwl = len(wl)

# load star spectrum
fstar = np.loadtxt(modelname+'/stars.inp', skiprows=3+nwl)

# convert to SED
sed = np.log10(4 * np.pi * pc**2 * nu * f1pc / Lsun)
ssed = np.log10(4 * np.pi * pc**2 * nu * fstar / Lsun)

# plot
plt.plot(wl, sed)
plt.plot(wl, ssed)
plt.xscale('log')
plt.ylim([-6, 2])
plt.show()
