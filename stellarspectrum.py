import numpy as np
import os
import sys
from scipy.interpolate import interp1d, interp2d, RectBivariateSpline
import matplotlib.pyplot as plt


def stellarspectrum(teff, lstar, dpc=1.0, grid=None, logg=None, mstar=None, 
                    writegrid=True):

    # constants
    Lsun = 3.826e33
    Msun = 1.989e33
    pc = 3.08572e18
    GG = 6.67259e-8
    sigSB = 5.67051e-5

    # path to models
    dir_nextgen = '/pool/asha1/HOLD/nextgen/'

    # convert to proper units
    lstar *= Lsun
    dcm = dpc * pc

    # if a model grid is not provided, compute one
    nwl = 1221
    if grid is None:
        tgrid = np.array(np.loadtxt(dir_nextgen+'TEMP_GRID.dat'))
        ggrid = np.loadtxt(dir_nextgen+'GRAV_GRID.dat')
        spec = np.empty((len(ggrid), len(tgrid), nwl))
        for j in range(len(ggrid)):
            for i in range(len(tgrid)):
                tname = str(np.int(0.01*tgrid[i]))
                if (tgrid[i] < 10000.): tname = '0' + tname
                gname = f'{ggrid[j]:.1f}'
                infile = dir_nextgen + 'Fnu/NG_' + tname + '_' + gname + '_00'
                wl, ucf, cf = np.loadtxt(infile + '.dat').T
                spec[j,i,:] = cf
        if writegrid: 
            np.savez('sgrid.npz', spec=spec, tgrid=tgrid, ggrid=ggrid, wl=wl)
    else:
        spec, wl, tgrid, ggrid = grid.spec, grid.wl, grid.tgrid, grid.ggrid

    # if logg is not provided, assume it or compute it
    if logg is None:
        if mstar is None:
            logg = 4.0		# asserted (lacking other input information)
        else:
            mstar *= Msun
            rstar = np.sqrt(lstar / (4. * np.pi * sigSB * teff**4))
            logg = np.log10(GG * mstar / rstar**2)


    # bivariate spline interpolation
    ispec = np.empty(nwl)
    for iw in range(nwl):
        fint = RectBivariateSpline(ggrid, tgrid, spec[:,:,iw])
        ispec[iw] = fint(logg, teff)

    # scale to appropriate distance, Jy units
    fnu = 1e23 * ispec * (rstar / dcm)**2

    # return spectrum and wavelengts as tuple
    return 1e-3*wl, fnu
