import numpy as np
from astropy.io import fits
from mk_FITScube import mk_FITScube
from vis_sample import vis_sample

def lnprob(theta, data):

    # constants
    CC = 2.9979245800000e10
    KK = 1.3807e-16
    restfreq = 230.538e9

    # unpack the data
    data_uu, data_vv, data_vis, data_wgt, data_freqs = data

    # generate a model
    # presumes theta = [inc, PA, mstar, Tb0, Tbq, r_max, V_sys]
    disk = simple_disk(inc=theta[0], PA=theta[1], mstar=theta[2], 
                       FOV=12.0, dist=150., Npix=512, 
                       Tb0=theta[3], Tbq=theta[4], r_max=theta[5], Tbmax=500.)

    # convert data frequencies into velocities (in m/s)
    vel = CC * (1. - data_freqs / restfreq) / 100.

    # adjust for systemic velocity
    vlsr = vel - (theta[6] * 1000.)

    # generate channel maps
    cube = disk.get_cube(vlsr)

    # convert from brightness temperatures to Jy / pixel
    pixel_area = (disk.cell_sky * np.pi / (180. * 3600.))**2
    for i in range(len(data_freqs)):
        cube[i,:,:] *= 1e23 * pixel_area * 2 * data_freqs[i]**2 * KK / CC**2

    # pack the cube into a FITS file (change later?)
    hdu = fits.PrimaryHDU(cube[:,::-1,:])
    header = hdu.header
    
    # basic header inputs
    header['EPOCH'] = 2000.
    header['EQUINOX'] = 2000.
    header['LATPOLE'] = -1.436915713634E+01
    header['LONPOLE'] = 180.

    # spatial coordinates
    header['CTYPE1'] = 'RA---SIN'
    header['CUNIT1'] = 'DEG'
    header['CDELT1'] = -disk.cell_sky / 3600.
    header['CRPIX1'] = 0.5 * disk.Npix + 0.5
    header['CRVAL1'] = 65.
    header['CTYPE2'] = 'DEC--SIN'
    header['CUNIT2'] = 'DEG'
    header['CDELT2'] = disk.cell_sky / 3600.
    header['CRPIX2'] = 0.5 * disk.Npix + 0.5
    header['CRVAL2'] = 25.

    # frequency coordinates
    header['CTYPE3'] = 'FREQ'
    header['CUNIT3'] = 'Hz'
    header['CRPIX3'] = 1.
    header['CDELT3'] = data_freqs[1]-data_freqs[0]
    header['CRVAL3'] = data_freqs[0]
    header['SPECSYS'] = 'LSRK'
    header['VELREF'] = 257

    # intensity units
    header['BSCALE'] = 1.
    header['BZERO'] = 0.
    header['BUNIT'] = 'JY/PIXEL'
    header['BTYPE'] = 'Intensity'

    # output FITS
    hdu.writeto('model.fits', overwrite=True)



    # now sample the model onto the observed (u,v) points
    model_vis = vis_sample(imagefile='model.fits', (data_uu, data_vv))
 



