import numpy as np
from astropy.io import fits
from simple_disk import simple_disk

def mk_FITScube(inc=45., PA=90., mstar=1.0, FOV=5., dist=150., Npix=256,
                Tb0=150., Tbq=-1.0, r_max=250., vsys=0., Tbmax=300., vels=None,
                datafile=None, restfreq=230.538e9, RA=65., DEC=25., 
                outfile='model.fits'):


    # constants
    CC = 2.9979245800000e10
    KK = 1.3807e-16


    # generate an emission model
    disk = simple_disk(inc=inc, PA=PA, mstar=mstar, FOV=FOV, dist=dist, 
                       Npix=Npix, Tb0=Tb0, Tbq=Tbq, r_max=r_max, Tbmax=Tbmax)


    # decide on velocities
    if ((vels == None) & (datafile == None)):
        vel = np.linspace(-5000, 5100, 100)
    
    if datafile is not None:
        # load datafile header
        dat = fits.open(datafile)
        hdr = dat[0].header

        # frequencies
        freq0 = hdr['CRVAL4']
        indx0 = hdr['CRPIX4']
        nchan = hdr['NAXIS4']
        dfreq = hdr['CDELT4']
        freqs = freq0 + (np.arange(nchan) - indx0 + 1) * dfreq

        # velocities
        vel = CC * (1. - freqs / restfreq) / 100.
    else:
        freqs = restfreq * (1. - vel / (CC / 100.))     


    # adjust for systemic velocity
    vlsr = vel - (vsys * 1000.)


    # generate channel maps
    cube = disk.get_cube(vlsr)


    # convert from brightness temperatures to Jy / pixel
    pixel_area = (disk.cell_sky * np.pi / (180. * 3600.))**2
    for i in range(len(freqs)):
        cube[i,:,:] *= 1e23 * pixel_area * 2 * freqs[i]**2 * KK / CC**2


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
    header['CRVAL1'] = RA
    header['CTYPE2'] = 'DEC--SIN'
    header['CUNIT2'] = 'DEG'
    header['CDELT2'] = disk.cell_sky / 3600.
    header['CRPIX2'] = 0.5 * disk.Npix + 0.5
    header['CRVAL2'] = DEC

    # frequency coordinates
    header['CTYPE3'] = 'FREQ'
    header['CUNIT3'] = 'Hz'
    header['CRPIX3'] = 1.
    header['CDELT3'] = freqs[1]-freqs[0]
    header['CRVAL3'] = freqs[0]
    header['SPECSYS'] = 'LSRK'
    header['VELREF'] = 257

    # intensity units
    header['BSCALE'] = 1.
    header['BZERO'] = 0.
    header['BUNIT'] = 'JY/PIXEL'
    header['BTYPE'] = 'Intensity'

    # output FITS
    hdu.writeto(outfile, overwrite=True)
