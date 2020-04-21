import numpy as np
import sys
import os
from simple_disk import simple_disk
from astropy.io import fits

# generate a simple disk model class
disk = simple_disk(inc=40., PA=90., mstar=2.0, FOV=10.20, dist=150., Npix=256,
                   z0=0.4, psi=0.5, Tb0=50, Tbqi=1.0, Tbqo=2.0, TbR=2.0)

# specify the velocity channels of interest
velax = -6200. + 92. * np.arange(136)

# generate channel maps
cube = disk.get_cube(velax)

# pixel area (in sr)
pixel_area = (disk.cell_sky * np.pi / (180 * 3600))**2
print(disk.cell_sky)

# frequencies (Hz)
restfreq, cc = 230.538e9, 2.9979e10
nu = restfreq * (1. - velax / (cc / 100.))

# convert cube brightness temperatures into Jy / pixel
kk = 1.381e-16
for i in range(len(nu)):
    cube[i,:,:] *= 1e23 * pixel_area * 2 * nu[i]**2 * kk / cc**2 

# specify coordinates / filename you want
RA  = 65.0
DEC = 23.0
outfile = 'testrich2.fits'

# convert to proper FITS formatting
cube_for_fits = cube
       

# pack all this away into a FITS file

# initiate cube
hdu = fits.PrimaryHDU(cube_for_fits)
header = hdu.header

# basic inputs
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
header['CDELT3'] = nu[1]-nu[0]
header['CRVAL3'] = nu[0]
header['SPECSYS'] = 'LSRK'
header['VELREF'] = 257

# intensity units
header['BSCALE'] = 1.
header['BZERO'] = 0.
header['BUNIT'] = 'JY/PIXEL'
header['BTYPE'] = 'Intensity'

# output FITS
hdu.writeto(outfile, overwrite=True)


