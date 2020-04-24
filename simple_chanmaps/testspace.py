import numpy as np
import sys
import os
from simple_disk import simple_disk
from astropy.io import fits

# constants to match RADMC3D
PI = 3.14159265358979323846264338328
PC = 3.08572e18
AU = 1.49698e10
CC = 2.9979245800000e10
KK = 1.3807e-16
HH = 6.6262000e-27


# generate a simple disk model class
disk = simple_disk(inc=20., PA=130., mstar=2.0, FOV=14.66, dist=150., Npix=734,
                   Tb0=150, Tbq=-0.5, r_max=600., Tbmax=500.)

# specify the velocity channels of interest
# (choose same velocities as my RADMC3D setup)
widthkms_0 = 6.2
velres = 0.092
extra_width = (2 * widthkms_0 / velres) % 1
nchan = np.int(2 * widthkms_0 / velres - extra_width)
widthkms = velres * nchan / 2.
velax = 1000 * np.linspace(-widthkms, widthkms, nchan)

# generate channel maps
cube = disk.get_cube(velax)

# pixel area (in sr)
pixel_area = (disk.cell_sky * PI / (180 * 3600))**2

# frequencies (Hz)
restfreq = 230.538e9
nu = restfreq * (1. - velax / (CC / 100.))

# convert cube brightness temperatures into Jy / pixel
for i in range(len(nu)):
    cube[i,:,:] *= 1e23 * pixel_area * 2 * nu[i]**2 * KK / CC**2 

# specify coordinates / filename you want
RA  = 65.0
DEC = 23.0
outfile = 'testrich3.fits'

# convert to proper FITS formatting
cube_for_fits = cube[:,::-1,:]
       

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


