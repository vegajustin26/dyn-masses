# Original script written by Jane Huang, CfA
# cleaned up a bit (for Python 3.x)

# Reads in an image.out file from RADMC-3D and creates a new FITS file.
# Ideal for conversion from RADMC output to CASA simobserve, for ALMA proposals

import numpy as np
from astropy.io import fits, ascii
import os

AU = 1.49597870700e13 	# [cm]
pc = 3.08567758e18 	# [cm]
cc = 2.99792458e10 	# [cm s^-1]


def convert_to_fits(imagename, outfile, dpc, RA=0, DEC=0, 
                    tau=False, tausurf=False):

    # Read in the file from the RADMC-3D format
    imagefile = open(imagename)
    iformat = imagefile.readline()

    im_nx, im_ny = imagefile.readline().split()	#npixels along x and y axes
    im_nx, im_ny = np.int(im_nx), np.int(im_ny)

    nlam = np.int(imagefile.readline())

    pixsize_x, pixsize_y = imagefile.readline().split() #pixel sizes in cm 
    pixsize_x = np.float(pixsize_x)
    pixsize_y = np.float(pixsize_y)

    imvals = ascii.read(imagename, format='fast_csv', guess=False, 
                        data_start=4, 
                        fast_reader={'use_fast_converter':True})['1']
    lams = imvals[:nlam]

    # Convert lams (in microns) into frequencies in Hz
    freqs =  cc / (lams * 1e-4) 	

    CRVAL3 = freqs[0]
    if (len(lams) > 1):
        CDELT3 = freqs[1] - freqs[0]
    else:
        CDELT3 = 125.e6 	# just the bandwidth? Setting to 125 MHz now 

    pixsize = pixsize_x * pixsize_y / (dpc * pc)**2 	# pixel area in sr
    if tau:
        intensities = np.reshape(imvals[nlam:],[nlam, im_ny, im_nx]) 
    elif tausurf:
        intensities = np.reshape(imvals[nlam:],[nlam, im_ny, im_nx]) / AU 
    else:
        # RADMC gives intensities in erg cm^(-2) s^(-1) Hz^(-1) ster^(-1); 
        # want to convert to Jy/pixel (especially for CASA)
        intensities = np.reshape(imvals[nlam:],[nlam, im_ny, im_nx]) * \
                      pixsize * 1e23

    # Convert to float32 to store in FITS?
#    intensities = intensities.astype('float32')
    intensities = np.float32(intensities)





    # Now, export the image to a FITS file

    #check back later to make this more general (deal with the bug in cvel)
    hdu = fits.PrimaryHDU(intensities)
    header = hdu.header
    header['EPOCH'] = 2000.
    header['EQUINOX'] = 2000.
    # Latitude and Longitude of the pole of the coordinate system.
    header['LATPOLE'] = -1.436915713634E+01
    header['LONPOLE'] = 180.

    # Define the RA coordinate
    header['CTYPE1'] = 'RA---SIN'
    header['CUNIT1'] = 'DEG'

    cdelt1 = -pixsize_x / (pc * dpc) * 180 / np.pi
    header['CDELT1'] = cdelt1
    # Pixel coordinates of the reference point. For example, if the image is 
    # 256 pixels wide, then this would refer to the center of the 127th pixel.
    header['CRPIX1'] = np.int(0.5 * im_nx + 0.5)
    header['CRVAL1'] = RA

    # Define the DEC coordinate
    header['CTYPE2'] = 'DEC--SIN'
    header['CUNIT2'] = 'DEG'
    header['CDELT2'] = -1 * cdelt1 	#assumes square image
    #header['CRPIX2'] = int(0.5*im_ny + 1.0)
    header['CRPIX2'] = np.int(0.5 * im_ny + 0.5)
    header['CRVAL2'] = DEC

    # Define the frequency coordiante
    header['CTYPE3'] = 'FREQ'
    header['CUNIT3'] = 'Hz'
    header['CRPIX3'] = 1.
    header['CDELT3'] = CDELT3
    header['CRVAL3'] = CRVAL3

    header['SPECSYS'] = 'LSRK'
    header['VELREF'] = 257
    header['BSCALE'] = 1.
    header['BZERO'] = 0.
    if tau:
        header['BUNIT'] = 'dimensionless'
        header['BTYPE'] = 'Opacity'
    else:
        header['BUNIT'] = 'JY/PIXEL'
        header['BTYPE'] = 'Intensity'

    hdu.writeto(outfile)


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(description="Convert image.out into a FITS file. Optionally provide information that will be added to the header of the FITS file.")
    parser.add_argument("--image", default="image.out", help="The name of the file created by RADMC-3D.")
    parser.add_argument("--FITS", default="image.FITS", help="The name of the FITS file to which you want to export the image.")
    parser.add_argument("--dpc", default=140., type=float, help="At what distance [pc] is this source? Assumes Taurus distance by default.")
    parser.add_argument("--RA", default=0, type=float, help="Assign this as the RA to the object.")
    parser.add_argument("--DEC", default=0, type=float, help="Assign this as the DEC to the object.")
    parser.add_argument("--tau", default=False, type=bool, help="By default, assumes intensity image is being made")
    args = parser.parse_args()

    dpc = args.dpc # [pc]
    RA = args.RA
    DEC = args.DEC

    convert_to_fits(args.image, args.FITS, dpc, RA, DEC, args.tau)
