import os, sys, time
import yaml
import numpy as np
from gofish import imagecube
from scipy.integrate import trapz, cumtrapz
from scipy import interpolate
from astropy.io import fits
import matplotlib.pyplot as plt

# model
mname = 'phys1_i40'


# load parameters of model
conf = open(mname + ".yaml")
config = yaml.load(conf, Loader=yaml.FullLoader)
dpars = config["disk_params"]
opars = config["outputs"]
conf.close()

# fetch what you need from .yaml
incl = opars["geometry"]["incl"]
PA = opars["geometry"]["PA"]
dpc = opars["geometry"]["dpc"]


# generate a spectrum
dat = fits.open(mname+'/'+mname+'_co.fits')
chanmaps = np.squeeze(dat[0].data)
spec = np.sum(chanmaps, axis=(1,2))
print('integrated CO = %f Jy km/s' % \
      (np.sum(spec * opars["velocity"]["velres"])))
plt.plot(np.arange(len(spec)), spec)
plt.show()



# generate and load 0th moment map (made with bettermoments)
os.chdir(mname)
os.system('bettermoments '+mname+'_co.fits -method zeroth -clip 0')
cube = imagecube(mname+'_co_M0.fits')
os.chdir('../')

# extract radial profile 
x, y, dy = cube.radial_profile(inc=incl, PA=PA, x0=0.0, y0=0.0, z0=0.3, phi=1.,
                               PA_min=90, PA_max=270, 
                               abs_PA=True, exclude_PA=False)


# integrated flux profile
def fraction_curve(radius, intensity):
    
    intensity[np.isnan(intensity)] = 0
    total =trapz(2*np.pi*radius*intensity, radius)   
    cum = cumtrapz(2*np.pi*radius*intensity, radius)

    return cum/total


# size interpolator
def Reff_fraction_smooth(radius, intensity, fraction=0.95):

    curve = fraction_curve(radius, intensity)
    curve_smooth = interpolate.interp1d(curve, radius[1:])

    return curve_smooth(fraction)


# return the size
print(' ')
print("CO effective radius at 90th percentile = %f au" % \
      (dpc * Reff_fraction_smooth(x, y, fraction=0.9)))
