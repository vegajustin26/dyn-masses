import os
import numpy as np
execfile('/home/sandrews/mypy/keplerian_mask/keplerian_mask.py')


# simulation
fname = 'simp3_std_medv_medr_noisy'

# channelization
chanstart, chanwidth, nchan = '-7.5km/s', '0.2km/s', 77

# mask parameters
mstar, inc, PA, zr, rmax = 0.7, 40., 310., 0.23, 1.35

# imaging parameters
rms = '6.6mJy'

#########

# - fixed imaging parameters
cleanscales = [0, 5, 10, 15]
extens = ['.image', '.mask', '.model', '.pb', '.psf', '.residual', '.sumwt']

# import the FITS into MS format
os.system('rm -rf sim_MS/'+fname+'.ms')
importuvfits(fitsfile='sim_uvfits/'+fname+'.uvfits', vis='sim_MS/'+fname+'.ms')

# - Make a dirty image to guide a clean mask
for ext in extens:
    os.system('rm -rf sim_images/im_'+fname+'_dirty'+ext)
tclean(vis='sim_MS/'+fname+'.ms', 
       imagename='sim_images/im_'+fname+'_dirty', specmode='cube', 
       start=chanstart, width=chanwidth, nchan=nchan, outframe='LSRK', 
       restfreq='230.538GHz', imsize=512, cell='0.02arcsec', 
       deconvolver='multiscale', scales=cleanscales, niter=0, 
       weighting='natural', interactive=False, nterms=1, 
       restoringbeam='common')

# - Make a keplerian mask from the (noise-free) dirty image
os.system('rm -rf sim_images/im_'+fname+'_dirty.mask.image')
make_mask('sim_images/im_'+fname+'_dirty.image', inc=inc, PA=PA, mstar=mstar, 
          dist=150., vlsr=0.0, r_max=rmax, nbeams=1.3)

# - Make a CLEAN image 
for ext in extens:
    os.system('rm -rf sim_images/im_'+fname+ext)
tclean(vis='sim_MS/'+fname+'.ms', 
       imagename='sim_images/im_'+fname, specmode='cube', 
       start=chanstart, width=chanwidth, nchan=nchan, outframe='LSRK', 
       restfreq='230.538GHz', imsize=512, cell='0.02arcsec', 
       deconvolver='multiscale', scales=cleanscales, niter=10000000, 
       threshold=rms, weighting='natural', 
       mask='sim_images/im_'+fname+'_dirty.mask.image', interactive=False, 
       nterms=1, restoringbeam='common')
