import os
import numpy as np
execfile('/home/sandrews/mypy/keplerian_mask/keplerian_mask.py')
#execfile('/pool/asha0/SCIENCE/p484/reduction_scripts/reduction_utils.py')


# - fixed simulation parameters
fname = 'grid2_F_co_i60'
dt  = '2100s'     # in hours
pwv = 1.262
cfg = 'alma.cycle7.6'
cfg_dir = '/pool/asha0/casa-pipeline-release-5.6.1-8.el6/data/alma/simmos/'

# - fixed imaging parameters
cleanscales = [0, 5, 10, 30, 90]


# - generate visibilities
#default('simobserve')
#simobserve(project=fname+'.sim', skymodel=fname+'.fits', integration='6s', 
#           antennalist=cfg_dir+cfg+'.cfg', totaltime=dt, 
#           thermalnoise='tsys-atm', user_pwv=pwv, overwrite=True, 
#           refdate='2020/12/01', mapsize="8arcsec")


# - Make a dirty (noise-free) image to guide a clean mask
#for ext in ['.image', '.mask', '.model', '.pb', '.psf', '.residual', '.sumwt']:
#    os.system('rm -rf im_'+fname+'_dirty'+ext)
#tclean(vis=fname+'.sim/'+fname+'.sim.'+cfg+'.ms', 
#       imagename='im_'+fname+'_dirty', 
#       specmode='cube', start='-6.0km/s', width='0.0928km/s', nchan=130,
#       outframe='LSRK', restfreq='230.538GHz', imsize=512, cell='0.02arcsec', 
#       deconvolver='multiscale', scales=cleanscales,
#       niter=0, weighting='briggs', robust=0.5, interactive=False, 
#       nterms=1, restoringbeam='common')

# - Make a keplerian mask from the (noise-free) dirty image
#os.system('rm -rf im_'+fname+'_dirty.mask.image')
#make_mask('im_'+fname+'_dirty.image',
#          inc=60., PA=310., mstar=2.0, dist=150., vlsr=0.0, zr=0.3, nbeams=1.5)

# - Make a CLEAN (noisy) image 
for ext in ['.image', '.mask', '.model', '.pb', '.psf', '.residual', '.sumwt']:
    os.system('rm -rf im_'+fname+ext)
tclean(vis=fname+'.sim/'+fname+'.sim.'+cfg+'.noisy.ms', imagename='im_'+fname, 
       specmode='cube', start='-6.0km/s', width='0.0928km/s', nchan=130,
       outframe='LSRK', restfreq='230.538GHz', imsize=512, cell='0.02arcsec', 
       deconvolver='multiscale', scales=cleanscales,
       niter=50000, threshold='5mJy', weighting='briggs', robust=0.5, 
       mask='im_grid2_F_co_i60_dirty.mask.image', 
       interactive=True, nterms=1, restoringbeam='common')


#for ext in ['.image', '.mask', '.model', '.pb', '.psf', '.residual', '.sumwt']:
#    os.system('rm -rf im_nf_'+fname+ext)
#tclean(vis=fname+'.sim/'+fname+'.sim.'+cfg+'.ms', imagename='im_nf_'+fname,
#       specmode='cube', imsize=512, deconvolver='multiscale', 
#       cell='0.02arcsec', mask='im_grid1_F_co_i60.mask',  
#       scales=cleanscales, gain=0.1, niter=50000, weighting='briggs', 
#       robust=0.5, threshold='5mJy', interactive=True, nterms=1, 
#       restoringbeam='common')



# 0 = 0.4939  ;  0.3885  ;  0.3785 
# 1 = 0.3953  ;  0.3867  ;  0.3765
# 2 = 0.3251  ;  0.3893  ;  0.3785
