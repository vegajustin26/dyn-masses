import os
import numpy as np
execfile('/home/sandrews/mypy/keplerian_mask/keplerian_mask.py')

# - fixed simulation parameters
fname = 'grid2_C8_co'
dt  = '33min'     
conf = '6'

# - what to do
sim_vis = True
sim_dirty = False
sim_mask = False
sim_clean = False
sim_nf = False

# - reconfigure inputs for use
pwv = 1.796
cfg = 'alma.cycle7.' + conf
cfg_str = 'config' + conf
cfg_dir = '/pool/asha0/casa-pipeline-release-5.6.1-8.el6/data/alma/simmos/'

# - fixed imaging parameters
cleanscales = [0, 5, 10, 15]
extens = ['.image', '.mask', '.model', '.pb', '.psf', '.residual', '.sumwt']

# - generate visibilities
if sim_vis:
    default('simobserve')
    simobserve(project=fname+'.sim', skymodel=fname+'.fits', integration='6s', 
               antennalist=cfg_dir+cfg+'.cfg', totaltime=dt, 
               thermalnoise='tsys-atm', user_pwv=pwv, overwrite=True, 
               refdate='2020/12/01', mapsize="8arcsec")

# - Make a dirty (noise-free) image to guide a clean mask
if sim_dirty:
    for ext in extens:
        os.system('rm -rf im_'+fname+'_'+cfg_str+'_'+dt+'_dirty'+ext)
    tclean(vis=fname+'.sim/'+fname+'.sim.'+cfg+'.ms', 
           imagename='im_'+fname+'_'+cfg_str+'_'+dt+'_dirty', 
           specmode='cube', start='-6.0km/s', width='0.1km/s', nchan=121,
           outframe='LSRK', restfreq='230.538GHz', imsize=512, 
           cell='0.02arcsec', deconvolver='multiscale', scales=cleanscales,
           niter=0, weighting='briggs', robust=0.5, interactive=False, 
           nterms=1, restoringbeam='common', 
           uvtaper='0.10arcsec')

# - Make a keplerian mask from the (noise-free) dirty image
if sim_mask:
    os.system('rm -rf im_'+fname+'_'+cfg_str+'_'+dt+'_dirty.mask.image')
    make_mask('im_'+fname+'_'+cfg_str+'_'+dt+'_dirty.image', inc=45, PA=310., 
              mstar=0.45, dist=150., vlsr=0.0, r_max=1.2, nbeams=1.5)

# - Make a CLEAN (noisy) image 
if sim_clean:
    for ext in extens:
        os.system('rm -rf im_'+fname+'_'+cfg_str+'_'+dt+ext)
    tclean(vis=fname+'.sim/'+fname+'.sim.'+cfg+'.noisy.ms', 
           imagename='im_'+fname+'_'+cfg_str+'_'+dt,
           specmode='cube', start='-6.0km/s', width='0.1km/s', nchan=121,
           outframe='LSRK', restfreq='230.538GHz', imsize=512, 
           cell='0.02arcsec', deconvolver='multiscale', scales=cleanscales,
           niter=10000000, threshold='5mJy', weighting='briggs', robust=0.5, 
           mask='im_'+fname+'_'+cfg_str+'_'+dt+'_dirty.mask.image', 
           interactive=False, nterms=1, restoringbeam='common',
           uvtaper='0.10arcsec')


# - Make a CLEAN (noise-free) image
if sim_nf:
    for ext in extens:
        os.system('rm -rf im_'+fname+'_'+cfg_str+'_'+dt+'_nf'+ext)
    tclean(vis=fname+'.sim/'+fname+'.sim.'+cfg+'.ms',
           imagename='im_'+fname+'_'+cfg_str+'_'+dt+'_nf',
           specmode='cube', start='-6.0km/s', width='0.1km/s', nchan=121,
           outframe='LSRK', restfreq='230.538GHz', imsize=512,
           cell='0.02arcsec', deconvolver='multiscale', scales=cleanscales,
           niter=10000000, threshold='5mJy', weighting='briggs', robust=0.5,
           mask='im_'+fname+'_'+cfg_str+'_'+dt+'_dirty.mask.image',
           interactive=False, nterms=1, restoringbeam='common')
