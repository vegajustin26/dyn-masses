import os
import numpy as np
execfile('/home/sandrews/mypy/keplerian_mask/keplerian_mask.py')

# - fixed simulation parameters
fname = 'grid2_Alo_co'
dt_min = 30
dt = str(dt_min)+'min'     
conf = '6'

# - noise-associated parameters
rms = 5.25   			# mJy/beam per channel, target
nant = 43			# number of antennas
nbase = nant * (nant - 1) / 2.	# number of baselines
nchan = 0.2 / 0.092 		# number of native / output channels
npol = 2			# number of polarizations
nint = dt_min * 60. / 30.
rms_in = rms * np.sqrt(nchan * npol * nbase * nint) * 1.08
rms_simple = ('%.1f' % rms_in) + 'mJy'

# - what to do
sim_vis = True
sim_dirty = True
sim_mask = True
sim_clean = True
sim_nf = False

# - reconfigure inputs for use
cfg = 'alma.cycle7.' + conf
cfg_str = 'config' + conf
cfg_dir = '/pool/asha0/casa-pipeline-release-5.6.1-8.el6/data/alma/simmos/'

# - fixed imaging parameters
cleanscales = [0, 5, 10, 15]
extens = ['.image', '.mask', '.model', '.pb', '.psf', '.residual', '.sumwt']


# - generate visibilities and corrupt them
if sim_vis:
    default('simobserve')
    simobserve(project=fname+'.sim', skymodel=fname+'.fits', integration='30s', 
               antennalist=cfg_dir+cfg+'.cfg', totaltime=dt, thermalnoise='', 
               overwrite=True, refdate='2020/12/01', mapsize="8arcsec")

    # clean up old files
    os.system('rm -rf '+fname+'.sim/'+fname+'.sim.'+cfg_str+'.'+dt+'.ms')
    os.system('rm -rf '+fname+'.sim/'+fname+'.sim.'+cfg_str+'.'+dt+'.noisy.ms')

    # change filename for better tracking
    os.system('mv '+fname+'.sim/'+fname+'.sim.'+cfg+'.ms ' + \
                    fname+'.sim/'+fname+'.sim.'+cfg_str+'.'+dt+'.ms')

    # make a copy of noise-free MS to corrupt
    os.system('cp -r '+fname+'.sim/'+fname+'.sim.'+cfg_str+'.'+dt+'.ms ' + \
                       fname+'.sim/'+fname+'.sim.'+cfg_str+'.'+dt+'.noisy.ms')

    # corrupt the noisy MS
    sm.openfromms(fname+'.sim/'+fname+'.sim.'+cfg_str+'.'+dt+'.noisy.ms')
    sm.setnoise(mode='simplenoise', simplenoise=rms_simple)
    sm.corrupt()
    sm.done()
    

# - Make a dirty (noise-free) image to guide a clean mask
if sim_dirty:
    for ext in extens:
        os.system('rm -rf im_'+fname+'_'+cfg_str+'_'+dt+'_dirty'+ext)
    tclean(vis=fname+'.sim/'+fname+'.sim.'+cfg_str+'.'+dt+'.ms', 
           imagename='im_'+fname+'_'+cfg_str+'_'+dt+'_dirty', 
           specmode='cube', start='-6.0km/s', width='0.2km/s', nchan=61,
           outframe='LSRK', restfreq='230.538GHz', imsize=512, 
           cell='0.02arcsec', deconvolver='multiscale', scales=cleanscales,
           niter=0, weighting='natural', #weighting='briggs', robust=0.5, 
           interactive=False, 
           nterms=1, restoringbeam='common') 


# - Make a keplerian mask from the (noise-free) dirty image
if sim_mask:
    os.system('rm -rf im_'+fname+'_'+cfg_str+'_'+dt+'_dirty.mask.image')
    make_mask('im_'+fname+'_'+cfg_str+'_'+dt+'_dirty.image', inc=45, PA=310., 
              mstar=0.2, dist=150., vlsr=0.0, r_max=0.7, nbeams=1.5)


# - Make a CLEAN (noisy) image 
if sim_clean:
    for ext in extens:
        os.system('rm -rf im_'+fname+'_'+cfg_str+'_'+dt+ext)
    tclean(vis=fname+'.sim/'+fname+'.sim.'+cfg_str+'.'+dt+'.noisy.ms', 
           imagename='im_'+fname+'_'+cfg_str+'_'+dt,
           specmode='cube', start='-6.0km/s', width='0.2km/s', nchan=61,
           outframe='LSRK', restfreq='230.538GHz', imsize=512, 
           cell='0.02arcsec', deconvolver='multiscale', scales=cleanscales,
           niter=10000000, threshold='5mJy', weighting='natural', #'briggs', robust=0.5, 
           mask='im_'+fname+'_'+cfg_str+'_'+dt+'_dirty.mask.image', 
           interactive=False, nterms=1, restoringbeam='common')


# - Make a CLEAN (noise-free) image
if sim_nf:
    for ext in extens:
        os.system('rm -rf im_'+fname+'_'+cfg_str+'_'+dt+'_nf'+ext)
    tclean(vis=fname+'.sim/'+fname+'.sim.'+cfg_str+'.'+dt+'.ms',
           imagename='im_'+fname+'_'+cfg_str+'_'+dt+'_nf',
           specmode='cube', start='-6.0km/s', width='0.1km/s', nchan=121,
           outframe='LSRK', restfreq='230.538GHz', imsize=512,
           cell='0.02arcsec', deconvolver='multiscale', scales=cleanscales,
           niter=10000000, threshold='5mJy', weighting='briggs', robust=0.5,
           mask='im_'+fname+'_'+cfg_str+'_'+dt+'_dirty.mask.image',
           interactive=False, nterms=1, restoringbeam='common')
