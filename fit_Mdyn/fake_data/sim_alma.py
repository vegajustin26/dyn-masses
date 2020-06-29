import os
import numpy as np

def mk_template_uvfits(cube, conf=5, dt=15, vel_span=None, ch_width=None,
                       vsys=0.0, restfreq=230.538e9):

    # parse inputs
    conf_str = str(conf)
    dt_str = str(dt)+'min'

    # configuration file
    cfg_dir = '/pool/asha0/casa-pipeline-release-5.6.1-8.el6/data/alma/simmos/'
    cfg_str = cfg_dir + 'alma.cycle7.' + conf_str + '.cfg'

    # generate (u,v) tracks
    os.chdir('template_sims/')
    default('simobserve')
    simobserve(project=cube+'.sim', skymodel='../template_cubes/'+cube+'.fits', 
               antennalist=cfg_str, totaltime=dt_str, integration='30s', 
               thermalnoise='', refdate='2021/05/01', mapsize='10arcsec')

    # if no velocity span or channel width are specified, extract the values
    # from the template cube and adopt them
    if np.logical_or(vel_span is None, ch_width is None):
        importfits('../template_cubes/'+cube+'.fits', 'temp.image',
                   overwrite=True, defaultaxes=True,
                   defaultaxesvalues=['', '', '', 'I'])
        imhead('temp.image', mode='list')
        freq0 = imhead('temp.image', mode='get', hdkey='CRVAL4')['value']
        idx0 = imhead('temp.image', mode='get', hdkey='CRPIX4')
        nfreq = np.array(imhead('temp.image', mode='get', hdkey='SHAPE'))[3]
        dfreq = imhead('temp.image', mode='get', hdkey='CDELT4')['value']
        freqs = freq0 + (np.arange(nfreq) - idx0) * dfreq
        vel = 2.99792458000000e5 * (1. - freqs / restfreq)
        def_vel_span = np.abs(np.min(vel))
        def_ch_width = np.around(np.mean(np.diff(vel)), 2)
    if vel_span is None:
        vel_span = def_vel_span
    if ch_width is None:
        ch_width = def_ch_width
    
    # set the velocity channels for binning / regridding
    chanstart = ('%.2f' % (vsys - vel_span)) + 'km/s'
    chanwidth = str(ch_width) + 'km/s'
    nchan = np.int(2 * np.ceil(vel_span / ch_width) + 1)
    print(chanstart, chanwidth, nchan)

    # rebin and regrid
    infile = cube+'.sim/'+cube+'.sim.alma.cycle7.'+conf_str+'.ms'
    outfile = cube+'.sim/'+cube+'.sim.alma.cycle7.'+conf_str+'_cvel.ms'
    os.system('rm -rf '+outfile)
    mstransform(vis=infile, outputvis=outfile, keepflags=False, 
                datacolumn='data', regridms=True, mode='velocity', 
                start=chanstart, width=chanwidth, nchan=nchan, outframe='LSRK', 
                veltype='radio', restfreq=str(restfreq / 1e9)+'GHz')

    # convert to a UVFITS file
    fitsout = 'template_cfg'+conf_str+'_'+dt_str+\
              '_dv'+str(ch_width)+'kms'+\
              '_v0'+('%.2f' % (vsys - vel_span))+'kms'+\
              '_nch'+str(nchan)+'.uvfits'
    exportuvfits(vis=outfile, fitsfile='../template_uvfits/'+fitsout,
                 datacolumn='data', overwrite=True)
    os.chdir('../')

    # notification
    print('Wrote a template UVFITS file to \n      template_uvfits/'+fitsout)
