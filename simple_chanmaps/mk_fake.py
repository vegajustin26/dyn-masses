import os
import sys
import numpy as np
from mk_FITScube import mk_FITScube
from vis_sample import vis_sample
from vis_sample.file_handling import import_data_uvfits
from astropy.io import fits

# template file
template = 'rich_io.sim.config5.30min.noisy.specbin'

# output files
fout = 'rich_io_lowSNR'

# RMS noise per naturally weighted beam per channel in output
RMS = 5.25 * 5.	# in mJy

# parameters: theta = [inc, PA, mstar, Tb0, Tbq, r_max, vsys, dx, dy]
theta = np.array([30., 40., 1.75, 150., -0.5, 300., 0.0, 0.0, 0.0])
FOV = 8.0
dist = 150.
Npix = 256
Tbmax = 500.
restfreq = 230.538e9
fixed = FOV, dist, Npix, Tbmax, restfreq

### - extract the velocities from the template file
dat = fits.open('fake_data/'+template+'.uvfits')
hdr = dat[0].header
freq0 = hdr['CRVAL4']
indx0 = hdr['CRPIX4']
nchan = hdr['NAXIS4']
dfreq = hdr['CDELT4']
freqs = freq0 + (np.arange(nchan) - indx0 + 1) * dfreq
vel = 2.9979245800000e10 * (1. - freqs / restfreq) / 100.

### - compute a model cube (SkyImage object)
foo = mk_FITScube(inc=theta[0], PA=theta[1], mstar=theta[2], FOV=FOV, 
                  dist=dist, Npix=Npix, Tb0=theta[3], Tbq=theta[4], 
                  r_max=theta[5], vsys=theta[6], Tbmax=Tbmax, 
                  restfreq=restfreq, vel=vel)

### - sample it on the template (u,v) spacings: NOISE FREE
os.system('rm -rf fake_data/'+fout+'_noiseless.uvfits')
vis_sample(imagefile=foo, uvfile='fake_data/'+template+'.uvfits', 
           mu_RA=theta[7], mu_DEC=theta[8], mod_interp=False, 
           outfile='fake_data/'+fout+'_noiseless.uvfits')

### - clone and corrupt the datafile according to the desired noise
vis = import_data_uvfits('fake_data/'+fout+'_noiseless.uvfits')
clone = fits.open('fake_data/'+fout+'_noiseless.uvfits')
clone_data = clone[0].data
clone_hd = clone[0].header

npol = clone_hd['NAXIS3']
nvis, nchan = vis.VV.shape[1], vis.VV.shape[0]
clone_vis = clone_data['data']
sig_noise = 1e-3 * RMS * np.sqrt(nvis)
if (npol == 2):
    darr = np.zeros([vis.VV.shape[1], vis.VV.shape[0], 2, 3])
    noise1 = np.random.normal(0, sig_noise, (nvis, nchan)) + \
             np.random.normal(0, sig_noise, (nvis, nchan))*1.j
    noise2 = np.random.normal(0, sig_noise, (nvis, nchan)) + \
             np.random.normal(0, sig_noise, (nvis, nchan))*1.j
    darr[:,:,0,0] = np.real(vis.VV.T) + noise1.real * np.sqrt(2.)
    darr[:,:,1,0] = np.real(vis.VV.T) + noise2.real * np.sqrt(2.)
    darr[:,:,0,1] = np.imag(vis.VV.T) + noise1.imag * np.sqrt(2.)
    darr[:,:,1,1] = np.imag(vis.VV.T) + noise2.imag * np.sqrt(2.)
    darr[:,:,0,2] = 0.5 * np.ones_like(vis.wgts) / sig_noise**2
    darr[:,:,1,2] = 0.5 * np.ones_like(vis.wgts) / sig_noise**2
else:
    darr = np.zeros([vis.VV.shape[1], vis.VV.shape[0], 2, 3])
    noise = np.random.normal(0, sig_noise, (nvis, nchan)) + \
            np.random.normal(0, sig_noise, (nvis, nchan))*1.j
    darr[:,:,0,0] = np.real(vis.VV.T) + noise.real
    darr[:,:,0,1] = np.imag(vis.VV.T) + noise.imag
    darr[:,:,0,2] = np.ones_like(vis.wgts) / sig_noise**2

clone_data['data'] = np.expand_dims(np.expand_dims(np.expand_dims(darr,1),1),1)
clone.writeto('fake_data/'+fout+'_noisy.uvfits', overwrite=True)
print('Wrote fake_data/'+fout+'_noisy.uvfits')
