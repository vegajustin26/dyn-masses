import numpy as np
from scipy.interpolate import interp1d

spt = [35., 38., 40., 42., 45., 48., 50., 52., 55., 57., 60., 61., 62., 63., 65., 65., 66., 67., 68., 69.]
teff = [6600., 6130., 5930., 5690., 5430., 5180., 4870., 4710., 4210., 4020., 3900., 3720., 3560., 3410., 3190., 2980., 2860., 2770., 2670., 2570.]

in_spts = [53., 53., 55.5, 55.5, 55.5, 56., 56.5, 58.5, 60., 60., 60.3, 60.4, 60.5, 60.6, 60.6, 61, 61.1, 61.7, 62., 62.3, 62.3, 62.3, 62.5, 62.6, 63., 63.2, 63.3, 64.5]

in_names = ['K3', 'K3', 'K5.5', 'K5.5', 'K5.5', 'K6', 'K6.5', 'K8.5', 'M0', 'M0', 'M0.3', 'M0.4', 'M0.5', 'M0.6', 'M0.6', 'M1', 'M1.1', 'M1.7', 'M2', 'M2.3', 'M2.3', 'M2.3', 'M2.5', 'M2.6', 'M3', 'M3.2', 'M3.3', 'M4.5']


in_spts = [52., 54., 54.5, 57., 58., 60.1, 60.8, 61.3, 61.5, 61.8, 62.4, 63.5, 63.7, 64., 64.3, 64.7, 64.8, 62.8]

in_names = ['K2', 'K4', 'K4.5', 'K7', 'K8', 'M0.1', 'M0.8', 'M1.3', 'M1.5', 'M1.8', 'M2.4', 'M3.5', 'M3.7', 'M4', 'M4.3', 'M4.7', 'M4.8', 'M2.8']

in_spts = [52.0, 65.1, 65.5, 60.9, 62.3, 64.8, 62.8]
in_names = ['K2', 'M5.1', 'M5.5', 'M0.9', 'M2.3', 'M4.8', 'M2.8']

tint = interp1d(spt, teff)

for i in range(len(in_spts)):
   in_teffs = tint(in_spts[i])
   if (in_spts[i] >= 60.):
       err_hi = tint(in_spts[i]-0.3) - in_teffs
       err_lo = in_teffs - tint(in_spts[i]+0.3)
       uncert = 0.5*(err_hi+err_lo)
   if (in_spts[i] < 60.):
       err_hi = tint(in_spts[i]-0.5) - in_teffs
       err_lo = in_teffs - tint(in_spts[i]+0.5)
       uncert = 0.5*(err_hi+err_lo)
   print('%s  %i  % i' % (in_names[i], in_teffs, uncert))