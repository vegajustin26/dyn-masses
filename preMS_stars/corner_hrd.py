import numpy as np
import os
import sys
import corner
from post_summary import post_summary


name = 'IPTau_f085'


dat = np.load(name+'.age-mass.posterior.npz')
logM = dat['logM']
logAGE = dat['logAGE']

posts = np.column_stack([logM, logAGE])

levs = 1.-np.exp(-0.5*(np.arange(2)+1)**2)
fig = corner.corner(posts, plot_datapoints=False, bins=30, levels=levs, 
                    range=[(-0.5, 0.2), (5.0, 7.5)], no_fill_contours=True,
                    plot_density=False, color='b', labels=['log Mstar / Msun', 
                    'log age / yr'])

#print(np.percentile(logM, [50., 84.135, 15.865]))
#print(np.percentile(logAGE, [50., 84.135, 15.865]))

print(post_summary(logM, mu='median', prec=0.01))
print(post_summary(logAGE, mu='median', prec=0.01))


fig.savefig('corner_'+name+'.png')
fig.clf()
