import numpy as np
import os, sys
import matplotlib.pyplot as plt

# constants
kB = 1.381e-16
AU = 1.496e13
GG = 6.67e-8
mu = 2.37
mp = 1.67e-24
msun = 1.989e33

# z grid, r position
z = np.logspace(-3, 1.5, 128) * AU
r = 10 * AU

# T(z) structure
Tmid = 19.
Tatm = 5 * Tmid
Mstar = 0.1
hs_p = 2.5
ws_p = 0.4
cs = np.sqrt(kB * Tmid / (mu * mp))
om = np.sqrt(GG * Mstar * msun / r**3)
H = cs / om
fz = 0.5 * np.tanh( (z - hs_p * H) / (ws_p * H) ) + 0.5
T = (Tmid**4 + fz * Tatm**4)**0.25

# plot the Tz structure
plt.plot(z/AU, T)

# overplot 1, 2, 3 * H and z/r = 0.1, 0.2, 0.3
for i in [1, 2, 3, 4]:
    plt.plot([i*H/AU, i*H/AU], [Tmid, Tatm], ':k')
    plt.plot([0.1*i*r/AU, 0.1*i*r/AU], [Tmid, Tatm], '-r')

plt.show()
