from scipy.interpolate import griddata
from grid import grid
from disk import disk
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import yaml
import scipy.constants as sc
from scipy import integrate



# generate the spherical coordinate grid and associated control files
sim_grid = grid('testrich_iso', writegrid=True)

# generate a structure model on a cylindrical grid
hyd_disk = disk('testrich_hyd', plotstruct=False)
iso_disk = disk('testrich_iso', plotstruct=False)


# grid points
z = iso_disk.zvals
r = iso_disk.rvals

# grab density, temperature structures
rho_hyd = hyd_disk.density_g
rho_iso = iso_disk.density_g
T_hyd = hyd_disk.temperature
T_iso = hyd_disk.temperature

# define a radius
index = 100
print(r[index])
print(hyd_disk.Hp[index])


# grab vertical density, temperature profiles
rhoz_iso = rho_iso[:,index]
rhoz_hyd = rho_hyd[:,index]
Tz_iso = T_iso[:,index]
Tz_hyd = T_hyd[:,index]
sigma_g = iso_disk.sigma_g[index]
H_iso = iso_disk.scaleheight()


# manual calculation

# constants
msun = 1.989e33
AU = sc.au * 1e2
mu = 2.37
m_p = sc.m_p * 1e3
kB = sc.k * 1e7
G = sc.G * 1e3

# stellar mass
Mstar = iso_disk.mstar 

# vertically isothermal density prescription
Hp = iso_disk.Hp[index]
rho_vi = sigma_g * np.exp(-0.5*(z/Hp)**2) / (np.sqrt(2.*np.pi)*Hp*AU)

# analytic vertical temperature gradient (dlnT/dz)
delta = 2.0
Tatm = 30. * (r[index] / 30.)**-0.5
Tmid = 30. * (r[index] / 30.)**-0.5
z_atm = 4.0 * hyd_disk.Hp[index]
T = Tatm + (Tmid - Tatm) * (np.cos(np.pi * z / (2 * z_atm)))**(2.*delta)
T[z > z_atm] = Tatm
# clip temperatures
min_temp = 5e0  # minimum temperature in [K]
max_temp = 5e2  # maximum temperature in [K]
T = np.clip(T, min_temp, max_temp)
dlnTdz = -2 * delta * (Tmid-Tatm) * (np.cos(np.pi*z/(2*z_atm)))**(2*delta-1) * \
         np.sin(np.pi*z/(2*z_atm)) * np.pi / (2.*z_atm*AU) / T
dlnTdz[z > z_atm] = 0

# vertical density gradient (d ln(rho) / dz)
#gz = G * Mstar * z*AU / ((r[index]*AU)**2 + (z*AU)**2)**1.5
gz = G * Mstar * z*AU / (r[index]*AU)**3
dlnpdz = -mu * m_p * gz / (kB * T) - dlnTdz

# numerical integration 
lnp = integrate.cumtrapz(dlnpdz, z*AU, initial=0)
rho_temp = np.exp(lnp)
A = 0.5 * sigma_g / integrate.trapz(rho_temp, z*AU)
rho_vg = A * rho_temp

# clip densities
min_dens = 1e3  # minimum gas density in [H2/cm**3]
max_dens = 1e20  # maximum gas density in [H2/cm**3]
rho_vg = np.clip(rho_vg, min_dens * mu * m_p, max_dens * mu * m_p)
rho_vi = np.clip(rho_vi, min_dens * mu * m_p, max_dens * mu * m_p)


plt.semilogy(z, rhoz_iso, 'C0')
plt.semilogy(z, rho_vi, '--b')
plt.semilogy(z, rhoz_hyd, 'C1')
plt.semilogy(z, rho_vg, '--r')
plt.axis([0, 2, 1e-21, 1e-8])
plt.show()





