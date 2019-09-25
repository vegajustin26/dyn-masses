import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy import integrate

# constants
AU = 1.496e13
Msun = 1.989e33
mu = 2.37
m_H = 1.67e-24
PI = np.pi
G = 6.67e-8
kB = 1.380e-16

# vertical grid
z = AU * np.logspace(-3, 2, 1024)

# boundary condition
Sigg = 100.
r = 100. * AU

# stellar mass
Mstar = 1.0 * Msun

# thermal structure
Tmid = 30.
Tatm = 60.
Hp = np.sqrt(kB * Tmid / (mu*m_H)) / np.sqrt(G * Mstar / r**3)
delta = 1.
z_atm = 4 * Hp
T = Tatm + (Tmid - Tatm) * (np.cos(PI * z / (2 * z_atm)))**(2.*delta)
T[z > z_atm] = Tatm


# vertically isothermal density prescription
rho_vi = Sigg * np.exp(-0.5*(z/Hp)**2) / (np.sqrt(2.*PI)*Hp)


# analytic vertical temperature gradient (dlnT/dz)
dlnTdz = -2 * delta * (Tmid-Tatm) * (np.cos(PI*z/(2*z_atm)))**(2*delta-1) * \
         np.sin(PI*z/(2*z_atm)) * PI / (2.*z_atm) / T
dlnTdz[z > z_atm] = 0


# chintzy vertical temperature gradient
#dlnTdz = np.zeros_like(z)
#for iz in np.arange(1,len(z)):
#    dz = z[iz]-z[iz-1]
#    dlnTdz[iz] = (np.log(T[iz])-np.log(T[iz-1]))/dz


# vertical density gradient (d ln(rho) / dz)
gz = G * Mstar * z / (r**2 + z**2)**1.5
gz = G * Mstar * z / r**3
dlnpdz = -mu * m_H * gz / (kB * T) - dlnTdz


# numerical integration 
lnp = integrate.cumtrapz(dlnpdz, z, initial=0)
rho_temp = np.exp(lnp)

# normalize this!
A = 0.5 * Sigg / integrate.trapz(rho_temp, z)
rho_vg = A * rho_temp


plt.semilogy(z/AU, rho_vi, 'C0')
plt.semilogy(z/AU, rho_vg, '--C1')
plt.show()
