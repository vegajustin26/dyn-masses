import numpy as np
import matplotlib.pyplot as plt



AU = 1.49597871e13
Msun = 1.98847542e33
mu_gas = 2.37
m_H = 1.67353284e-24
f_H = 0.706
f_H2 = 0.8
G = 6.67408e-8
kB = 1.38064852e-16
PI = np.pi


Mstar = 1.32 * Msun


r = 4.39 * AU
z = np.logspace(-2, 2, 256) * AU


# midplane temperature profile
T0_mid = 50.
q_mid = 0.5
T_mid = T0_mid * (r / (10.*AU))**(-q_mid)

# pressure scale height
Omega = np.sqrt(G * Mstar / r**3)
c_s = np.sqrt(kB * T_mid / (mu_gas * m_H))
Hp = c_s / Omega 

# atmosphere temperature profile
T0_atm = 80.
q_atm = 0.5
delta = 1.
T_atm = T0_atm * (r / (10.*AU))**(-q_atm)

z_atm = Hp * 4
Trz = T_atm + (T_mid - T_atm) * np.cos(PI * z / (2 * z_atm))**(2 * delta)
Trz[z > z_atm] = T_atm

plt.plot(z/r, Trz)
plt.show()



