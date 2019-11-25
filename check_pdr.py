import numpy as np
import sys
import os
import yaml
import build_structure as model
from look_utils import read_Tgas, read_nmol
import matplotlib.pyplot as plt


# no PDR / freezeout model
simple_grid = model.Grid('testmodel')
simple_nco = read_nmol(simple_grid, fname='testmodel/numberdens_co.inp')
simple_zr = 0.5*np.pi - simple_grid.theta_centers

# with PDR / freezeout
abund_grid = model.Grid('testabund')
abund_nco = read_nmol(abund_grid, fname='testabund/numberdens_co.inp')
abund_zr = 0.5*np.pi - abund_grid.theta_centers

index = 30
print(abund_grid.r_centers[index]/1.496e13)

plt.semilogy(simple_zr, simple_nco[index, :], 'C0')
plt.semilogy(abund_zr, abund_nco[index, :], '--C1')
plt.axis([0, 1, 1e-15, 1e9])
plt.show()


T_co = read_Tgas(abund_grid, fname='testmodel/gas_temperature.inp')
plt.plot(abund_zr, T_co[index, :], 'C0')
plt.show()


