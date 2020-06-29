import os, sys, time
import yaml
import numpy as np
from gofish import imagecube
from scipy.integrate import trapz, cumtrapz
from scipy.interpolate import interp1d
from astropy.io import fits
import matplotlib.pyplot as plt

# models
mns = 1 + np.arange(5)
cols = ['k', 'firebrick', 'tomato', 'skyblue', 'royalblue', 'darkslateblue']
T0 = np.array([28., 42., 52., 60., 83.])  * np.sqrt(150./100.)
print(T0)
sys.exit()
rco = np.array([40., 115., 200., 285., 540.])

fig, ax = plt.subplots()
for i in range(5, 0, -1):
    print(i)

    mname = 'phys'+str(i)+'_i40'

    # load parameters of model
    conf = open(mname + ".yaml")
    config = yaml.load(conf, Loader=yaml.FullLoader)
    dpars = config["disk_params"]
    opars = config["outputs"]
    hpars = config["host_params"]
    conf.close()

    # fetch what you need from .yaml
    incl = opars["geometry"]["incl"]
    PA = opars["geometry"]["PA"]
    dpc = opars["geometry"]["dpc"]
    zbot = dpars["abundance"]["arguments"]["zrmin"] 
    ztop = dpars["abundance"]["arguments"]["zrmax"]
    z0 = 0.5 * (zbot + ztop)

    # generate and load 8th moment map (made with bettermoments)
    os.chdir(mname)
    os.system('bettermoments '+mname+'_co.fits -method eighth -clip 0')
    cube = imagecube(mname+'_co_M8.fits')
    os.chdir('../')

    # extract radial Tpeak profile (from 8th moment map)
    x, y, dy = cube.radial_profile(inc=incl, PA=PA, z0=z0, phi=1., unit='K') 
    Tb = cube.jybeam_to_Tb(data=y, nu=230.538e9)

    # plot a simple power-law prescription
    xmod = np.logspace(0, np.log10(rco[i-1]), 128) 
    ax.plot(xmod, T0[i-1] * (xmod/100.)**-0.5, ':', color=cols[i], 
            alpha=0.6, lw=2)

    # extract and plot the model temperature profile at the CO layer
    _ = np.loadtxt(mname+'/amr_grid.inp', skiprows=5, max_rows=1)
    nr, nt = np.int(_[0]), np.int(_[1])
    Rw = np.loadtxt(mname+'/amr_grid.inp', skiprows=6, max_rows=nr+1)
    Tw = np.loadtxt(mname+'/amr_grid.inp', skiprows=nr+7, max_rows=nt+1)
    Rgrid = 0.5*(Rw[:-1] + Rw[1:])
    Tgrid = 0.5*(Tw[:-1] + Tw[1:])
    T_in = np.loadtxt(mname+'/gas_temperature.inp', skiprows=2)
    temp = np.reshape(T_in, (nt, nr))
    modl_zr = 1 / np.tan(Tgrid)

    # associate the top and bottom layers
    top_layer = np.where(np.abs(modl_zr - ztop) == 
                         np.min(np.abs(modl_zr - ztop)))[0]
    bot_layer = np.where(np.abs(modl_zr - zbot) == 
                         np.min(np.abs(modl_zr - zbot)))[0]
    r_top = Rgrid * np.sin(Tgrid[top_layer[0]]) / 1.496e13
    temp_top = temp[top_layer[0], :]
    r_bot = Rgrid * np.sin(Tgrid[bot_layer[0]]) / 1.496e13
    temp_bot = temp[bot_layer[0], :]
    top_int = interp1d(r_top, temp_top, fill_value='extrapolate')
    bot_int = interp1d(r_bot, temp_bot, fill_value='extrapolate')
    xmod = np.logspace(0, np.log10(rco[i-1]), 512) / dpc
    ax.fill_between(xmod * dpc, top_int(xmod * dpc), bot_int(xmod * dpc), 
                    color=cols[i], alpha=0.3, interpolate=True)


    # plot the radial Tb profile
    ax.plot(x * dpc, Tb, cols[i], lw=2)


ax.set_xlim([12.5, 800.])
ax.set_ylim([7, 300])
ax.set_xscale('log')
ax.set_yscale('log')
ax.text(20, 6.2, '20', color='k', ha='center', va='center')
ax.text(50, 6.2, '50', color='k', ha='center', va='center')
ax.text(200, 6.2, '200', color='k', ha='center', va='center')
ax.text(500, 6.2, '500', color='k', ha='center', va='center')
ax.text(12., 20, '20', color='k', ha='right', va='center')
ax.text(12., 50, '50', color='k', ha='right', va='center')
ax.text(12., 200, '200', color='k', ha='right', va='center')
ax.set_xticks([100])
ax.set_xticklabels(['100'])
ax.set_yticks([10, 100])
ax.set_yticklabels(['10', '100'])
ax.set_xlabel('radius  [au]')
ax.set_ylabel('temperature  [K]')
fig.savefig('Tb_from_phys.png')
