import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import yaml
import time
import scipy.constants as sc
from scipy import integrate
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable


# constants
AU = sc.au * 1e2
mu = 2.37
m_p = sc.m_p * 1e3
kB = sc.k * 1e7
G = sc.G * 1e3
msun = 1.989e33


# power-law form
def powerlaw(x, y0, q, x0=1.0):
    """ Simple powerlaw function. """
    return y0 * (x / x0) ** q


# Dartois temperature format
def Temp(R, THETA, **args):

    # convert grid to cylindrical coordinates
    rcyl = R * np.sin(THETA)
    zcyl = R * np.cos(THETA)

    # parse parameters
    r0, T0mid, T0atm = args["rT0"] * AU, args["T0mid"], args["T0atm"]
    Tqmid, Tqatm = args["Tqmid"], args["Tqatm"]
    delta, ZqHp = args["delta"], args["ZqHp"]
    Mstar = args["M_star"]

    # boundary radial profiles
    Tmid = powerlaw(rcyl, T0mid, Tqmid, r0)
    Tatm = powerlaw(rcyl, T0atm, Tqatm, r0)

    # characteristic height of atmosphere
    Hp = np.sqrt(kB * Tmid * rcyl**3 / (G * Mstar * msun * mu * m_p))
    zatm = ZqHp * Hp

    # two-dimensional structure
    T = Tatm + (Tmid - Tatm) * \
        (np.cos(np.pi * zcyl / (2*zatm)))**(2*delta)
    T = np.cos(np.pi * zcyl / (2*zatm))**(2*delta)
    T = np.where(abs(zcyl) < zatm, (Tmid - Tatm) * T, 0.0)

    return T + Tatm


def Tgrad_z(R, THETA, **args):

    # convert grid to cylindrical coordinates
    rcyl = R * np.sin(THETA)
    zcyl = R * np.cos(THETA)

    # temperatures
    # parse parameters
    r0, T0mid, T0atm = args["rT0"] * AU, args["T0mid"], args["T0atm"]
    Tqmid, Tqatm = args["Tqmid"], args["Tqatm"]
    delta, ZqHp = args["delta"], args["ZqHp"]
    Mstar = args["M_star"]

    # boundary radial profiles
    Tmid = powerlaw(rcyl, T0mid, Tqmid, r0)
    Tatm = powerlaw(rcyl, T0atm, Tqatm, r0)

    # characteristic height of atmosphere
    Hp = np.sqrt(kB * Tmid * rcyl**3 / (G * Mstar * msun * mu * m_p))
    zatm = ZqHp * Hp

    # two-dimensional structure
    T = Tatm + (Tmid - Tatm) * \
        (np.cos(np.pi * zcyl / (2*zatm)))**(2*delta)
    T = np.cos(np.pi * zcyl / (2*zatm))**(2*delta)
    T = np.where(abs(zcyl) < zatm, (Tmid - Tatm) * T, 0.0)
    T += Tatm

    # analytical gradient
    dT = -2 * delta * (Tmid - Tatm) * \
            (np.cos(np.pi * zcyl / (2*zatm)))**(2*delta-1) * \
            np.sin(np.pi * zcyl / (2*zatm)) * np.pi / (2 * zatm) / T
    dT[abs(zcyl) > zatm] = 0

    return dT




# load parameters
conf = open('demo.yaml')
config = yaml.load(conf, Loader=yaml.FullLoader)
grid_par = config["grid"]
disk_par = config["disk_params"]
host_par = config["host_params"]
setups = config["setup"]
conf.close()


# set up radius grid
nr = grid_par["spatial"]["nr"]
r_i = grid_par["spatial"]["r_min"] * AU
r_o = grid_par["spatial"]["r_max"] * AU
rwalls = np.logspace(np.log10(r_i), np.log10(r_o), nr+1)
rsph = np.average([rwalls[:-1], rwalls[1:]], axis=0)

# set up altitude grid
nt = grid_par["spatial"]["nt"]
t_offset = 0.1	
t_min = t_offset
t_max = 0.5 * np.pi + t_offset
t_walls = np.logspace(np.log10(t_min), np.log10(t_max), nt+1)
t_walls = 0.5 * np.pi + t_offset - t_walls[::-1]
t_min = t_walls.min()
t_max = t_walls.max()
thet = np.average([t_walls[:-1], t_walls[1:]], axis=0)

# grid quantities
RR, TT = np.meshgrid(rsph, thet)
rcyl, zcyl = RR * np.sin(TT), RR * np.cos(TT)


# calculate the 2-D temperature structure
T_args = disk_par["temperature"]["arguments"]
args = {**T_args, **host_par}
temperature = Temp(RR, TT, **args)

# enforce boundary conditions
max_temp, min_temp = 5e2, 5e0 
temperature[temperature > max_temp] = max_temp
temperature[temperature <= min_temp] = min_temp

# temperature gradient
dTdz = Tgrad_z(RR, TT, **args)


# density parameters
d_args = disk_par["gas_surface_density"]["arguments"]
min_dens = 1e0
max_dens = 1e20


# ******
# calculate the *un-normalized* density profile in the usual silly way
#
# ******
rhog = np.zeros_like(RR)
for j in range(len(thet)):
    for i in range(len(rsph)):

        # cylindrical coordinates
        r = RR[j,i] * np.sin(TT[j,i])
        z = RR[j,i] * np.cos(TT[j,i])

        # define a special z grid for integration (zg)
        zmin, zmax, nz = 0.1, 5.*r, 1024
        zg = np.logspace(np.log10(zmin), np.log10(zmax + zmin), nz) - zmin

        # if z >= zmax, return the minimum density
        if (z >= zmax):
            rhoz = min_dens * m_p * mu
        # else, do the full calculation
        else:
            # STEP 1: CALCULATE d ln T / dz
            # temperature parameters
            r0, T0mid, T0atm = args["rT0"] * AU, args["T0mid"], args["T0atm"]
            Tqmid, Tqatm = args["Tqmid"], args["Tqatm"]
            delta, ZqHp = args["delta"], args["ZqHp"]
            Mstar = args["M_star"]

            # boundary radial profiles
            Tmid = powerlaw(r, T0mid, Tqmid, r0)
            Tatm = powerlaw(r, T0atm, Tqatm, r0)

            # characteristic height of atmosphere
            Hp = np.sqrt(kB * Tmid * r**3 / (G * Mstar * msun * mu * m_p))
            zatm = ZqHp * Hp

            # two-dimensional temperature structure
            T = Tatm + (Tmid - Tatm) * \
                (np.cos(np.pi * zg / (2*zatm)))**(2*delta)
            T[zg > zatm] = Tatm

            # vertical temperature gradient
            dT = -2 * delta * (Tmid - Tatm) * \
                    (np.cos(np.pi * zg / (2*zatm)))**(2*delta-1) * \
                    np.sin(np.pi * zg / (2*zatm)) * np.pi / (2 * zatm) / T
            dT[zg > zatm] = 0

            
            # STEP 2: CALCULATE VERTICAL GRAVITY TERM
            cs2 = kB * T / (mu * m_p)
            gz = G * Mstar * msun * zg / cs2
            gz /= np.hypot(r, zg)**3


            # STEP 3: CALCULATE d ln rho / dz
            dlnpdz = -dT - gz


            # STEP 4: NUMERICALLY INTEGRATE TO GET rho(zg)
            lnp = integrate.cumtrapz(dlnpdz, zg, initial=0)
            rho0 = np.exp(lnp)


            # STEP 5: NORMALIZE
            Rc, sig0, pg1 = d_args["Rc"] * AU, d_args["sig0"], d_args["pg1"]
            pg2 = d_args["pg2"]
            sigg = powerlaw(r, sig0, -pg1, Rc) * np.exp(-(r / Rc)**pg2)
            rho = 0.5 * sigg * rho0 / integrate.trapz(rho0, zg)


            # STEP 6: INTERPOLATE BACK TO ORIGINAL GRIDPOINT
            f = interp1d(zg, rho)
            rhog[j,i] = np.max([f(z), min_dens * m_p * mu])



_ = np.loadtxt('demo/amr_grid.inp', skiprows=5, max_rows=1)
nr, nt = np.int(_[0]), np.int(_[1])
Rwalls = np.loadtxt('demo/amr_grid.inp', skiprows=6, max_rows=nr+1)
Twalls = np.loadtxt('demo/amr_grid.inp', skiprows=nr+7, max_rows=nt+1)
rsph  = 0.5*(Rwalls[:-1] + Rwalls[1:])
thet = 0.5*(Twalls[:-1] + Twalls[1:])
RR, TT = np.meshgrid(rsph, thet)

# recover temperature structure
temp_in = np.loadtxt('demo/gas_temperature.inp', skiprows=2)
temperature = np.reshape(temp_in, (nt, nr))

# recover gas density structure
rho_in = np.loadtxt('demo/gas_density.inp', skiprows=2)
rhog = np.reshape(rho_in, (nt, nr))


# COMPUTE VELOCITY FIELD
vkep2 = G * Mstar * msun * (np.sin(TT))**2 / RR

# pressure
Pgas = rhog * kB * temperature / mu / m_p
dPdr = np.gradient(Pgas, RR[0,:], axis=1) * np.sin(TT) + \
       np.gradient(Pgas, TT[:,0], axis=0) * np.cos(TT) / RR
vprs2 = RR * np.sin(TT) * dPdr / rhog

velo = np.sqrt(vkep2 + vprs2)
print(velo / 1e5)



def _grab_axes(fig):
    """ Split off the axis from the figure if provided. """
    if fig is None:
        fig, ax = plt.subplots()
    else:
        ax = fig.axes[0]
    return fig, ax

def _gentrify_structure_ax(ax, full=True):
    """ Gentrify the plot. """
    ax.set_xlim([2, 500])
    ax.set_xscale('log')
    ax.set_ylim([0.0, 0.7])
    ax.set_xlabel("$R$ [au]")
    ax.set_ylabel("$\pi$/2 - $\Theta$")
    if not full:
        ax.set_ylim(0.0, ax.get_ylim()[1])


# velocities
tempfig = None
fig, ax = _grab_axes(tempfig)
R = rsph / AU
THETA = 0.5*np.pi - thet[::-1]
RHO = velo[::-1]
toplot = np.vstack([RHO[::-1], RHO])
toplot = np.log10(toplot / 1e5)
yaxis = np.concatenate([-THETA[::-1], THETA])

contourf_kwargs = {} 
levels = np.linspace(toplot.min(), toplot.max(), 50)
levels = np.linspace(0, 1.5, 50)
levels = contourf_kwargs.pop("levels", levels)
cmap = contourf_kwargs.pop("cmap", "viridis")
im = ax.contourf(R, yaxis, toplot, levels=levels,
                 cmap=cmap, **contourf_kwargs)

full = False
cax = make_axes_locatable(ax)
cax = cax.append_axes("right", size="4.5%" if full else "3%",
                      pad="2.25%" if full else "1.5%")
cb = plt.colorbar(im, cax=cax)
cb.set_label(r"$\log(v_{\phi}\,\,[{\rm km/s}])$",
             rotation=270, labelpad=15)
_gentrify_structure_ax(ax, full=full)
fig.savefig('vel.png')



# load the standard velocities
velo_in = np.loadtxt('demo/gas_velocity.inp', skiprows=2)
velo_demo = np.reshape(velo_in[:,2], (nt, nr))
dvel = (velo_demo - velo) / velo_demo
tempfig = None
fig, ax = _grab_axes(tempfig)
R = rsph / AU
THETA = 0.5*np.pi - thet[::-1]
RHO = dvel[::-1]
toplot = np.vstack([RHO[::-1], RHO])
yaxis = np.concatenate([-THETA[::-1], THETA])

contourf_kwargs = {}
levels = np.linspace(toplot.min(), toplot.max(), 50)
levels = np.linspace(-0.1, 0.1, 50)
levels = contourf_kwargs.pop("levels", levels)
cmap = contourf_kwargs.pop("cmap", "RdBu")
im = ax.contourf(R, yaxis, toplot, levels=levels,
                 cmap=cmap, **contourf_kwargs)

full = False
cax = make_axes_locatable(ax)
cax = cax.append_axes("right", size="4.5%" if full else "3%",
                      pad="2.25%" if full else "1.5%")
cb = plt.colorbar(im, cax=cax, ticks=np.arange(-1, 1, 50))
cb.set_label(r"$\log(v_{\phi}\,\,[{\rm km/s}])$",
             rotation=270, labelpad=15)
_gentrify_structure_ax(ax, full=full)
fig.savefig('diff_vel.png')

