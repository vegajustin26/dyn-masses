import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import yaml
import scipy.constants as sc
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


def Tgrad(R, THETA, **args):

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
    dTana = -2 * delta * (Tmid - Tatm) * \
            (np.cos(np.pi * zcyl / (2*zatm)))**(2*delta-1) * \
            np.sin(np.pi * zcyl / (2*zatm)) * np.pi / (2 * zatm) / T
    dTana[abs(zcyl) > zatm] = 0

    # numerical gradient (in spherical coordinates)
    dT = -np.gradient(np.log(T), THETA[:,0], axis=0) / rcyl
    dT[abs(zcyl) > zatm] = 0

    # fractional difference from analytic
    diff_dT = (dT - dTana) / dTana
    print(np.nanmin(diff_dT), np.nanmax(diff_dT))

    return diff_dT




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


# calculate the 2-D temperature structure
T_args = disk_par["temperature"]["arguments"]
args = {**T_args, **host_par}
temperature = Temp(RR, TT, **args)

# enforce boundary conditions
max_temp, min_temp = 5e2, 5e0 
temperature[temperature > max_temp] = max_temp
temperature[temperature <= min_temp] = min_temp

# temperature gradient
dTdz = Tgrad(RR, TT, **args)



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
    ax.set_ylim([0.0, 0.5])
    ax.set_xlabel("$R$ [au]")
    ax.set_ylabel("$\pi$/2 - $\Theta$")
    if not full:
        ax.set_ylim(0.0, ax.get_ylim()[1])


# temperatures
tempfig = None
fig, ax = _grab_axes(tempfig)
R = rsph / AU
THETA = 0.5*np.pi - thet[::-1]
TEMP = temperature[::-1]
toplot = np.vstack([TEMP[::-1], TEMP])
yaxis = np.concatenate([-THETA[::-1], THETA])

contourf_kwargs = {} 
levels = np.linspace(toplot.min(), toplot.max(), 50)
levels = np.linspace(3, 300, 50)
levels = contourf_kwargs.pop("levels", levels)
cmap = contourf_kwargs.pop("cmap", "plasma")
im = ax.contourf(R, yaxis, toplot, levels=levels,
                 cmap=cmap, **contourf_kwargs)

full = False
cax = make_axes_locatable(ax)
cax = cax.append_axes("right", size="4.5%" if full else "3%",
                      pad="2.25%" if full else "1.5%")
cb = plt.colorbar(im, cax=cax, ticks=np.arange(0, 300, 50))
cb.set_label(r"$T\,\,[{\rm K}]$", rotation=270, labelpad=15)

_gentrify_structure_ax(ax, full=full)

fig.savefig('test.temp.png')



# temperature gradients
dTfig = None
fig, ax = _grab_axes(dTfig)
R = rsph / AU
THETA = 0.5*np.pi - thet[::-1]
#TEMP = np.log10(dTdz[::-1])
TEMP = dTdz[::-1]
toplot = np.vstack([TEMP[::-1], TEMP])
yaxis = np.concatenate([-THETA[::-1], THETA])

contourf_kwargs = {}
levels = np.linspace(toplot.min(), toplot.max(), 50)
levels = np.linspace(-16, -12, 50)
levels = np.linspace(-1, 1, 50)
levels = contourf_kwargs.pop("levels", levels)
cmap = contourf_kwargs.pop("cmap", "RdBu")
im = ax.contourf(R, yaxis, toplot, levels=levels,
                 cmap=cmap, **contourf_kwargs)

full = False
cax = make_axes_locatable(ax)
cax = cax.append_axes("right", size="4.5%" if full else "3%",
                      pad="2.25%" if full else "1.5%")
#cb = plt.colorbar(im, cax=cax, ticks=np.arange(-16, -12, 50))
cb = plt.colorbar(im, cax=cax, ticks=np.arange(-1, 1, 50))

cb.set_label(r"$T\,\,[{\rm K}]$", rotation=270, labelpad=15)

_gentrify_structure_ax(ax, full=full)

fig.savefig('test.gradT.png')











sys.exit()





# gas surface density
d_args = disk_par["gas_surface_density"]["arguments"]
s_args = disk_par["substructure"]["arguments"]
args = {**d_args, **s_args}
if disk_par["gas_surface_density"]["type"] == 'self_similar':
    try:
        Rc, sig0, pg1 = args["Rc"] * AU, args["sig0"], args["pg1"]
        pg2 = args.pop("pg2", 2.0 - pg1)
    except KeyError:
        raise ValueError("Specify at least `Rc`, `sig0`, `pg1`.")
    sigg = powerlaw(rcm, sig0, -pg1, Rc) * np.exp(-(rcm / Rc)**pg2)
    sig0 = powerlaw(rcm, sig0, -pg1, Rc) * np.exp(-(rcm / Rc)**pg2)

    if setups["substruct"]:
        # impose substructures
        rss, wss, dss = args["rgaps"], args["wgaps"], args["dgaps"]
        depl = 0.0
        for ig in range(len(rss)):
            rg, wg = rss[ig] * AU, wss[ig] * AU
            depl += (dss[ig] - 1.) * np.exp(-0.5*((rcm - rg) / wg)**2)
        sigg /= (1. + depl)

reg = (rau > rgaps[1]-dr*wgaps[1]) & (rau < rgaps[1]+dr*wgaps[1])
for i in range(len(rau[reg])):
    print((rau[reg])[i], sigg[reg][i], sig0[reg][i])


#plt.semilogy(rau, sigg, 'oC0')
#plt.semilogy(rau, sig0, 'oC1')
plt.loglog(rau, sigg, 'oC0')
plt.loglog(rau, sig0, 'oC1')
plt.axis([0.1, 1000, 0.001, 5000])
plt.show()
