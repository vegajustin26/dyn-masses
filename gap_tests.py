import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import yaml
import scipy.constants as sc


# constants
AU = sc.au * 1e2


def powerlaw(x, y0, q, x0=1.0):
    """ Simple powerlaw function. """
    return y0 * (x / x0) ** q


# load parameters
conf = open('demo.yaml')
config = yaml.load(conf, Loader=yaml.FullLoader)
grid_par = config["grid"]
disk_par = config["disk_params"]
setups = config["setup"]
conf.close()


# set up initial radius grid
nr = grid_par["spatial"]["nr"]
r_i = grid_par["spatial"]["r_min"] * AU
r_o = grid_par["spatial"]["r_max"] * AU
rwalls = np.logspace(np.log10(r_i), np.log10(r_o), nr+1)
rcm = np.average([rwalls[:-1], rwalls[1:]], axis=0)
rau = rcm / AU


if setups["substruct"]:

    # fetch substructure parameters
    args = disk_par["substructure"]["arguments"]
    rgaps, wgaps = args["rgaps"], args["wgaps"]
    ngaps = len(rgaps)

    # for each feature, refine the radial grid around it
    dr = 3.0	# sigma
    for i in range(ngaps):
        reg = (rau > rgaps[i]-dr*wgaps[i]) & (rau < rgaps[i]+dr*wgaps[i])
        print(len(rau[reg]))
        if (len(rau[reg]) < 60):
            rexc = rau[~reg]
            rextra = rgaps[i] + np.linspace(-dr*wgaps[i], dr*wgaps[i], 61)
            print(rextra)
            rau = np.sort(np.concatenate((rexc, rextra)))

    nr = len(rau)
    rcm = rau * AU





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
