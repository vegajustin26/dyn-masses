import os
import sys
import yaml
import numpy as np
import scipy.constants as sc
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import integrate 
from scipy.interpolate import interp1d


class disk:

    # constants
    msun = 1.989e33
    AU = sc.au * 1e2
    mu = 2.37
    m_p = sc.m_p * 1e3
    kB = sc.k * 1e7
    G = sc.G * 1e3

    # fixed values
    min_dens = 1e0  # minimum gas density in [H2/cm**3]
    max_dens = 1e20  # maximum gas density in [H2/cm**3]
    min_temp = 5e0  # minimum temperature in [K]
    max_temp = 5e2  # maximum temperature in [K]


    def __init__(self, modelname, grid, writestruct=True):

        # load parameters
        conf = open(modelname + ".yaml")
        config = yaml.load(conf, Loader=yaml.FullLoader)
        self.host_params = config["host_params"]
        self.disk_params = config["disk_params"]
        self.setups = config["setup"]
        conf.close()

        # stellar properties
        self.mstar = self.host_params["M_star"] * self.msun

        # grid properties (spherical coordinate system)
        self.rvals, self.tvals = grid.r_centers, grid.t_centers
        self.nr, self.nt = grid.nr, grid.nt
        self.rr, self.tt = np.meshgrid(self.rvals, self.tvals)

        # corresponding cylindrical quantities
        self.rcyl = self.rr * np.sin(self.tt)
        self.zcyl = self.rr * np.cos(self.tt)


        # compute gas temperature structure
        self.T_args = self.disk_params["temperature"]["arguments"]
        self.temp = self.temperature(**self.T_args)
        self.temp = np.clip(self.temp, self.min_temp, self.max_temp)


        # compute density structure
        self.sigma_g = self._parse_function("surface_density",
                           self.disk_params["gas_surface_density"])



    # Generic wrapper functions to parse arguments.
    def _parse_function(self, func_family, user_input):
        func = user_input["type"]
        try:
            args = user_input["arguments"]
        except KeyError:
            args = {}
        return eval("self._{}_{}(**args)".format(func_family, func))



    # Temperature Structure.

    def temperature(self, r=None, z=None, **args):
        # Dartois et al. 2003 (type II)
        if (self.disk_params["temperature"]["type"] == 'dartois'):
            try:
                r0 = args["rT0"] * self.AU
                T0mid, Tqmid = args["T0mid"], args["Tqmid"]
                T0atm, Tqatm = args.pop("T0atm", T0mid), args.pop("Tqatm", Tqmid)
                delta, ZqHp = args.pop("delta", 2.0), args.pop("ZqHp", 4.0)
            except KeyError:
                raise ValueError("Specify at least `rT0`, `T0mid`, `Tqmid`.")
            r = self.rcyl if r is None else r
            z = self.zcyl if z is None else z
            Tmid = self.powerlaw(r, T0mid, Tqmid, r0)
            Tatm = self.powerlaw(r, T0atm, Tqatm, r0)
            zatm = ZqHp * self.scaleheight(r=r, T=Tmid)
            T = np.cos(np.pi * z / 2.0 / zatm)**(2 * delta)
            T = np.where(abs(z) < zatm, (Tmid - Tatm) * T, 0.0)
            return T + Tatm

        # vertically isothermal
        if (self.disk_params["temperature"]["type"] == 'isoz'):
            try:
                r0 = args["rT0"] * self.AU
                T0mid, Tqmid = args["T0mid"], args["Tqmid"]
            except KeyError:
                raise ValueError("Specify at least `rT0`, `T0mid`, `Tqmid`.")
            return self.powerlaw(self.r, T0mid, Tqmid, r0)


    def scaleheight(self, r=None, T=None):
        T = self.temperature if T is None else T
        r = self.rcyl if r is None else r
        return self.soundspeed(T=T) / np.sqrt(self.G * self.mstar / r**3)


    def soundspeed(self, T=None):
        T = self.temperature if T is None else T
        return np.sqrt(self.kB * T / self.mu / self.m_p)


    def Tgrad_z(self, r, z, **args):

        # Dartois et al. 2003 (type II)
        if self.disk_params["temperature"]["type"] == 'dartois':
            try:
                r0, T0mid = args["rT0"] * self.AU, args["T0mid"]
                T0atm = args.pop("T0atm", T0mid)
                Tqmid = args["Tqmid"]
                Tqatm = args.pop("Tqatm", Tqmid)
                delta = args.pop("delta", 2.0)
                ZqHp = args.pop("ZqHp", 4.0)
            except KeyError:
                raise ValueError("Specify at least `rT0`, `T0mid`, `Tqmid`.")
            Tmid = self.powerlaw(r, T0mid, Tqmid, r0)
            Tatm = self.powerlaw(r, T0atm, Tqatm, r0)
            zatm = ZqHp * self.scaleheight(r, T=Tmid)
            T = Tatm + (Tmid - Tatm) * \
                (np.cos(np.pi * z / (2*zatm)))**(2*delta)
            dT = -2 * delta * (Tmid - Tatm) * \
                 (np.cos(np.pi * z / (2*zatm)))**(2*delta-1) * \
                 np.sin(np.pi * z / (2*zatm)) * np.pi / (2 * zatm) / T
            if (z.size > 1):
                dT[z >= zatm] = 0
            else:
                if (z >= zatm): dT = 0
            return dT

        # vertically isothermal
        if self.disk_params["temperature"]["type"] == 'isothermal':
            return 0



       
    # Density Structure.

    def sigma_g(self, r=None, **args):
    
        # Similarity-solution
        if self.disk_params["gas_surface_density"]["type"] == 'self-similar':
            try:
                Rc, sig0, p1 = args["Rc"] * self.AU, args["sig0"], args["p1"]
                p2 = args.pop("p2", 2.-p1)
            except KeyError:
                raise ValueError("Specify at least `Rc`, `sig0`, `pg1`.")
            r = self.rvals if r is None else r
            return self.powerlaw(r, sig0, -p1, Rc) * np.exp(-(r / Rc)**p2)

        # Power-law
        if self.disk_params["gas_surface_density"]["type"] == 'powerlaw':
            try:
                Rc, sig0, p1 = args["Rc"] * self.AU, args["sig0"], args["p1"]
                p2 = args.pop("p2", 10.)
            except KeyError:
                raise ValueError("Specify at least `Rc`, `sig0`, `pg1`.")
            r = self.rvals if r is None else r
            sig_in  = self.powerlaw(r, sig0, -p1, Rc)
            sig_out = self.powerlaw(r, sig0, -p2, Rc)
            sig = np.where(abs(r) < Rc, sig_in, sig_out)
            return sig

        
    def _density_hydrostatic(self, r=None, z=None, **args):

        # define a special z grid for integration (zg)
        zmin, zmax, nz = 0.1, 5.*r, 1024
        zg = np.logspace(np.log10(zmin), np.log10(zmax + zmin), nz) - zmin

        # if z >= zmax, return the minimum density
        if (z >= zmax): 
            rhoz = self.min_dens * self.m_p * self.mu
            try:
                xmol = args["xmol"]
            except KeyError:
                print("Specify at least `xmol`.")
            abund = xmol * args.pop("depletion", 1e-8)
            nmol = abund * rhoz / self.m_p / self.mu

        else:
            # vertical temperature profile
            Tz = self.temperature(r, zg, **args)

            # vertical temperature gradient
            dlnTdz = self.Tgrad_z(r, zg, **args)
                
            # vertical gravity
            gz = self.G * self.mstar * zg / (self.soundspeed(T=Tz))**2
            gz /= np.hypot(r, zg)**3

            # vertical density gradient
            dlnpdz = -dlnTdz - gz

            # numerical integration
            lnp = integrate.cumtrapz(dlnpdz, zg, initial=0)
            rho0 = np.exp(lnp)

            # normalize
            rho = 0.5 * self.Sigma_g(r, **args) * \
                  rho0 / integrate.trapz(rho0, zg)

            # set up an interpolator for going back to the original gridpoint
            f = interp1d(zg, rho) 

            # gas density at specified height
            rhoz = np.max([f(z), self.min_dens * self.m_p * self.mu])


            """ Molecular (number) densities """

            try:
                xmol = args["xmol"]
            except KeyError:
                print("Specify at least `xmol`.")

            # 'chemical' setup, with constant abundance in a layer between the 
            # freezeout temperature and photodissociation column
            if self.disk_params["abundance"]["type"] == 'chemical':

                # find the index of the nearest z cell
                index = np.argmin(np.abs(zg-z))

                # integrate the vertical density profile *down* to that height
                sig_index = integrate.trapz(rho[index:], zg[index:])
                NH2_index = sig_index / self.m_p / self.mu

                # note the column density for photodissociation
                Npd = 10.**(args.pop("logNpd", 21.11))

                # note the freezeout temperature
                Tfreeze = args.pop("tfreeze", 21.0)

                # compute abundance
                if ((NH2_index >= Npd) & 
                    (self.Temp(r, z, **self.temp_args) >= Tfreeze)):
                    abund = xmol
                else: abund = xmol * args.pop("depletion", 1e-8)

            # 'layer' setup, with constant abundance in a layer between 
            # specified radial and height (z / r) bounds
            if self.disk_params["abundance"]["type"] == 'layer':

                # identify the layer heights
                zrmin = args.pop("zrmin", 0.0)
                zrmax = args.pop("zrmax", 1.0)

                # identify the layer radii
                rmin = args.pop("rmin", self.rcyl.min()) * self.AU
                rmax = args.pop("rmax", self.rcyl.max()) * self.AU

                # compute abundance
                if ((r > rmin) & (r <= rmax) & (z/r > zrmin) & (z/r <= zrmax)):
                    abund = xmol
                else: abund = xmol * args.pop("depletion", 1e-8)


            # molecular number density
            nmol = rhoz * abund / self.m_p / self.mu


        return rhoz, nmol





    # Dynamical functions.

    def velocity(self, r, z, **args):

        # Keplerian rotation (treating or not the vertical height)
        vkep2 = self.G * self.mstar * r**2
        if args.pop("height", True):
            vkep2 /= np.hypot(r, z)**3
        else:
            vkep2 /= r**3


        # radial pressure contribution
        if args.pop("pressure", False):

            # compute the gas pressure
            rhogas0, nmol = self.Density_g(r, z, **self.dens_args)
            Tgas0 = self.Temp(r, z, **self.temp_args)
            Pgas0 = rhogas0 * self.kB * Tgas0 / self.m_p / self.mu

            # define some neighboring radii
            n_extra = 6
            extra_range = 1.2
            rext = np.logspace(np.log10(r / extra_range), 
                               np.log10(r * extra_range), 2*n_extra + 1)
 
            # compute radial pressure gradient
            Pgas = np.zeros_like(rext)
            rhog = np.zeros_like(rext)
            for ir in range(len(rext)):
                rhog[ir], nmol = self.Density_g(rext[ir], z, **self.dens_args)
                Tgas = self.Temp(rext[ir], z, **self.temp_args)
                Pgas[ir] = rhog[ir] * self.kB * Tgas / self.m_p / self.mu
            gradP = np.gradient(Pgas, rext)
            dPdr = gradP[n_extra]
        #    print(dPdr)

            # calculate pressure perturbation to velocity field
            if (rhog[n_extra] > 0.):
                vprs2 = r * dPdr / rhog[n_extra]
            else: vprs2 = 0.0
        else:
            vprs2 = 0.0

        #print(vkep2, vprs2, vkep2 + vprs2)



        # self-gravity
        vgrv2 = 0.0


        # return the combined velocity field
        return np.sqrt(vkep2 + vprs2 + vgrv2)


    def vturb(self, r, z, **args):
        try:
            xi = args["xi"]
        except KeyError:
            raise ValueError("Specify at least `xi`.")
        
        return self.soundspeed(T=self.Temp(r, z, **self.temp_args)) * xi

        

    # Analytical functions.

    @staticmethod
    def powerlaw(x, y0, q, x0=1.0):
        """ Simple powerlaw function. """
        return y0 * (x / x0) ** q
