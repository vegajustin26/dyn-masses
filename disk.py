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


        # grid properties
        self.rvals, self.tvals = grid.r_centers, grid.t_centers
        self.nr, self.nt = grid.nr, grid.nt
        self.rr, self.tt = np.meshgrid(self.rvals, self.tvals)

        # cylindrical quantities
        self.rcyl = self.rr * np.sin(self.tt)
        self.zcyl = self.rr * np.cos(self.tt)


        # structure setups
        self.temperature = np.zeros_like(self.rcyl)
        self.temp_args = self.disk_params["temperature"]["arguments"]

        if self.setups["incl_lines"]:
            self.rhog = np.zeros_like(self.rcyl)
            rho_args = self.disk_params["gas_surface_density"]["arguments"]
            nmol_args = self.disk_params["abundance"]["arguments"]
            self.dens_args = {**rho_args, **self.temp_args, **nmol_args}
            self.nmol = np.zeros_like(self.rcyl)
            self.vel = np.zeros_like(self.rcyl)
            self.vel_args = self.disk_params["rotation"]["arguments"]
            self.turb = np.zeros_like(self.rcyl)
            self.turb_args = self.disk_params["turbulence"]["arguments"]

            if writestruct:
                # file headers
                rhog_inp = open(modelname+'/gas_density.inp', 'w')
                rhog_inp.write('1\n%d\n' % (self.nr * self.nt))
                temp_inp = open(modelname+'/gas_temperature.inp', 'w')
                temp_inp.write('1\n%d\n' % (self.nr * self.nt))
                mname = self.setups["molecule"]
                nmol_inp = open(modelname+'/numberdens_'+mname+'.inp', 'w')
                nmol_inp.write('1\n%d\n' % (self.nr * self.nt))
                vel_inp = open(modelname+'/gas_velocity.inp', 'w')
                vel_inp.write('1\n%d\n' % (self.nr * self.nt))
                turb_inp = open(modelname+'/microturbulence.inp', 'w')
                turb_inp.write('1\n%d\n' % (self.nr * self.nt))

        if self.setups["incl_dust"]:
            self.rhod = np.zeros_like(self.rcyl)
            d_args = self.disk_params["dust_surface_density"]["arguments"]
            self.rhod_args = {**d_args, **self.temp_args}

            if writestruct:
                rhod_inp = open(modelname+'/dust_density.inp', 'w')
                rhod_inp.write('1\n%d\n1\n' % (self.nr * self.nt))
                tdust_inp = open(modelname+'/dust_temperature.dat', 'w')
                tdust_inp.write('1\n%d\n1\n' % (self.nr * self.nt))


#        rr = 100. * self.AU
#        tt = np.pi/2. - 0.05
#        r = rr * np.sin(tt)
#        z = rr * np.cos(tt)
#        vel = self.velocity(r, z, **self.vel_args)
#        sys.exit()



        # structure (and output) loop
        for j in range(len(self.tvals)):
            for i in range(len(self.rvals)):

                # cylindrical quantities
                r = self.rr[j,i] * np.sin(self.tt[j,i])
                z = self.rr[j,i] * np.cos(self.tt[j,i])

                # temperature
                self.temperature[j,i] = self.Temp(r, z, **self.temp_args)

                if self.setups["incl_lines"]:
                    # gas density and number density (of given molecule)
                    self.rhog[j,i], self.nmol[j,i] = self.Density_g(r, z, 
                                                         **self.dens_args)

                    # orbital motion
                    self.vel[j,i] = self.velocity(r, z, **self.vel_args)

                    # microturbulence
                    self.turb[j,i] = self.vturb(r, z, **self.turb_args)

                    # write into structure files
                    if writestruct:
                        temp_inp.write('%.6e\n' % self.temperature[j,i])
                        rhog_inp.write('%.6e\n' % self.rhog[j,i])
                        nmol_inp.write('%.6e\n' % self.nmol[j,i])
                        vel_inp.write('0 0 %.6e\n' % self.vel[j,i])
                        turb_inp.write('%.6e\n' % self.turb[j,i])

                if self.setups["incl_dust"]:
                    # dust density
                    self.rhod[j,i] = self.Density_d(r, z, **self.rhod_args)
             
                    # write into structure files
                    if writestruct:
                        tdust_inp.write('%.6e\n' % self.temperature[j,i])
                        rhod_inp.write('%.6e\n' % self.rhod[j,i])



        # close structure files                
        if writestruct:
            if self.setups["incl_lines"]:
                temp_inp.close()
                rhog_inp.close()
                nmol_inp.close()
                vel_inp.close()
                turb_inp.close()
            if self.setups["incl_dust"]:
                tdust_inp.close()
                rhod_inp.close()



    # Temperature functions.

    def Temp(self, r, z, **args):

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
            if (z.size > 1):
                T[z >= zatm] = Tatm
                T[T > self.max_temp] = self.max_temp
                T[T <= self.min_temp] = self.min_temp
            else:
                if (z >= zatm): T = Tatm
                if (T > self.max_temp): T = self.max_temp
                if (T <= self.min_temp): T = self.min_temp
            return T

        # vertically isothermal
        if self.disk_params["temperature"]["type"] == 'isothermal':
            try:
                r0, T0mid = args["rT0"], args["T0mid"]
                Tqmid = args["Tqmid"]
            except KeyError:
                raise ValueError("Specify at least `rT0`, `T0mid`, `Tqmid`.")
            return self.powerlaw(r, T0mid, Tqmid, r0)


    def scaleheight(self, r, T=None):
        """ Midplane gas pressure scale height """
        T = self.Temp if T is None else T
        Hp2 = self.kB * T * r**3 / (self.G * self.mstar * self.mu * self.m_p)
        return np.sqrt(Hp2) 


    def soundspeed(self, T=None):
        """ Gas soundspeed in [cm/s] """
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
            dT[z >= zatm] = 0
            return dT

        # vertically isothermal
        if self.disk_params["temperature"]["type"] == 'isothermal':
            return 0



       
    # Density functions.

    def Sigma_d(self, r, **args):

        # power-law
        if self.disk_params["dust_surface_density"]["type"] == 'powerlaw':
            try:
                rdedge, sigd0 = args["rdedge"] * self.AU, args["sigd0"]
                pd1 = args["pd1"]
            except KeyError:
                raise ValueError("Specify at least `rdedge`, `sigd0`, `pd1`.")
            sigd = self.powerlaw(r, sigd0, -pd1, rdedge)
            if (r > rdedge): 
                return 0
            else:
                return sigd


    def Sigma_g(self, r, **args):

        # similarity solution
        if self.disk_params["gas_surface_density"]["type"] == 'self_similar':
            try:
                Rc, sig0, pg1 = args["Rc"] * self.AU, args["sig0"], args["pg1"]
                pg2 = args.pop("pg2", 2.0 - pg1)
            except KeyError:
                raise ValueError("Specify at least `Rc`, `sig0`, `pg1`.")
            return self.powerlaw(r, sig0, -pg1, Rc) * np.exp(-(r / Rc)**pg2)

        # power-law
        if self.disk_params["gas_surface_density"]["type"] == 'powerlaw':
            try:
                redge, sig0 = args["redge"] * self.AU, args["sig0"]
                pg1 = args["pg1"]
            except KeyError:
                raise ValueError("Specify `redge`, `sig0`, `pg1`.")
            sigg = self.power(r, sig0, -pg1, redge)
            if (r > redge):
                return 0
            else:
                return sigg


    def Density_d(self, r, z, **args):

        # define a dust scale height
        r0, T0mid, Tqmid = args["rT0"] * self.AU, args["T0mid"], args["Tqmid"]
        Tmid = self.powerlaw(r, T0mid, Tqmid, r0)
        zdust = args.pop("hdust", 1.0) * self.scaleheight(r, T=Tmid)

        # use it to define vertical dimension
        dnorm = self.Sigma_d(r, **args) / (np.sqrt(2 * np.pi) * zdust)
        return dnorm * np.exp(-0.5 * (z / zdust)**2)


    def Density_g(self, r, z, **args):

        """ Gas densities """

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
            Tz = self.Temp(r, zg, **args)

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
            n_extra = 3
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

            # calculate pressure perturbation to velocity field
            if (rhog[n_extra] > 0.):
                vprs2 = r * dPdr / rhog[n_extra]
            else: vprs2 = 0.0

        vprs2 = 0.0


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
