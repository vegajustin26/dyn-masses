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
    min_dens = 1e3  # minimum gas density in [H2/cm**3]
    max_dens = 1e20  # maximum gas density in [H2/cm**3]
    min_temp = 5e0  # minimum temperature in [K]
    max_temp = 5e2  # maximum temperature in [K]


    def __init__(self, modelname, grid, writestruct=True):

        # load parameters
        conf = open(modelname + ".yaml")
        config = yaml.load(conf, Loader=yaml.FullLoader)
        self.host_params = config["host_params"]
        self.disk_params = config["disk_params"]
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

        self.rhog = np.zeros_like(self.rcyl)
        rho_args = self.disk_params["gas_surface_density"]["arguments"]
        nmol_args = self.disk_params["abundance"]["arguments"]
        self.dens_args = {**rho_args, **self.temp_args, **nmol_args}

        self.nmol = np.zeros_like(self.rcyl)


        # structure loop
        for j in range(len(self.tvals)):
            for i in range(len(self.rvals)):

                # cylindrical quantities
                r = self.rr[j,i] * np.sin(self.tt[j,i])
                z = self.rr[j,i] * np.cos(self.tt[j,i])

                # temperature
                self.temperature[j,i] = self.Temp(r, z, **self.temp_args)

                # gas density and number density (of given molecule)
                self.rhog[j,i], self.nmol[j,i] = self.Density_g(r, z, 
                                                     **self.dens_args)





        # temporary plotter!
        _ = self.plot_temp(full=False)
        _.savefig('test_temp.png')


        _ = self.plot_dens(full=False)
        _.savefig('test_dens.png')


        _ = self.plot_nmol(full=False)
        _.savefig('test_nmol.png')










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
            else:
                if (z >= zatm): T = Tatm
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
            return self.powerlaw(self.rvals, sig0, -pg1, redge)


    def Density_g(self, r, z, **args):

        """ Gas densities """

        # define a special z grid for integration (zg)
        zmin, zmax, nz = 0.1, 5.*r, 1024
        zg = np.logspace(np.log10(zmin), np.log10(zmax + zmin), nz) - zmin

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
        f = interp1d(zg, rho, bounds_error=False, fill_value=(np.max(rho, 0)))

        # gas density at specified height
        rhoz = f(z)


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

        # 'layer' setup, with constant abundance in a layer between specified
        # radial and height (z / r) bounds
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

            


        

    # Analytical functions.

    @staticmethod
    def powerlaw(x, y0, q, x0=1.0):
        """ Simple powerlaw function. """
        return y0 * (x / x0) ** q

    @staticmethod
    def _grab_axes(fig):
        """ Split off the axis from the figure if provided. """
        if fig is None:
            fig, ax = plt.subplots()
        else:
            ax = fig.axes[0]
        return fig, ax

    @staticmethod
    def _gentrify_structure_ax(ax, full=True):
        """ Gentrify the plot. """
        ax.set_xlim([2, 500])
        ax.set_xscale('log')
        ax.set_ylim([0.0, 0.6])
        ax.set_xlabel("$R$ [au]")
        ax.set_ylabel("$\pi$/2 - $\Theta$")
        #    ax.set_aspect(1)
        if not full:
            ax.set_ylim(0.0, ax.get_ylim()[1])

    def plot_temp(self, fig=None, contourf_kwargs=None, full=True):
        fig, ax = self._grab_axes(fig)
        R = self.rvals / self.AU
        THETA = 0.5*np.pi - self.tvals[::-1]
        TEMP = self.temperature[::-1]
        toplot = np.vstack([TEMP[::-1], TEMP])
        yaxis = np.concatenate([-THETA[::-1], THETA])

        contourf_kwargs = {} if contourf_kwargs is None else contourf_kwargs
        levels = np.linspace(toplot.min(), toplot.max(), 50)
        levels = np.linspace(3, 300, 50)
        levels = contourf_kwargs.pop("levels", levels)
        cmap = contourf_kwargs.pop("cmap", "plasma")
        im = ax.contourf(R, yaxis, toplot, levels=levels,
                         cmap=cmap, **contourf_kwargs)

        cax = make_axes_locatable(ax)
        cax = cax.append_axes("right", size="4.5%" if full else "3%",
                              pad="2.25%" if full else "1.5%")
        cb = plt.colorbar(im, cax=cax, ticks=np.arange(0, 300, 50))
        cb.set_label(r"$T\,\,[{\rm K}]$", rotation=270, labelpad=15)

        self._gentrify_structure_ax(ax, full=full)
        return fig


    def plot_dens(self, fig=None, contourf_kwargs=None, full=True):
        fig, ax = self._grab_axes(fig)
        R = self.rvals / self.AU
        THETA = 0.5*np.pi - self.tvals[::-1]
        RHO = self.rhog[::-1]
        toplot = np.vstack([RHO[::-1], RHO])
        toplot = np.log10(toplot / (sc.m_p*1e3) / 2.37)
        yaxis = np.concatenate([-THETA[::-1], THETA])

        contourf_kwargs = {} if contourf_kwargs is None else contourf_kwargs
        levels = np.linspace(toplot.min(), toplot.max(), 50)
        levels = np.linspace(3, 17, 50)
        levels = contourf_kwargs.pop("levels", levels)
        cmap = contourf_kwargs.pop("cmap", "bone_r")
        im = ax.contourf(R, yaxis, toplot, levels=levels,
                         cmap=cmap, **contourf_kwargs)

        cax = make_axes_locatable(ax)
        cax = cax.append_axes("right", size="4.5%" if full else "3%",
                              pad="2.25%" if full else "1.5%")
        cb = plt.colorbar(im, cax=cax, ticks=np.arange(-30, 30, 2))
        cb.set_label(r"$\log_{10}(n_{\rm H_2}\,\,[{\rm cm^{-2}}])$",
                     rotation=270, labelpad=15)
        self._gentrify_structure_ax(ax, full=full)
        return fig


    def plot_nmol(self, fig=None, contourf_kwargs=None, full=True):
        fig, ax = self._grab_axes(fig)
        R = self.rvals / self.AU
        THETA = 0.5*np.pi - self.tvals[::-1]
        NMOL = self.nmol[::-1]
        toplot = np.vstack([NMOL[::-1], NMOL])
        toplot = np.log10(toplot)
        yaxis = np.concatenate([-THETA[::-1], THETA])

        contourf_kwargs = {} if contourf_kwargs is None else contourf_kwargs
        levels = np.linspace(toplot.min(), toplot.max(), 50)
        levels = np.linspace(-12, 12, 50)
        levels = contourf_kwargs.pop("levels", levels)
        cmap = contourf_kwargs.pop("cmap", "afmhot_r")
        im = ax.contourf(R, yaxis, toplot, levels=levels,
                         cmap=cmap, **contourf_kwargs)

        cax = make_axes_locatable(ax)
        cax = cax.append_axes("right", size="4.5%" if full else "3%",
                              pad="2.25%" if full else "1.5%")
        cb = plt.colorbar(im, cax=cax, ticks=np.arange(-30, 30, 2))
        cb.set_label(r"$\log_{10}(n_{\rm CO}\,\,[{\rm cm^{-2}}])$",
                     rotation=270, labelpad=15)
        self._gentrify_structure_ax(ax, full=full)
        return fig










#    # dust surface density parameters
#    self.Sig0_d = disk_params["Sig0_d"]
#    self.R0_d = disk_params["R0_d"] * AU
#    self.pd1 = disk_params["pd1"]
#    self.pd2 = disk_params["pd2"]

#        # gas surface density parameters
#        self.Sig0_g = disk_params["Sig0_g"]
#        self.R0_g = disk_params["R0_g"] * AU
#        self.pg1 = disk_params["pg1"]
#        self.pg2 = disk_params["pg2"]
#        self.sigma_pdr = 10.**(disk_params["sig_pdr"])
#        self.depl_pdr = disk_params["depl_pdr"]
#        self.T_frz = disk_params["T_frz"]
#        self.depl_frz = disk_params["depl_frz"]
#        self.fmol = 10.**(disk_params["fmol"])

#        # thermal structure parameters
#        self.T0_mid = disk_params["T0_mid"]
#        self.q_mid = disk_params["q_mid"]
#        self.T0_atm = disk_params["T0_atm"]
#        self.q_atm = disk_params["q_atm"]
#        self.delta = disk_params["delta"]

        # non-thermal broadening (as fraction of local sound speed)
#        self.xi = disk_params["xi"]


#    # DUST SURFACE DENSITY PROFILE
#    def Sigma_d(self, r):
#        sd = self.Sig0_d * (r / self.R0_d)**(-self.pd1) * \
#             np.exp(-(r / self.R0_d)**self.pd2)    
#        return sd


#    # GAS SURFACE DENSITY PROFILE
#    def Sigma_g(self, r):
#        sg = self.Sig0_g * (r / self.R0_g)**(-self.pg1) * \
#             np.exp(-(r / self.R0_g)**self.pg2)
#        return sg


#    # MIDPLANE TEMPERATURE PROFILE
#    def T_mid(self, r):
#        return self.T0_mid * (r / (30.*AU))**(-self.q_mid)


#    # ATMOSPHERE TEMPERATURE PROFILE (saturates at z_atm)
#    def T_atm(self, r):
#        return self.T0_atm * (r / (30.*AU))**(-self.q_atm)


#    # PRESSURE SCALE HEIGHTS
#    def Hp(self, r):
#        Omega = np.sqrt(G * self.Mstar / r**3)
#        c_s = np.sqrt(kB * self.T_mid(r) / (mu_gas * m_H))
#        return c_s / Omega


#    # 2-D TEMPERATURE STRUCTURE
#    def Temp(self, r, z):
#        self.z_atm = self.Hp(r) * 4	    # fix "atmosphere" to 4 * Hp
#        Trz =  self.T_atm(r) + (self.T_mid(r) - self.T_atm(r)) * \
#               np.cos(PI * z / (2 * self.z_atm))**(2.*self.delta)
#        if (z > self.z_atm): Trz = self.T_atm(r)
#        return Trz


#    # VERTICAL TEMPERATURE GRADIENT (dlnT / dz)
#    def logTgrad(self, r, z):
#        dT = -2 * self.delta * (self.T_mid(r) - self.T_atm(r)) * \
#             (np.cos(PI * z / (2 * self.z_atm)))**(2*self.delta-1) * \
#             np.sin(PI * z / (2 * self.z_atm)) * PI / (2 * self.z_atm) / \
#             self.Temp(r,z)
#        if (z > self.z_atm): dT = 0
#        return dT


#    # 2-D DUST DENSITY STRUCTURE
#    def rho_d(self, r, z):
#        z_dust = self.Hp(r) * 0.2	# fix dust scale height to lower
#        dnorm = self.Sigma_d(r) / (np.sqrt(2 * PI) * z_dust)
#        return dnorm * np.exp(-0.5 * (z / z_dust)**2)


#    # 2-D GAS DENSITY STRUCTURE
#    def rho_g(self, r, z):
#    
#        # set an upper atmosphere boundary
#        z_max = 10 * self.z_atm
#        PDR = False
#
#        # grid of z values for integration
#        zvals = np.logspace(np.log10(0.1), np.log10(z_max+0.1), 1024) - 0.1
#
#        # load temperature gradient
#        dlnTdz = self.logTgrad(r, z)
# 
#        # density gradient
#        gz = G * self.Mstar * zvals / (r**2 + zvals**2)**1.5
#        dlnpdz = -mu_gas * m_H * gz / (kB * self.Temp(r,z)) - dlnTdz
#
#        # numerical integration
#        lnp = integrate.cumtrapz(dlnpdz, zvals, initial=0)
#        dens0 = np.exp(lnp)
#
#        # normalized densities
#        dens = 0.5 * self.Sigma_g(r) * dens0 / integrate.trapz(dens0, zvals)
#        
#        # interpolator for moving back onto the spatial grid
#        f = interp1d(zvals, np.squeeze(dens), bounds_error=False, 
#                     fill_value=(np.max(dens), 0))
#
#        # properly normalized gas densities
#        rho_gas = np.float(f(z))
#
#        ## boolean indicator if this height is in the molecule's PDR
#        # find index of nearest zvals cell
#        index = np.argmin(np.abs(zvals-z))	
#        # integrate the vertical density profile down to that height
#        sig_index = integrate.trapz(dens[index:], zvals[index:])
#        # criterion for photodissociation
#        if (sig_index < (self.sigma_pdr * mu_gas * m_H * f_H)): 
#            PDR = True

#        return rho_gas, PDR
 

#    # 2-D MOLECULAR NUMBER DENSITY STRUCTURE
#    def nmol(self, r, z):
    
        # read in gas volume densities
#        rho_gas, PDR = self.rho_g(r,z)

        # abundance variations
##        if (self.Temp(r,z) < self.T_frz): 
##            Xmol = self.depl_frz * self.fmol
##        elif PDR: 
##            Xmol = self.depl_pdr * self.fmol
##        else: 
#        Xmol = self.fmol
#
#        return rho_gas * f_H2 * Xmol / (mu_gas * m_H)


#    # GAS VELOCITY STRUCTURE
#    def velocity(self, r):
    
#        vkep = np.sqrt(G * self.Mstar / r)

#        return vkep


#    # MICROTURBULENCE
#    def vturb(self, r, z):
#
#        c_s = np.sqrt(kB * self.Temp(r, z) / (mu_gas * m_H))
#        dv = self.xi * c_s   

#        return dv


    # WRITE OUT RADMC FILES
#    def write_Model(self, Grid):
        
#        # file headers
#        if (self.do_dust == 1):
#            dustdens_inp = open(self.mdir+'dust_density.inp', 'w')
#            dustdens_inp.write('1\n%d\n1\n' % Grid.ncells)

#            dusttemp_inp = open(self.mdir+'dust_temperature.dat', 'w')
#            dusttemp_inp.write('1\n%d\n1\n' % Grid.ncells)

#        if (self.do_gas == 1):
#            gasdens_inp = open(self.mdir+'gas_density.inp', 'w')
#            gasdens_inp.write('1\n%d\n' % Grid.ncells)
#
#            gastemp_inp = open(self.mdir+'gas_temperature.inp', 'w')
#            gastemp_inp.write('1\n%d\n' % Grid.ncells)
#
#            nmol_inp = open(self.mdir+'numberdens_'+self.molecule+'.inp', 'w')
#            nmol_inp.write('1\n%d\n' % Grid.ncells)
#
#            vel_inp = open(self.mdir+'gas_velocity.inp', 'w')
#            vel_inp.write('1\n%d\n' % Grid.ncells)
#
#            turb_inp = open(self.mdir+'microturbulence.inp', 'w')
#            turb_inp.write('1\n%d\n' % Grid.ncells)
#
#        # populate files
#        for phi in Grid.phi_centers:
#            for theta in Grid.theta_centers:
#                for r in Grid.r_centers:
#                    r_cyl = r * np.sin(theta)
#                    z = r * np.cos(theta)
#
#                    if (self.do_dust == 1):
#                        dusttemp_inp.write('%.6e\n' % self.Temp(r_cyl, z))
#                        dustdens_inp.write('%.6e\n' % self.rho_d(r_cyl, z))
#
#                    if (self.do_gas == 1):
#                        gastemp_inp.write('%.6e\n' % self.Temp(r_cyl, z))
#                        gasdens, dum = self.rho_g(r_cyl, z)
#                        gasdens_inp.write('%.6e\n' % gasdens)
#                        nmol_inp.write('%.6e\n' % self.nmol(r_cyl, z))
#                        vel_inp.write('0 0 %.6e\n' % self.velocity(r_cyl))
#                        turb_inp.write('%.6e\n' % self.vturb(r_cyl, z))
#
#        # close files
#        if (self.do_dust == 1):
#            dusttemp_inp.close()
#            dustdens_inp.close()
#        if (self.do_gas == 1):
#            gastemp_inp.close()
#            gasdens_inp.close()
#            nmol_inp.close()
#            vel_inp.close()
#            turb_inp.close()
