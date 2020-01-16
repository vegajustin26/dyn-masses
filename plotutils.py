import os
import sys
import yaml
import numpy as np
import scipy.constants as sc
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable

class plotutils:

    # constants
    msun = 1.989e33
    AU = sc.au * 1e2
    mu = 2.37
    m_p = sc.m_p * 1e3
    kB = sc.k * 1e7
    G = sc.G * 1e3


    def __init__(self, modelname):

        # setups
        conf = open(modelname + ".yaml")
        config = yaml.load(conf, Loader=yaml.FullLoader)
        self.grids = config["grid"]["spatial"]
        self.setups = config["setup"]
        self.rmin = self.grids["r_min"]
        self.rmax = self.grids["r_max"]
        conf.close()


        # recover the spatial grid
        #nr, nt = self.grids["nr"], self.grids["nt"]
        _ = np.loadtxt(modelname+'/amr_grid.inp', skiprows=5, max_rows=1)
        nr, nt = np.int(_[0]), np.int(_[1])
        Rwalls = np.loadtxt(modelname+'/amr_grid.inp', 
                            skiprows=6, max_rows=nr+1)
        Twalls = np.loadtxt(modelname+'/amr_grid.inp', 
                            skiprows=nr+7, max_rows=nt+1)
        self.Rgrid  = 0.5*(Rwalls[:-1] + Rwalls[1:])
        self.Tgrid  = 0.5*(Twalls[:-1] + Twalls[1:])

        # recover temperature structure
        temp_in = np.loadtxt(modelname+'/gas_temperature.inp', skiprows=2)
        self.temp = np.reshape(temp_in, (nt, nr))
        _ = self.plot_temp(full=False)
        _.savefig(modelname+'/temp.png')

        # recover temperature gradient structure
        tgrad_in = np.loadtxt(modelname+'/gas_tempgradient.inp', skiprows=2)
        self.tgrad = np.reshape(tgrad_in, (nt, nr))
        _ = self.plot_tgrad(full=False)
        _.savefig(modelname+'/tgrad.png')

        if self.setups["incl_lines"]:
            # recover gas density structure
            rho_in = np.loadtxt(modelname+'/gas_density.inp', skiprows=2)
            self.rhog = np.reshape(rho_in, (nt, nr))
            _ = self.plot_dens(full=False)
            _.savefig(modelname+'/dens.png')

            # recover abundance structure
            nmol_in = np.loadtxt(modelname+'/numberdens_' + 
                                 self.setups["molecule"] + '.inp', skiprows=2)
            self.nmol = np.reshape(nmol_in, (nt, nr))
            _ = self.plot_nmol(full=False)
            _.savefig(modelname+'/nmol.png')

            # recover velocity structure
            vel_in = np.loadtxt(modelname+'/gas_velocity.inp', skiprows=2)
            self.vel = np.reshape(vel_in[:,2], (nt, nr))
            _ = self.plot_rotation(full=False)
            _.savefig(modelname+'/vel.png')

        if self.setups["incl_dust"]:
            # recover dust density structure
            rhod_in = np.loadtxt(modelname+'/dust_density.inp', skiprows=3)
            self.rhod = np.reshape(rhod_in, (nt, nr))
            _ = self.plot_dustdens(full=False)
            _.savefig(modelname+'/dustdens.png')



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
        #ax.set_xlim([40, 160])
        ax.set_xlim([2., 500])
        ax.set_xscale('log')
        ax.set_ylim([0.0, np.pi/2.])
        ax.set_ylim([0.0, 0.5])
        ax.set_xlabel("$R$ [au]")
        ax.set_ylabel("$\pi$/2 - $\Theta$")
        #    ax.set_aspect(1)
        if not full:
            ax.set_ylim(0.0, ax.get_ylim()[1])

    def plot_temp(self, fig=None, contourf_kwargs=None, full=True):
        fig, ax = self._grab_axes(fig)
        R = self.Rgrid / self.AU
        THETA = 0.5*np.pi - self.Tgrid[::-1]
        TEMP = self.temp[::-1]
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


    def plot_tgrad(self, fig=None, contourf_kwargs=None, full=True):
        fig, ax = self._grab_axes(fig)
        R = self.Rgrid / self.AU
        THETA = 0.5*np.pi - self.Tgrid[::-1]
        TGRAD = np.log10(self.tgrad[::-1])
        toplot = np.vstack([TGRAD[::-1], TGRAD])
        yaxis = np.concatenate([-THETA[::-1], THETA])

        contourf_kwargs = {} if contourf_kwargs is None else contourf_kwargs
        levels = np.linspace(toplot.min(), toplot.max(), 50)
        levels = np.linspace(-16, -12, 50)
        levels = contourf_kwargs.pop("levels", levels)
        cmap = contourf_kwargs.pop("cmap", "plasma")
        im = ax.contourf(R, yaxis, toplot, levels=levels,
                         cmap=cmap, **contourf_kwargs)

        cax = make_axes_locatable(ax)
        cax = cax.append_axes("right", size="4.5%" if full else "3%",
                              pad="2.25%" if full else "1.5%")
        cb = plt.colorbar(im, cax=cax, ticks=np.arange(-16, -12, 50))
        cb.set_label(r"$T\,\,[{\rm K}]$", rotation=270, labelpad=15)

        self._gentrify_structure_ax(ax, full=full)
        return fig


    def plot_dens(self, fig=None, contourf_kwargs=None, full=True):
        fig, ax = self._grab_axes(fig)
        R = self.Rgrid / self.AU
        THETA = 0.5*np.pi - self.Tgrid[::-1]
        RHO = self.rhog[::-1]
        toplot = np.vstack([RHO[::-1], RHO])
        toplot = np.log10(toplot / (sc.m_p*1e3) / 2.37)
        yaxis = np.concatenate([-THETA[::-1], THETA])

        contourf_kwargs = {} if contourf_kwargs is None else contourf_kwargs
        levels = np.linspace(toplot.min(), toplot.max(), 50)
        levels = np.linspace(3, 12, 50)
        levels = contourf_kwargs.pop("levels", levels)
        cmap = contourf_kwargs.pop("cmap", "bone_r")
        im = ax.contourf(R, yaxis, toplot, levels=levels,
                         cmap=cmap, **contourf_kwargs)

        cax = make_axes_locatable(ax)
        cax = cax.append_axes("right", size="4.5%" if full else "3%",
                              pad="2.25%" if full else "1.5%")
        cb = plt.colorbar(im, cax=cax, ticks=np.arange(-30, 30, 2))
        cb.set_label(r"$\log_{10}(n_{\rm H_2}\,\,[{\rm cm^{-3}}])$",
                     rotation=270, labelpad=15)
        self._gentrify_structure_ax(ax, full=full)
        return fig


    def plot_dustdens(self, fig=None, contourf_kwargs=None, full=True):
        fig, ax = self._grab_axes(fig)
        R = self.Rgrid / self.AU
        THETA = 0.5*np.pi - self.Tgrid[::-1]
        RHO = self.rhod[::-1]
        toplot = np.vstack([RHO[::-1], RHO])
        toplot = np.log10(toplot)
        yaxis = np.concatenate([-THETA[::-1], THETA])

        contourf_kwargs = {} if contourf_kwargs is None else contourf_kwargs
        levels = np.linspace(toplot.min(), toplot.max(), 50)
        levels = np.linspace(-20, -7, 50)
        levels = contourf_kwargs.pop("levels", levels)
        cmap = contourf_kwargs.pop("cmap", "bone_r")
        im = ax.contourf(R, yaxis, toplot, levels=levels,
                         cmap=cmap, **contourf_kwargs)

        cax = make_axes_locatable(ax)
        cax = cax.append_axes("right", size="4.5%" if full else "3%",
                              pad="2.25%" if full else "1.5%")
        cb = plt.colorbar(im, cax=cax, ticks=np.arange(-50, -5, 2))
        cb.set_label(r"$\log_{10}(\rho_{\rm dust}\,\,[{\rm g cm^{-3}}])$",
                     rotation=270, labelpad=15)
        self._gentrify_structure_ax(ax, full=full)
        return fig


    def plot_nmol(self, fig=None, contourf_kwargs=None, full=True):
        fig, ax = self._grab_axes(fig)
        R = self.Rgrid / self.AU
        THETA = 0.5*np.pi - self.Tgrid[::-1]
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


    def plot_rotation(self, fig=None, contourf_kwargs=None, full=True):
        fig, ax = self._grab_axes(fig)
        R = self.Rgrid / self.AU
        THETA = 0.5*np.pi - self.Tgrid[::-1]
        VEL = self.vel[::-1]
        toplot = np.vstack([VEL[::-1], VEL]) / 1e5
        yaxis = np.concatenate([-THETA[::-1], THETA])

        contourf_kwargs = {} if contourf_kwargs is None else contourf_kwargs
        levels = np.linspace(0., 6.5, 50)
        levels = contourf_kwargs.pop("levels", levels)
        cmap = contourf_kwargs.pop("cmap", "viridis")
        im = ax.contourf(R, yaxis, toplot, levels=levels, cmap=cmap,
                         **contourf_kwargs)

        cax = make_axes_locatable(ax)
        cax = cax.append_axes("right", size="4.5%" if full else "3%",
                              pad="2.25%" if full else "1.5%")
        cb = plt.colorbar(im, cax=cax)
        cb.set_label(r"$v_{\phi} \,\, [{\rm km/s}]$", rotation=270,
                     labelpad=15)

        self._gentrify_structure_ax(ax, full=full)
        return fig
