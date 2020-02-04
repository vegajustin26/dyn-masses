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


    def __init__(self, mname, struct=None):

        # if no structure is passed, extract it from files
        if struct is None:
            print('using files')
            _ = np.loadtxt(mname+'/amr_grid.inp', skiprows=5, max_rows=1)
            nr, nt = np.int(_[0]), np.int(_[1])
            Rw = np.loadtxt(mname+'/amr_grid.inp', skiprows=6, max_rows=nr+1)
            Tw = np.loadtxt(mname+'/amr_grid.inp', skiprows=nr+7, max_rows=nt+1)
            self.Rgrid = 0.5*(Rw[:-1] + Rw[1:])
            self.Tgrid = 0.5*(Tw[:-1] + Tw[1:])

            temp_in = np.loadtxt(mname+'/gas_temperature.inp', skiprows=2)
            self.temp = np.reshape(temp_in, (nt, nr))

            _ = self.plot_temp(full=False)
            _.savefig(mname+'/temp.png')

        # otherwise, just use the passed structure
        else:
            print('using struct')
            self.Rgrid = struct.rvals
            self.Tgrid = struct.tvals
            self.temp = struct.temp
            self.rhog = struct.rhogas
            self.nmol = struct.nmol
            self.vel = struct.vel

            if not os.path.exists(mname): os.mkdir(mname)
            _ = self.plot_temp(full=False)
            _.savefig(mname+'/temp.png')
            _ = self.plot_dens(full=False)
            _.savefig(mname+'/dens.png')
            _ = self.plot_nmol(full=False)
            _.savefig(mname+'/nmol.png')
            _ = self.plot_vel(full=False)
            _.savefig(mname+'/vel.png')

            


    @staticmethod
    def _grab_axes(fig):
        if fig is None:
            fig, ax = plt.subplots()
        else:
            ax = fig.axes[0]
        return fig, ax

    @staticmethod
    def _gentrify_structure_ax(ax, full=True):
        ax.set_xlim([2., 500])
        ax.set_xscale('log')
        ax.set_ylim([0.0, 0.7])
        ax.set_xlabel("$R$ [au]")
        ax.set_ylabel("$\pi$/2 - $\Theta$")
        #if not full: ax.set_ylim(0.0, ax.get_ylim()[1])

    def plot_temp(self, fig=None, contourf_kwargs=None, full=True):
        fig, ax = self._grab_axes(fig)
        xx = self.Rgrid / self.AU
        yy = 0.5*np.pi - self.Tgrid[::-1]
        zz = self.temp[::-1]
        toplot = np.vstack([zz[::-1], zz])
        yaxis = np.concatenate([-yy[::-1], yy])

        contourf_kwargs = {} if contourf_kwargs is None else contourf_kwargs
        levels = np.linspace(toplot.min(), toplot.max(), 50)
        levels = np.linspace(3, 300, 50)
        levels = contourf_kwargs.pop("levels", levels)
        cmap = contourf_kwargs.pop("cmap", "plasma")
        im = ax.contourf(xx, yaxis, toplot, levels=levels,
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
        xx = self.Rgrid / self.AU
        yy = 0.5*np.pi - self.Tgrid[::-1]
        zz = self.rhog[::-1]
        toplot = np.vstack([zz[::-1], zz])
        toplot = np.log10(toplot / self.m_p / self.mu)
        yaxis = np.concatenate([-yy[::-1], yy])

        contourf_kwargs = {} if contourf_kwargs is None else contourf_kwargs
        levels = np.linspace(toplot.min(), toplot.max(), 50)
        levels = np.linspace(0, 14, 50)
        levels = contourf_kwargs.pop("levels", levels)
        cmap = contourf_kwargs.pop("cmap", "plasma")
        im = ax.contourf(xx, yaxis, toplot, levels=levels,
                         cmap=cmap, **contourf_kwargs)

        cax = make_axes_locatable(ax)
        cax = cax.append_axes("right", size="4.5%" if full else "3%",
                              pad="2.25%" if full else "1.5%")
        cb = plt.colorbar(im, cax=cax, ticks=np.arange(-30, 30, 2))
        cb.set_label(r"$\log_{10}(n_{\rm H_2}\,\,[{\rm cm^{-3}}])$",
                     rotation=270, labelpad=15)

        self._gentrify_structure_ax(ax, full=full)
        return fig

    def plot_nmol(self, fig=None, contourf_kwargs=None, full=True):
        fig, ax = self._grab_axes(fig)
        xx = self.Rgrid / self.AU
        yy = 0.5*np.pi - self.Tgrid[::-1]
        zz = self.nmol[::-1]
        toplot = np.vstack([zz[::-1], zz])
        toplot = np.log10(toplot)
        yaxis = np.concatenate([-yy[::-1], yy])

        contourf_kwargs = {} if contourf_kwargs is None else contourf_kwargs
        levels = np.linspace(toplot.min(), toplot.max(), 50)
        levels = np.linspace(-12, 12, 50)
        levels = contourf_kwargs.pop("levels", levels)
        cmap = contourf_kwargs.pop("cmap", "afmhot_r")
        im = ax.contourf(xx, yaxis, toplot, levels=levels,
                         cmap=cmap, **contourf_kwargs)

        cax = make_axes_locatable(ax)
        cax = cax.append_axes("right", size="4.5%" if full else "3%",
                              pad="2.25%" if full else "1.5%")
        cb = plt.colorbar(im, cax=cax, ticks=np.arange(-30, 30, 2))
        cb.set_label(r"$\log_{10}(n_{\rm CO}\,\,[{\rm cm^{-3}}])$",
                     rotation=270, labelpad=15)

        self._gentrify_structure_ax(ax, full=full)
        return fig

    def plot_vel(self, fig=None, contourf_kwargs=None, full=True):
        fig, ax = self._grab_axes(fig)
        xx = self.Rgrid / self.AU
        yy = 0.5*np.pi - self.Tgrid[::-1]
        zz = self.vel[::-1]
        toplot = np.vstack([zz[::-1], zz])
        toplot = np.log10(toplot / 1e5)
        yaxis = np.concatenate([-yy[::-1], yy])

        contourf_kwargs = {} if contourf_kwargs is None else contourf_kwargs
        levels = np.linspace(toplot.min(), toplot.max(), 50)
        levels = np.linspace(0, 1.5, 50)
        levels = contourf_kwargs.pop("levels", levels)
        cmap = contourf_kwargs.pop("cmap", "viridis")
        im = ax.contourf(xx, yaxis, toplot, levels=levels,
                         cmap=cmap, **contourf_kwargs)

        cax = make_axes_locatable(ax)
        cax = cax.append_axes("right", size="4.5%" if full else "3%",
                              pad="2.25%" if full else "1.5%")
        cb = plt.colorbar(im, cax=cax)
        cb.set_label(r"$\log_{10}(v_{\phi})\,\,[{\rm km/s}])$",
                     rotation=270, labelpad=15)

        self._gentrify_structure_ax(ax, full=full)
        return fig
