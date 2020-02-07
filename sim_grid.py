import os
import sys
import yaml
import numpy as np
import scipy.constants as sc

class sim_grid:

    # constants
    msun = 1.989e33
    AU = sc.au * 1e2
    mu = 2.37
    m_p = sc.m_p * 1e3
    kB = sc.k * 1e7
    G = sc.G * 1e3

    def __init__(self, modelname, writegrid=True):

        # load grid parameters 
        self.modelname = modelname
        conf = open(modelname+'.yaml')
        config = yaml.load(conf, Loader=yaml.FullLoader)
        self.gridpars = config["grid"]
        self.diskpars = config["disk_params"]
        self.setup = config["setup"]
        conf.close()

        # populate the spatial grids
        """ Manual (radial) refinement if there are substructures """
        if self.diskpars["substructure"]["type"] == 'None':
            self._read_spatial_grid(self.gridpars["spatial"])
        else:
            args = {**self.gridpars["spatial"], 
                    **self.diskpars["substructure"]["arguments"]}
            self._read_spatial_grid(args, refine=True)

        # populate the wavelength grid
        if "wavelength" not in self.gridpars: 
            self.gridpars["wavelength"] = self.gridpars.pop("wavelength", {})
        self._read_wavelength_grid(self.gridpars["wavelength"])


        # write out the grids into RADMC3D formats
        if writegrid:
            if not os.path.exists(self.modelname): os.mkdir(self.modelname)
            self.write_wavelength_grid()
            self.write_spatial_grid()
            self.write_config_files()



    def _read_spatial_grid(self, args, refine=False):
        """ Populate the spatial grid in spherical polar coordinates """
        # number of cells
        self.nr, self.nt, self.np = args["nr"], args["nt"], args.pop("np", 1)

        # radial grid in [cm]
        self.r_in  = args["r_min"] * self.AU
        self.r_out = args["r_max"] * self.AU
        self.r_walls = np.logspace(np.log10(self.r_in), np.log10(self.r_out),
                                   self.nr+1)
        self.r_centers = np.average([self.r_walls[:-1], self.r_walls[1:]],
                                    axis=0)

        """ Radial refinement if substructures implemented """
        if refine:
            print('possible refinement')
            # identify substructure features
            locs, wids = args["locs"], args["wids"]
            nfeat = len(locs)

            # define a refinement boundary characteristic
            if self.diskpars["substructure"]["type"] == 'gaps_gauss':
                dr, frac_r = 4.0, 0.004		# sigma
            elif self.diskpars["substructure"]["type"] == 'gaps_sqr':
                dr, frac_r = 1.2, 0.0012

            # refine the radial grid around the substructures
            for ix in range(nfeat):
                rss, wss = locs[ix] * self.AU, wids[ix] * self.AU

                # condition to be in refinement region:
                reg = ((self.r_walls > (rss - dr * wss)) & 
                       (self.r_walls < (rss + dr * wss)))
                nreg = len(self.r_walls[reg])

                # desired number of cells across feature
                nrefine = 2 * dr * wss / rss / frac_r
                
                # swap in refined cells with linear sampling across feature
                if (nreg < nrefine):
                    print('refining...')
                    r_exc = self.r_walls[~reg]
                    r_add = rss + np.linspace(-dr*wss, dr*wss, nrefine)
                    self.r_walls = np.sort(np.concatenate((r_exc, r_add)))

            # re-compute cell centers and number
            self.r_centers = np.average([self.r_walls[:-1], self.r_walls[1:]],
                                        axis=0)
            self.nr = len(self.r_centers)
            print(self.nr)

        assert self.r_centers.size == self.nr


        # number of cells
        self.ncells = self.nr * self.nt * self.np

        # theta (altitude angle from pole toward equator) grid in [rad]
        self.t_offset = args.get("t_offset", 0.1)
        self.t_min = args.get("t_min", 0.0) + self.t_offset
        self.t_max = args.get("t_max", 0.5 * np.pi) + self.t_offset
        self.t_walls = np.logspace(np.log10(self.t_min), np.log10(self.t_max),
                                   self.nt+1)
        self.t_walls = 0.5 * np.pi + self.t_offset - self.t_walls[::-1]
        self.t_min = self.t_walls.min()
        self.t_max = self.t_walls.max()
        self.t_centers = np.average([self.t_walls[:-1], self.t_walls[1:]],
                                    axis=0)
        assert self.t_centers.size == self.nt

        # phi (azimuth angle) grid in [rad]
        self.p_min = args.get("p_min", 0.0)
        self.p_max = args.get("p_max", 0.0)
        self.p_walls = np.linspace(self.p_min, self.p_max, self.np + 1)
        self.p_centers = np.average([self.p_walls[:-1], self.p_walls[1:]],
                                    axis=0)
        assert self.p_centers.size == self.np


    def _read_wavelength_grid(self, params):
        self.nw = params.get("nw", 100)
        self.logw_min = params.get("logw_min", -1.0)
        self.logw_max = params.get("logw_max", 4.0)
        self.w_centers = np.logspace(self.logw_min, self.logw_max, self.nw)


    def write_wavelength_grid(self, fileout='wavelength_micron.inp'):
        np.savetxt(self.modelname + '/' + fileout, self.w_centers, 
                   header=str(self.nw) + '\n', comments='')


    def write_spatial_grid(self, fileout='amr_grid.inp'):
        """ Write the spatial grid to file """
        header = '1\n0\n100\n0\n1 1 0\n'
        header += '{:d} {:d} {:d}'.format(self.nr, self.nt, self.np)
        tosave = np.concatenate([self.r_walls, self.t_walls, self.p_walls])
        np.savetxt(self.modelname + '/' + fileout, tosave, header=header,
                   comments='')


    def write_config_files(self, fileout='radmc3d.inp'):

        """ Write the RADMC3D configuration files """
        # open file
        f = open(self.modelname + '/' + fileout, 'w')

        # spectral line, continuum, or both
        f.write('incl_dust = %d\n' % self.setup["incl_dust"])
        f.write('incl_lines = %d\n' % self.setup["incl_lines"])
        f.write('incl_freefree = %d\n' % self.setup.pop("incl_freefree", 0))

        # treatment of (continuum) scattering
        if self.setup["scattering"] == 'None':
            f.write('scattering_mode_max= %d \n' % 0)
        elif self.setup["scattering"] == 'Isotropic':
            f.write('scattering_mode_max= %d\n' % 1)
            f.write('nphot_scat=2000000\n')
        elif self.setup["scattering"] == 'HG':
            f.write('scattering_mode_max = %d \n' % 2)
            f.write('nphot_scat=10000000\n')
        elif self.setup["scattering"] == 'Mueller':
            f.write('scattering_mode_max = %d \n' % 3)
            f.write('nphot_scat=100000000\n')

        # select ascii or binary storage
        if "binary" not in self.setup: self.setup["binary"] = False
        if self.setup["binary"]:
            f.write('writeimage_unformatted = 1\n')
            f.write('rto_single = 1\n')
            f.write('rto_style = 3\n')
        else:
            f.write('rto_style = 1\n')

        # raytrace intensities or optical depths
        if "camera_tracemode" not in self.setup: 
            self.setup["camera_tracemode"] = 'image'
        if self.setup["camera_tracemode"] == 'image':
            f.write('camera_tracemode = 1\n')
        elif self.setup["camera_tracemode"] == 'tau':
            f.write('camera_tracemode = -2\n')

        # LTE excitation calculations (hard-coded)
        f.write('lines_mode = 1\n')

        f.close()


        ### DUST CONFIG FILE
        if (self.setup["incl_dust"] == 1):
            f = open(self.modelname + '/' + 'dustopac.inp', 'w')
            f.write('2\n1\n')
            f.write('============================================================================\n')
            f.write('1\n0\n')
            f.write('%s\n' % self.setup["dustspec"])
            f.write('----------------------------------------------------------------------------')
            f.close()

            # copy appropriate opacity file
            os.system('cp opacs/dustkappa_'+self.setup["dustspec"]+'.inp ' + \
                      self.modelname + '/')


        ### LINE DATA CONFIG FILE
        if (self.setup["incl_lines"] == 1):
            f = open(self.modelname + '/lines.inp', 'w')
            f.write('2\n1\n')
            f.write('%s    leiden    0    0    0' % self.setup["molecule"])
            f.close()

            # copy appropriate molecular data file
            os.system('cp moldata/' + self.setup["molecule"]+'.dat ' + \
                      self.modelname + \
                      '/molecule_' + self.setup["molecule"]+'.inp')
