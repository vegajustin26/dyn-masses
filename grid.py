"""
Originally based on RADMC3D python wrappers; 
Modified by Jane Huang (CfA) and Sean Andrews (CfA);
Modified by Rich Teague (CfA).

TODO:
    - Check on the unit naming convention.
"""

import os
import yaml
import numpy as np
import scipy.constants as sc

class grid:

    """
    Generate a spatial and spectral grid structure used by RADMC3D.
    
    Args:
        modelname (str): Name of the model with parameters in modelname.yaml.
	writegrid (bool): Write out the grid into RADMC3D formats.
    """

    def __init__(self, modelname, writegrid=False):
        # bookkeeping
        self.modelname = modelname
        self.model_dir = modelname + '/'
        if not os.path.exists(self.model_dir):
            print("WARNING: Model directory not found.  Grid *NOT* saved.")

        # load grid parameters from model file (modelname.yaml)
        conf = open(modelname+'.yaml')
        self.grid_params = yaml.load(conf, Loader=yaml.FullLoader)["grid"]
        conf.close()

        # populate the grids
        self._read_spatial_grid(self.grid_params["spatial"])
        self.grid_params["wavelength"] = self.grid_params.pop("wavelength", {})
        self._read_wavelength_grid(self.grid_params["wavelength"])
        print('Grids successfully populated.')

        # write the grids to file (if requested)
        if writegrid:
            if not os.path.exists(self.model_dir):
                print("Generating new model directory: " + self.model_dir)
                os.mkdir(self.model_dir)
            self.write_wavelength_grid()
            self.write_spatial_grid()
            self.write_config_files()


    def _read_spatial_grid(self, params):
        """ Populate the spatial grid in spherical polar coordinates """
        # number of cells
        self.nr = params["nr"]
        self.nt = params["nt"]
        self.np = params["np"]
        self.ncells = self.nr * self.nt * self.np
        
        # radial grid in [cm]
        self.r_in = params["r_min"] * sc.au * 1e2
        self.r_out = params["r_max"] * sc.au * 1e2
        self.r_walls = np.logspace(np.log10(self.r_in), np.log10(self.r_out), 
                                   self.nr+1)
        self.r_centers = np.average([self.r_walls[:-1], self.r_walls[1:]], 
                                    axis=0)
        assert self.r_centers.size == self.nr

        # theta (altitude angle from pole toward equator) grid in [rad]
        self.t_offset = params.get("t_offset", 0.1)
        self.t_min = params.get("t_min", 0.0) + self.t_offset
        self.t_max = params.get("t_max", 0.5 * np.pi) + self.t_offset
        self.t_walls = np.logspace(np.log10(self.t_min), np.log10(self.t_max), 
                                   self.nt+1)
        self.t_walls = 0.5 * np.pi + self.t_offset - self.t_walls[::-1]
        self.t_min = self.t_walls.min()
        self.t_max = self.t_walls.max()
        self.t_centers = np.average([self.t_walls[:-1], self.t_walls[1:]], 
                                    axis=0)
        assert self.t_centers.size == self.nt

        # phi (azimuth angle) grid in [rad]
        self.p_min = params.get("p_min", 0.0)
        self.p_max = params.get("p_max", 0.0)
        self.p_walls = np.linspace(self.p_min, self.p_max, self.np + 1)
        self.p_centers = np.average([self.p_walls[:-1], self.p_walls[1:]], 
                                    axis=0)
        assert self.p_centers.size == self.np

    
    def _read_wavelength_grid(self, params):
        """ Populate the wavelength grid in [microns] """
        self.nw = params.get("nw", 100)
        self.logw_min = params.get("logw_min", -1.0)
        self.logw_max = params.get("logw_max", 4.0)
        self.w_centers = np.logspace(self.logw_min, self.logw_max, self.nw)



    # Output functions.

    def write_wavelength_grid(self, fileout='wavelength_micron.inp'):
        """ Write the wavelength grid to file """
        header = '100\n'
        np.savetxt(self.model_dir + fileout, self.w_centers, header=header, 
                   comments='')

        
    def write_spatial_grid(self, fileout='amr_grid.inp'):
        """ Write the spatial grid to file """
        header = '1\n0\n100\n0\n1 1 0\n'
        header += '{:d} {:d} {:d}'.format(self.nr, self.nt, self.np)
        tosave = np.concatenate([self.r_walls, self.t_walls, self.p_walls])
        np.savetxt(self.model_dir + fileout, tosave, header=header, 
                   comments='')
        

    def write_config_files(self, fileout='radmc3d.inp'):
        """ Write the RADMC3D configuration files """
        
        # Read in the .yaml file
        conf = open(self.modelname + '.yaml')
        conf_params = yaml.load(conf, Loader=yaml.FullLoader)["setup"]
        conf.close()
        
        # open file
        f = open(self.model_dir + fileout, 'w')

        # configuration contents
        f.write('incl_dust = %d\n' % conf_params["incl_dust"])
        f.write('incl_lines = %d\n' % conf_params["incl_lines"])
        f.write('incl_freefree = %d\n' % conf_params["incl_freefree"])
        if conf_params["scattering"] == 'None':
            f.write('scattering_mode_max= %d \n' % 0)
        elif conf_params["scattering"] == 'Isotropic':
            f.write('scattering_mode_max= %d\n' % 1)
            f.write('nphot_scat=2000000\n')
        elif conf_params["scattering"] == 'HG':
            f.write('scattering_mode_max = %d \n' % 2)
            f.write('nphot_scat=10000000\n')
        elif conf_params["scattering"] == 'Mueller':
            f.write('scattering_mode_max = %d \n' % 3)
            f.write('nphot_scat=100000000\n')
        if conf_params["binary"]:
            f.write('writeimage_unformatted = 1\n')
            f.write('rto_single = 1\n')
            f.write('rto_style = 3\n')
        else:
            f.write('rto_style = 1\n')
        if conf_params["camera_tracemode"]=='image':
            f.write('camera_tracemode = 1\n')
        elif conf_params["camera_tracemode"]=='tau':
            f.write('camera_tracemode = -2\n')
        if conf_params["lines_mode"]=='LTE':
            f.write('lines_mode = 1\n')
        f.close()

        ### LINE DATA CONFIG FILE
        if (conf_params["incl_lines"] == 1):
            f = open(self.model_dir + 'lines.inp', 'w')
            f.write('2\n1\n')
            f.write('%s    leiden    0    0    0' % conf_params["molecule"]) 
            f.close()

            # copy appropriate molecular data file
            os.system('cp moldata/'+conf_params["molecule"]+'.dat ' + \
                      self.model_dir + \
                      'molecule_'+conf_params["molecule"]+'.inp')
