import numpy as np
import yaml
import os
from scipy import integrate 
from scipy.interpolate import interp1d


# shorthand constants
AU = 1.49597871e13
Msun = 1.98847542e33
mu_gas = 2.37
m_H = 1.67353284e-24
f_H = 0.706
f_H2 = 0.8
G = 6.67408e-8
kB = 1.38064852e-16
PI = np.pi


"""
Sets up the spatial grid, outputs into appropriate RADMC3D files. 
Configures AMR_GRID.inp, RADMC3D.inp, LINES.inp.
"""
class Grid:
    def __init__(self, modelname):

        # open configuration file
        conf = open(modelname+'.yaml')
        config = yaml.load(conf)
        conf.close()
        mdir = modelname+'/'
        grid_params = config["grid"]
        conf_params = config["setup"]

        # parameters
        self.nr = grid_params["nr"]
        self.ntheta = grid_params["ntheta"]
        self.nphi = grid_params["nphi"]
        self.ncells = self.nr * self.ntheta * self.nphi
        self.r_in = grid_params["r_in"] * AU
        self.r_out = grid_params["r_out"] * AU

        ### RADIAL GRID

        # define cell walls and centers
        """ not sure about the centers defined in this way... """
        self.r_walls = np.logspace(np.log10(self.r_in), np.log10(self.r_out), 
                                   self.nr+1)
        self.r_centers = 0.5 * (self.r_walls[:-1] + self.r_walls[1:])

    
        ### THETA GRID (= zenith angle)
    
        # set a slight offset bound at pole
        po = 0.1

        # define cell walls and centers
        """ not sure about the centers defined in this way... """
        self.theta_walls = 0.5*PI + po - np.logspace(np.log10(po), 
                           np.log10(0.5*PI+po), self.ntheta+1)[::-1]
        self.theta_centers = 0.5*(self.theta_walls[:-1]+self.theta_walls[1:])


        ### PHI GRID (= azimuth) : here fixed to enforce axisymmetry

        # define cell walls and centers
        self.phi_walls = np.array([0.0, 0.0])
        self.phi_centers = np.array([0.0])


        ### OUTPUT SPATIAL GRID

        # open file
        f = open(mdir+'amr_grid.inp', 'w')

        # file header
        f.write('1\n')		# format code
        f.write('0\n')		# regular grid
        f.write('100\n')	# spherical coordinate system
        f.write('0\n')		# no grid info written to file (as recommended)
        f.write('1 1 0\n')	# axisymmetric coding
        f.write('%d %d %d\n' % (self.nr, self.ntheta, self.nphi))

        # write wall coordinates to file
        for r in self.r_walls: f.write('%.9e\n' % r)
        for t in self.theta_walls: f.write('%.9e\n' % t)
        for phi in self.phi_walls: f.write('%.9e\n' % phi)

        # close file
        f.close()


        ### OUTPUT RT CONFIG FILE

        # open file
        f = open(mdir+'radmc3d.inp', 'w')

        # configuration contents
        f.write('incl_dust=%d\n' % conf_params["incl_dust"])
        f.write('incl_lines=%d\n' % conf_params["incl_lines"])
        f.write('incl_freefree=%d\n' % conf_params["incl_freefree"])
        if conf_params["scattering"] == 'None':
            f.write('scattering_mode_max=%d\n' % 0)
        elif conf_params["scattering"] == 'Isotropic':
            f.write('scattering_mode_max=%d\n' % 1)
            f.write('nphot_scat=2000000\n')
        elif conf_params["scattering"] == 'HG':
            f.write('scattering_mode_max=%d\n' % 2)
            f.write('nphot_scat=10000000\n')
        elif conf_params["scattering"] == 'Mueller':
            f.write('scattering_mode_max=%d\n' % 3)
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


        ### WAVELENGTH GRID
        f = open(mdir+'wavelength_micron.inp', 'w')
        f.write('100\n')
        for w in np.logspace(-1, 4, 100): f.write('%e\n' % w)
        f.close()


        ### LINE DATA CONFIG FILE
        if (conf_params["incl_lines"] == 1):
           f = open(mdir+'lines.inp', 'w')
           f.write('2\n1\n')
           f.write('%s    leiden    0    0    0' % conf_params["molecule"]) 
           f.close()

           # copy appropriate molecular data file
           os.system('cp moldata/'+conf_params["molecule"]+'.dat ' + mdir + \
                     'molecule_'+conf_params["molecule"]+'.inp')



class DiskModel:
    def __init__(self, modelname):

        # open configuration file
        conf = open(modelname+'.yaml')
        config = yaml.load(conf)
        conf.close()
        self.mdir = modelname+'/'
        disk_params = config["disk_params"]
        host_params = config["host_params"]
        self.do_dust = config["setup"]["incl_dust"]
        self.do_gas = config["setup"]["incl_lines"]
        self.molecule = config["setup"]["molecule"]

        # stellar properties
        self.Mstar = host_params["M_star"] * Msun

        # dust surface density parameters
        self.Sig0_d = disk_params["Sig0_d"]
        self.R0_d = disk_params["R0_d"] * AU
        self.pd1 = disk_params["pd1"]
        self.pd2 = disk_params["pd2"]

        # gas surface density parameters
        self.Sig0_g = disk_params["Sig0_g"]
        self.R0_g = disk_params["R0_g"] * AU
        self.pg1 = disk_params["pg1"]
        self.pg2 = disk_params["pg2"]
        self.sigma_pdr = 10.**(disk_params["sig_pdr"])
        self.depl_pdr = disk_params["depl_pdr"]
        self.T_frz = disk_params["T_frz"]
        self.depl_frz = disk_params["depl_frz"]
        self.fmol = 10.**(disk_params["fmol"])

        # thermal structure parameters
        self.T0_mid = disk_params["T0_mid"]
        self.q_mid = disk_params["q_mid"]
        self.T0_atm = disk_params["T0_atm"]
        self.q_atm = disk_params["q_atm"]
        self.delta = disk_params["delta"]

        # non-thermal broadening (as fraction of local sound speed)
        self.xi = disk_params["xi"]


    # DUST SURFACE DENSITY PROFILE
    def Sigma_d(self, r):
        sd = self.Sig0_d * (r / self.R0_d)**(-self.pd1) * \
             np.exp(-(r / self.R0_d)**self.pd2)    
        return sd


    # GAS SURFACE DENSITY PROFILE
    def Sigma_g(self, r):
        sg = self.Sig0_g * (r / self.R0_g)**(-self.pg1) * \
             np.exp(-(r / self.R0_g)**self.pg2)
        return sg


    # MIDPLANE TEMPERATURE PROFILE
    def T_mid(self, r):
        return self.T0_mid * (r / (10.*AU))**(-self.q_mid)


    # ATMOSPHERE TEMPERATURE PROFILE (saturates at z_atm)
    def T_atm(self, r):
        return self.T0_atm * (r / (10.*AU))**(-self.q_atm)


    # PRESSURE SCALE HEIGHTS
    def Hp(self, r):
        Omega = np.sqrt(G * self.Mstar / r**3)
        c_s = np.sqrt(kB * self.T_mid(r) / (mu_gas * m_H))
        return c_s / Omega


    # 2-D TEMPERATURE STRUCTURE
    def Temp(self, r, z):
        self.z_atm = self.Hp(r) * 4	    # fix "atmosphere" to 4 * Hp
        Trz =  self.T_atm(r) + (self.T_mid(r) - self.T_atm(r)) * \
               np.cos(PI * z / (2 * self.z_atm))**(2.*self.delta)
        if (z > self.z_atm): Trz = self.T_atm(r)
        return Trz


    # VERTICAL TEMPERATURE GRADIENT (dlnT / dz)
    def logTgrad(self, r, z):
        dT = -2 * self.delta * (self.T_mid(r) - self.T_atm(r)) * \
             (np.cos(PI * z / (2 * self.z_atm)))**(2*self.delta-1) * \
             np.sin(PI * z / (2 * self.z_atm)) * PI / (2 * self.z_atm) / \
             self.Temp(r,z)
        if (z > self.z_atm): dT = 0
        return dT


    # 2-D DUST DENSITY STRUCTURE
    def rho_d(self, r, z):
        z_dust = self.Hp(r) * 0.2	# fix dust scale height to lower
        dnorm = self.Sigma_d(r) / (np.sqrt(2 * PI) * z_dust)
        return dnorm * np.exp(-0.5 * (z / z_dust)**2)


    # 2-D GAS DENSITY STRUCTURE
    def rho_g(self, r, z):
    
        # set an upper atmosphere boundary
        z_max = 10 * self.z_atm
        PDR = False

        # grid of z values for integration
        zvals = np.logspace(np.log10(0.1), np.log10(z_max+0.1), 1024) - 0.1

        # load temperature gradient
        dlnTdz = self.logTgrad(r, z)
 
        # density gradient
        gz = G * self.Mstar * zvals / (r**2 + zvals**2)**1.5
        dlnpdz = -mu_gas * m_H * gz / (kB * self.Temp(r,z)) - dlnTdz

        # numerical integration
        lnp = integrate.cumtrapz(dlnpdz, zvals, initial=0)
        dens0 = np.exp(lnp)

        # normalized densities
        dens = 0.5 * self.Sigma_g(r) * dens0 / integrate.trapz(dens0, zvals)
        
        # interpolator for moving back onto the spatial grid
        f = interp1d(zvals, np.squeeze(dens), bounds_error=False, 
                     fill_value=(np.max(dens), 0))

        # properly normalized gas densities
        rho_gas = np.float(f(z))

        ## boolean indicator if this height is in the molecule's PDR
        # find index of nearest zvals cell
        index = np.argmin(np.abs(zvals-z))	
        # integrate the vertical density profile down to that height
        sig_index = integrate.trapz(dens[index:], zvals[index:])
        # criterion for photodissociation
        if (sig_index < (self.sigma_pdr * mu_gas * m_H * f_H)): 
            PDR = True

        return rho_gas, PDR
 

    # 2-D MOLECULAR NUMBER DENSITY STRUCTURE
    def nmol(self, r, z):
    
        # read in gas volume densities
        rho_gas, PDR = self.rho_g(r,z)

        # abundance variations
#        if (self.Temp(r,z) < self.T_frz): 
#            Xmol = self.depl_frz * self.fmol
#        elif PDR: 
#            Xmol = self.depl_pdr * self.fmol
#        else: 
        Xmol = self.fmol

        return rho_gas * f_H2 * Xmol / (mu_gas * m_H)


    # GAS VELOCITY STRUCTURE
    def velocity(self, r):
    
        vkep = np.sqrt(G * self.Mstar / r)

        return vkep


    # MICROTURBULENCE
    def vturb(self, r, z):

        c_s = np.sqrt(kB * self.Temp(r, z) / (mu_gas * m_H))
        dv = self.xi * c_s   

        return dv


    # WRITE OUT RADMC FILES
    def write_Model(self, Grid):
        
        # file headers
        if (self.do_dust == 1):
            dustdens_inp = open(self.mdir+'dust_density.inp', 'w')
            dustdens_inp.write('1\n%d\n1\n' % Grid.ncells)

            dusttemp_inp = open(self.mdir+'dust_temperature.dat', 'w')
            dusttemp_inp.write('1\n%d\n1\n' % Grid.ncells)

        if (self.do_gas == 1):
            gasdens_inp = open(self.mdir+'gas_density.inp', 'w')
            gasdens_inp.write('1\n%d\n' % Grid.ncells)

            gastemp_inp = open(self.mdir+'gas_temperature.inp', 'w')
            gastemp_inp.write('1\n%d\n' % Grid.ncells)

            nmol_inp = open(self.mdir+'numberdens_'+self.molecule+'.inp', 'w')
            nmol_inp.write('1\n%d\n' % Grid.ncells)

            vel_inp = open(self.mdir+'gas_velocity.inp', 'w')
            vel_inp.write('1\n%d\n' % Grid.ncells)

            turb_inp = open(self.mdir+'microturbulence.inp', 'w')
            turb_inp.write('1\n%d\n' % Grid.ncells)

        # populate files
        for phi in Grid.phi_centers:
            for theta in Grid.theta_centers:
                for r in Grid.r_centers:
                    r_cyl = r * np.sin(theta)
                    z = r * np.cos(theta)

                    if (self.do_dust == 1):
                        dusttemp_inp.write('%.6e\n' % self.Temp(r_cyl, z))
                        dustdens_inp.write('%.6e\n' % self.rho_d(r_cyl, z))

                    if (self.do_gas == 1):
                        gastemp_inp.write('%.6e\n' % self.Temp(r_cyl, z))
                        gasdens, dum = self.rho_g(r_cyl, z)
                        gasdens_inp.write('%.6e\n' % gasdens)
                        nmol_inp.write('%.6e\n' % self.nmol(r_cyl, z))
                        vel_inp.write('0 0 %.6e\n' % self.velocity(r_cyl))
                        turb_inp.write('%.6e\n' % self.vturb(r_cyl, z))

        # close files
        if (self.do_dust == 1):
            dusttemp_inp.close()
            dustdens_inp.close()
        if (self.do_gas == 1):
            gastemp_inp.close()
            gasdens_inp.close()
            nmol_inp.close()
            vel_inp.close()
            turb_inp.close()
