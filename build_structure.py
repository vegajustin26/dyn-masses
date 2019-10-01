import numpy as np
import yaml
from scipy import integrate 
from scipy.interpolate import interp1d

"""


"""

# shorthand constants
AU = 1.49597871e13
Msun = 1.98847542e33
mu_gas = 2.37
m_H = 1.67353284e-24
G = 6.67408e-8
kB = 1.38064852e-16
PI = np.pi


class Grid:
    def __init__(self, nr, nt, naz, rin, rout):

        # parameters
        self.nr = nr
        self.ntheta = nt
        self.nphi = naz
        self.ncells = self.nr * self.ntheta * self.nphi
        self.r_in = rin * AU
        self.r_out = rout * AU

        ### RADIAL GRID

        # define cell walls and centers
        """ not sure about the centers defined in this way... """
        self.r_walls = np.logspace(np.log10(self.r_in), np.log10(self.r_out), 
                                   nr+1)
        self.r_centers = 0.5 * (self.r_walls[:-1] + self.r_walls[1:])

    
        ### THETA GRID (= zenith angle)
    
        # set a slight offset bound at pole
        po = 0.1

        # define cell walls and centers
        """ not sure about the centers defined in this way... """
        self.theta_walls = 0.5*PI + po - np.logspace(np.log10(po), 
                           np.log10(0.5*PI+po), nt+1)[::-1]
        self.theta_centers = 0.5*(self.theta_walls[:-1]+self.theta_walls[1:])


        ### PHI GRID (= azimuth)

        # define cell walls and centers
        self.phi_walls = np.array([0.0, 0.0])
        self.phi_centers = np.array([0.0])


        ### OUTPUT

        # open file
        f = open('amr_grid.inp', 'w')

        # file header
        f.write('1\n')	# format code
        f.write('0\n')	# regular grid
        f.write('100\n')	# spherical coordinate system
        f.write('0\n')	# no grid info written to file (as recommended)
        f.write('1 1 1\n')	# 
        f.write('%d %d %d\n' % (self.nr, self.ntheta, self.nphi))

        # write wall coordinates to file
        for r in self.r_walls: f.write('%.9e\n' % r)
        for t in self.theta_walls: f.write('%.9e\n' % t)
        for az in self.phi_walls: f.write('%.9e\n' % az)

        # close file
        f.close()


class DiskModel:
    def __init__(self, configfile):

        # open configuration file
        conf = open(configfile)
        config = yaml.load(conf)
        conf.close()
        disk_params = config["disk_params"]
        host_params = config["host_params"]

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

        return rho_gas
 

    # 2-D MOLECULAR NUMBER DENSITY STRUCTURE
    def nCO(self, r, z):
    
        rho_gas = self.rho_g(r,z)
        f_CO = 6 * 10.**(-5)

        return rho_gas * 0.8 * f_CO / (mu_gas * m_H)


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
        dustdens_inp = open('dust_density.inp', 'w')
        dustdens_inp.write('1\n%d\n1\n' % Grid.ncells)

        dusttemp_inp = open('dust_temperature.dat', 'w')
        dusttemp_inp.write('1\n%d\n1\n' % Grid.ncells)

        gasdens_inp = open('gas_density.inp', 'w')
        gasdens_inp.write('1\n%d\n' % Grid.ncells)

        gastemp_inp = open('gas_temperature.inp', 'w')
        gastemp_inp.write('1\n%d\n' % Grid.ncells)

        codens_inp = open('numberdens_co.inp', 'w')
        codens_inp.write('1\n%d\n' % Grid.ncells)

        vel_inp = open('gas_velocity.inp', 'w')
        vel_inp.write('1\n%d\n' % Grid.ncells)

        turb_inp = open('microturbulence.inp', 'w')
        turb_inp.write('1\n%d\n' % Grid.ncells)

        # populate files
        for phi in Grid.phi_centers:
            for theta in Grid.theta_centers:
                for r in Grid.r_centers:
                    r_cyl = r * np.sin(theta)
                    z = r * np.cos(theta)

                    dusttemp_inp.write('%.6e\n' % self.Temp(r_cyl, z))
                    gastemp_inp.write('%.6e\n' % self.Temp(r_cyl, z))
                    dustdens_inp.write('%.6e\n' % self.rho_d(r_cyl, z))
                    gasdens_inp.write('%.6e\n' % self.rho_g(r_cyl, z))
                    codens_inp.write('%.6e\n' % self.nCO(r_cyl, z))
                    vel_inp.write('0 0 %.6e\n' % self.velocity(r_cyl))
                    turb_inp.write('%.6e\n' % self.vturb(r_cyl, z))

        # close files
        gastemp_inp.close()
        gasdens_inp.close()
        dusttemp_inp.close()
        dustdens_inp.close()
        codens_inp.close()
        vel_inp.close()
        turb_inp.close()
