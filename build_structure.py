import numpy as np
import constants as const

"""


"""

# shorthand constants
AU = const.au.cgs
Msun = const.M_sun.cgs
mu_gas = 2.37
m_H = const.m_p.cgs + const.m_e.cgs
G = const.G.cgs
kB = const.k_B.cgs
PI = np.pi


class Grid:
    def __init__(self, nr, nt, na, rin, rout):

    # parameters
    self.nr = nr
    self.nt = nt
    self.naz = na
    self.ncells = self.nr * self.nt * self.naz
    self.r_in = rin * AU
    self.r_out = rout * AU

    ### RADIAL GRID

    # define cell walls and centers
    """ not sure about the centers defined in this way... """
    self.r_walls = np.logspace(np.log10(self.r_in), np.log10(self.r_out), nr+1)
    self.r_centers = 0.5 * (self.r_walls[:-1] + self.r_walls[1:])

    
    ### THETA GRID (= zenith angle)
    
    # set a slight offset bound at pole
    po = 0.1

    # define cell walls and centers
    """ not sure about the centers defined in this way... """
    self.t_walls = 0.5*PI + po - \
                   np.logspace(np.log10(po), np.log10(0.5*PI+po), nt+1)[::-1]
    self.t_centers = 0.5 * (self.t_walls[:-1] + self.t_walls[1:])


    ### PHI GRID (= azimuth)

    # define cell walls and centers
    self.az_walls = np.array([0.0, 0.0])
    self.az_centers = np.array([0.0])


    ### OUTPUT

    # open file
    f = open('amr_grid.inp', 'w')

    # file header
    f.write('1\n')	# format code
    f.write('0\n')	# regular grid
    f.write('100\n')	# spherical coordinate system
    f.write('0\n')	# no grid info written to file (as recommended)
    f.write('1 1 1\n')	# 
    f.write('%d %d %d\n' % (self.nr, self.nt, self.naz))

    # write wall coordinates to file
    for r in self.r_walls: f.write('%.9e\n' % r)
    for t in self.t_walls: f.write('%.9e\n' % t)
    for az in self.az_walls: f.write('%.9e\n' % az)

    # close file
    f.close()



class DiskModel:
    def __init__(self, Mstar, Sig0_d, R0_d, pd1, pd2, 
                              Sig0_g, R0_g, pg1, pg2, 
                       T0_mid, q_mid, T0_atm, q_atm, delta):

    # stellar properties
    self.Mstar = Mstar * Msun

    # dust surface density parameters
    self.Sig0_d = np.float64(Sig0_d)
    self.R0_d = R0_d * AU
    self.pd1 = np.float64(pd1)
    self.pd2 = np.float64(pd2)

    # gas surface density parameters
    self.Sig0_g = np.float64(Sig0_g)
    self.R0_g = R0_g * AU
    self.pg1 = np.float64(pg1)
    self.pg2 = np.float64(pg2)

    # thermal structure parameters
    self.T0_mid = np.float64(T0_mid)
    self.q_mid = np.float64(q_mid)
    self.T0_atm = np.float64(T0_atm)
    self.q_atm = np.float64(q_atm)
    self.delta = np.float64(delta)

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
        self.z_atm = self.Hp(r) * 4	# fix "atmosphere" to 4 * Hp
        Trz =  self.T_atm(r) + (self.T_mid(r) - self.T_atm(r)) * \
               np.cos(PI * z / (2 * self.z_atm))**(2.*self.delta)
        if (z > self.z_atm): Trz = self.T_atm(r)
        return Trz

    # 2-D GAS DENSITY STRUCTURE
    def rho_g(self, r, z):
    
        # first 
    


