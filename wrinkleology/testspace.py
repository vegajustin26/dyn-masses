import numpy as np
from simple_disk import simple_disk

# generate a simple disk model class
disk = simple_disk(inc=40., PA=90., mstar=2.0, FOV=5.0, z0=0.4, psi=0.5, 
                   Npix=128)

# specify the velocity channels of interest
velax = np.arange(-600, 600, 100)

# generate channel maps
cube = disk.get_cube(velax, bmaj=0.25, rms=0.0)

print(np.shape(cube))
