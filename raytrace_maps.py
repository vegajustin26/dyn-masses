import numpy as np
import sys
import os
import yaml
from fitsconversion import convert_to_fits


class raytrace_maps:

    def __init__(self, modelname, writefits=True):

        # load information
        conf = open(modelname + ".yaml")
        config = yaml.load(conf, Loader=yaml.FullLoader)
        outpars = config["outputs"]
        setpars = config["setup"]
        gridpars = config["grid"]
        conf.close()

        # position angle convention (for now)
        posang = outpars["geometry"]["PA"] - 90.

        # specify pixel resolution, use 10% buffer over model grid size
        sizeau = 1.1 * 2 * gridpars["spatial"]["r_max"]
        npix = sizeau / outpars["geometry"]["dpc"] / outpars["spatial"]["ires"]
        npix = np.int(np.ceil(npix / 2) * 2)


        # velocity scale
        cc = 2.99792458e10
        nu0 = 230.538e9

        DV0 = 2 * 1e5 * outpars["velocity"]["widthkms"]
        dnu = 1e3 * outpars["velocity"]["dfreq"]	
        nchan = outpars["velocity"]["oversample"] * \
                np.int(np.ceil(DV0 * nu0 / cc / dnu))
        widthkms = (nchan - 1) * cc * dnu / nu0 / 1e5 / 2 / \
                   outpars["velocity"]["oversample"]


        # run raytracer
        os.chdir(modelname)
        os.system('radmc3d image ' + \
                  'incl %.2f ' % outpars["geometry"]["incl"] + \
                  'posang %.2f ' % posang + \
                  'npix %d ' % npix + \
                  'sizeau %d ' % sizeau + \
                  'iline %d ' % setpars["transition"] + \
                  'widthkms %.5f ' % widthkms + \
                  'linenlam %d ' % nchan + \
                  'setthreads 4')


        # make a FITS cube
        if writefits:
            outfile = modelname + '_' + setpars["molecule"] + '.fits'
            os.system('rm ' + outfile)
            convert_to_fits('image.out', outfile, outpars["geometry"]["dpc"], 
                            RA=outpars["spatial"]["RA"], 
                            DEC=outpars["spatial"]["DEC"], 
                            downsample=outpars["velocity"]["oversample"])