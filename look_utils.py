import numpy as np
import sys
import os


def read_Tgas(grid, fname=''):

    print('Reading the gas temperature structure from: %a' % fname)

    # open file
    temp_in = open(fname, 'r')
    
    # load data and check that the dimensionality is right
    dum = temp_in.readline()
    ncells = temp_in.readline()

    Tgas = np.zeros((grid.nr, grid.ntheta))
    for ir in range(grid.nr):
        for it in range(grid.ntheta):
            Tgas[ir, it] = np.float(temp_in.readline())
    temp_in.close()

    return(Tgas)


def read_nmol(grid, fname=''):

    print('Reading the volume density structure from: %a' % fname)

    # open file
    nmol_in = open(fname, 'r')                   
    
    # load data and check that the dimensionality is right
    dum = nmol_in.readline()
    ncells = nmol_in.readline()

    nmol = np.zeros((grid.nr, grid.ntheta))
    for ir in range(grid.nr):
        for it in range(grid.ntheta):
            nmol[ir, it] = np.float(nmol_in.readline())
    nmol_in.close()

    return(nmol)

