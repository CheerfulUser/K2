from mpi4py import MPI


import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import matplotlib.animation as animation

from scipy.ndimage.filters import convolve

from astropy.io import fits
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS

from glob import glob
import os

import time as t
import warnings
warnings.filterwarnings("ignore",category =RuntimeWarning)
# Import all the functions
import K2TranPix

field = 'c06'
path = '/avatar/ryanr/Data/'
Files = np.asarray(glob(path+'*.gz'))

save = '/mimsy/ryanr/PhD/Kepler/Results/'+field+'/'

Files = np.asarray(glob(path+'*.gz'))
dims = len(Files) # set to be length of your task
start = t.time()

def print_mpi(string):
    comm = MPI.COMM_WORLD
    print("["+str(comm.Get_rank())+"] "+string)

def print_master(string):
    comm = MPI.COMM_WORLD
    if comm.Get_rank() == 0:
        print("["+str(comm.Get_rank())+"] "+string)

# init MPI

comm = MPI.COMM_WORLD
nPE = comm.Get_size()
myPE = comm.Get_rank()
print_master("Total number of MPI ranks = "+str(nPE))
comm.Barrier()

# Remove previous test files

if comm.Get_rank() == 0:
    os.system("mv *cpu.txt oldouts/")
    os.system("mv total_prog.txt oldouts/")

# Progress saving function

def writemyprog(filename,out):
    np.savez(filename,out)

# domain decomposition

my_start = myPE * (dims / nPE);
my_end   = (myPE+1) * (dims / nPE) - 1;
# last PE gets the rest
if (myPE == nPE-1): my_end = dims-1;
print_mpi("my_start = "+str(my_start)+", my_end = "+str(my_end))

# parallel loop

filename = "my_prog_"+str(myPE).zfill(4)


for n in range(my_start, my_end+1):
    mytimestart = t.time()
    
    K2TranPix(Files(n))
    
    mytimestop = t.time()
    mytime = mytimestop-mytimestart
    print 'n=%g' %n, 'my_time=%f' %mytime


# MPI collective communication (all reduce)

stop = time.time()
print_master('Time taken=%f' %(stop-start))