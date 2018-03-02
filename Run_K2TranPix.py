%run  K2TranPix.py

from mpi4py import MPI
import time as t
from glob import glob

field = 'c05'

path = '/mimsy/ryanr/PhD/Kepler/Data/K2/'+field+'/'
save = '/mimsy/ryanr/PhD/Kepler/Resultss/'+field+'/K2tranPix/'

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


# domain decomposition

my_start = myPE * (dims / nPE);
my_end   = (myPE+1) * (dims / nPE) - 1;
# last PE gets the rest
if (myPE == nPE-1): my_end = dims-1;
print_mpi("my_start = "+str(my_start)+", my_end = "+str(my_end))

# parallel loop


for n in range(my_start, my_end+1):
    mytimestart = t.time()
        
    K2TranPix(Files[n],save)
    
    mytimestop = t.time()
    mytime = mytimestop-mytimestart
    print 'n=%g' %n, 'my_time=%f' %mytime


# MPI collective communication (all reduce)

stop = time.time()
print_master('Time taken=%f' %(stop-start))