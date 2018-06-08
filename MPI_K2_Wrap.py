from mpi4py import MPI

from glob import glob

import time as sys_time
# Import all the functions
from K2TranPixCode import *


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


field = 'c01'
path = '/avatar/ryanr/Data/'+field+'/'
Files = np.asarray(glob(path+'*.gz'))

save = '/avatar/ryanr/Results/'

Files = np.asarray(glob(path+'*.gz'))
# Code to remove files from the list that have already been calculated
if comm.Get_rank() == 0:
    try:
        Log = open('/avatar/ryanr/Code/shell' + field + '.out')
        log = Log.read()
        lines  = log.split('\n')
        files = []
        for line in lines:
            if '/avatar/ryanr/Data/' in line:
                print_mpi(line)
                files.append(line)
        for i in range(len(files)):
            files[i] = files[i].split(' ')[1]
            beep = set.intersection(set(files[i]), set('['))
            if len(beep) > 0:
                files[i] = files[i].split('[')[0]
                       
            Files = np.delete(Files,np.where(files[i] == Files))
            
    except (FileNotFoundError):
        print('No file')

    print_mpi('Filesbloop ' + str(len(Files)))

Files = comm.Bcast(Files, root = 0)
print("On Task "+str(myPE)+" Files was recvd "+str(len(Files)))

if comm.Get_rank() == 0:
    print('Blamo')
    size = []
    for i in range(len(Files)):
        size.append(os.path.getsize(Files[i]))
    size = np.array(size)
    totalsize = np.nansum(size)
    interval_size = totalsize / nPE

    starts = np.zeros(nPE,dtype=int)
    for i in range(nPE):
        if i == 0:
            starts[i] = 0
        else:
            sumsize = 0
            j = 1
            while (sumsize <= interval_size) & (starts[i-1] + 1 < len(size)):
                sumsize = np.nansum(size[starts[i-1]:starts[i-1]+j])
                j += 1
            starts[i] = starts[i-1] + (j - 1)
    print('starts computed')
starts = comm.Bcast(starts, root = 0)

dims = int(len(Files)) # set to be length of your task
start = sys_time.time()

# domain decomposition
my_start = starts[myPE]#int(myPE * (dims / nPE));
my_end   = starts[myPE + 1] - 1 #int((myPE+1) * (dims / nPE) - 1);
# last PE gets the rest
if (myPE == nPE-1): my_end = dims-1;
print_mpi("my_start = "+str(my_start)+", my_end = "+str(my_end))

# parallel loop
for n in range(my_start, my_end+1):
    mytimestart = sys_time.time()
    
    K2TranPix(Files[n],save)
    print_mpi(Files[n])
    
    mytimestop = sys_time.time()
    mytime = mytimestop-mytimestart


# MPI collective communication (all reduce)

stop = sys_time.time()
print_master('Time taken=%f' %(stop-start))