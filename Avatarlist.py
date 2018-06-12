# Code to make files that split list up into equal segments 

import os 
import numpy as np
from glob import glob



def Avatar_list(cores,campaign):
    path = '/export/maipenrai/skymap/brad/KEGS/K2/Data' + str(int(campaign)) + '/'
    Files = np.asarray(glob(path+'*.gz'))

    size = []
    for i in range(len(Files)):
        size.append(os.path.getsize(Files[i]))
    size = np.array(size)
    totalsize = np.nansum(size)
    interval_size = totalsize / cores

    starts = np.zeros(cores,dtype=int)
    for i in range(cores):
        if i == 0:
            starts[i] = 0
        else:
            sumsize = 0
            j = 1
            while (sumsize <= interval_size) & (starts[i-1] + 1 < len(size)):
                sumsize = np.nansum(size[starts[i-1]:starts[i-1]+j])
                j += 1
            starts[i] = starts[i-1] + (j - 1)
    np.save(path+'starts.npy', starts)