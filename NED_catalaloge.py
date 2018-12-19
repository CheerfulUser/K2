import numpy as np
import pandas as pd
import os

from astropy.io import fits
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astroquery.ned import Ned

def Save_space(Save):
    """
    Creates a pathm if it doesn't already exist.
    """
    try:
        if not os.path.exists(Save):
            os.makedirs(Save)
    except FileExistsError:
        pass

    return  

def NED_database(File, Save):
    coords = np.loadtxt(File,dtype='str')
    for camp in coords:
        c = SkyCoord(coords[0,1] + ' ' + coords[0,1], unit=(u.hourangle, u.deg))
        startra = c.ra-7*u.deg
        startdec = c.dec-7*u.deg
        step = 0.25
        steps = int(6/0.25) # make a grid of points quesried every 1/4 degree, lots of overlap
        for i in range(steps):
            for j in range(steps):
                ra = startra + i*step*u.deg
                dec = startdec + j*step*u.deg
                center = SkyCoord(str(ra.deg) + ' ' + str(dec.deg), unit=(u.deg, u.deg))
                result_table = Ned.query_region(center, radius = 30*u.arcmin, equinox='J2000')
                table = result_table.to_pandas()
                
                name = 'NED_database_' + camp[0] + '_' + str(ra) + '_' + str(dec) + '.csv'
                sav = Save + camp[0] + '/'
                Save_space(Save)
                table.to_csv(Save+name)
                print('Saved')
        print('Done', camp[0])
    return 

