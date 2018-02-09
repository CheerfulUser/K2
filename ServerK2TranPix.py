
# coding: utf-8

# In[1]:

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

from scipy.signal import convolve2d
from scipy.ndimage.filters import convolve

from astropy.io import fits
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS

from glob import glob

from tqdm import tnrange, tqdm_notebook
import warnings
warnings.filterwarnings("ignore",category =RuntimeWarning)
get_ipython().magic('matplotlib inline')
get_ipython().magic("config InlineBackend.figure_format = 'retina'")

def FindMinFrame(data):
    # Finding the reference frame
    n_steps = 12
    std_vec = np.zeros(n_steps)
    for i in range(n_steps):
        std_vec[i] = np.nanstd(data[i:-n_steps+i:n_steps,:,:] - data[i+n_steps*80,:,:])
    Framemin = np.where(std_vec==np.nanmin(std_vec))[0][0]
    return Framemin

def ObjectMask(datacube,Framemin):
    # Make a mask of the target object, using the reference frame 
    Mask = datacube[Framemin,:,:]/(np.nanmedian(datacube[Framemin,:,:])+np.nanstd(datacube[Framemin,:,:]))
    Mask[Mask>=1] = np.nan
    Mask[Mask<1] = 1
    # Generate a second mask from remainder of the first. This grabs the fainter pixels around known sources
    Maskv2 = datacube[Framemin,:,:]*Mask/(np.nanmedian(datacube[Framemin,:,:]*Mask)+np.nanstd(datacube[Framemin,:,:]*Mask))
    Maskv2[Maskv2>=1] = np.nan
    Maskv2[Maskv2<1] = 1
    Maskv3 = datacube[Framemin,:,:]*Mask/(np.nanmedian(datacube[Framemin,:,:]*Maskv2)+np.nanstd(datacube[Framemin,:,:]*Maskv2))
    Maskv3[Maskv3>=1] = np.nan
    Maskv3[Maskv3<1] = 1
    
    return Maskv3

def pix2coord(x,y,mywcs):
    wx, wy = mywcs.wcs_pix2world(x, y, 0)
    return np.array([float(wx), float(wy)])

def Get_gal_lat(mywcs,datacube):
    ra, dec = mywcs.wcs_pix2world(int(datacube.shape[1]/2), int(datacube.shape[2]/2), 0)
    b = SkyCoord(ra=float(ra)*u.degree, dec=float(dec)*u.degree, frame='icrs').galactic.b.degree
    return b



def K2tranPix(pixelfile,save): # More efficient in checking frames
    hdu = fits.open(pixelfile)
    dat = hdu[1].data
    datacube = fits.ImageHDU(hdu[1].data.field('FLUX')[:]).data
    time = dat["TIME"] + 2454833.0
    #calculate the reference frame
    Framemin = FindMinFrame(datacube)
    # Apply object mask to data
    Mask = ObjectMask(datacube,Framemin)
    Maskdata = datacube*Mask

    #Motion control
    Conv = convolve2d(np.ma.masked_invalid(Mask).mask, np.ones((3,3)), mode="same")
    Maskdata = Maskdata*(Conv < 2)

    # Calculating the standard deviations of all frames, from which the significance of each
    # frame's std can be compared.
    stddist = np.nanstd(Maskdata, axis = 0)

    Stdframe = np.ones(Maskdata.shape)
    framemask = np.zeros(Maskdata.shape)

    Index = ((np.nanstd(Maskdata,axis=(1,2)) > np.nanmedian(stddist)+2*np.nanstd(stddist)) & ((Maskdata.shape[1]>1) & (Maskdata.shape[2]>1))) 
    framemask[Index] = (Maskdata[Index]/(np.nanmedian(Maskdata[Index])+2*(np.nanstd(Maskdata[Index]))*(Conv < 2))) >= 1
    Index = ((np.nanstd(Maskdata) > np.nanmedian(stddist)+3*np.nanstd(stddist)) & ((Maskdata.shape[1]==1) | (Maskdata.shape[2] == 1))) 
    framemask[Index] = (Maskdata[Index]/(np.nanmedian(Maskdata[Index])+2*(np.nanstd(Maskdata[Index]))*(Conv < 1))) >= 1 

    # Identify if there is a sequence of consecutive or near consecutive frames that meet condtition 
    #Eventmask = (convolve(framemask,np.ones((5,3,3)),mode='constant', cval=0.0) >= 3)

    Eventmask = (convolve(framemask,np.ones((1,3,3)),mode='constant', cval=0.0) >= 1)
    Eventmask = (convolve(Eventmask,np.ones((5,1,1)),mode='constant', cval=0.0) >= 2)
    Index = np.where(np.sum(Eventmask, axis = (1,2)))[0]
    events = []
    eventtime = []

    while len(Index) > 1:
        if (Eventmask[Index[0]]*Eventmask[Index[1]]).any:
            temp = [Index[0],Index[1]]
            if np.nansum(Eventmask[Index[1]]) > np.nansum(Eventmask[Index[0]]):
                maskframe = Index[1]
            else:
                maskframe = Index[0]
            Index = np.delete(Index,1)
        else:
            events.append(maskframe)
            eventtime.append(temp)
            Index = np.delete(Index,0)
    if len(Index) ==1:
        events.append(Index[0])

    # Create an array that saves the total area of mask and time. 
    # 1st col pixelfile, 2nd duration, 3rd col area, 4th col number of events, 5th 0 if in galaxy, 1 if outside
    Result = np.zeros(5)
    # Define the coordinate system 
    funny_keywords = {'1CTYP4': 'CTYPE1',
                      '2CTYP4': 'CTYPE2',
                      '1CRPX4': 'CRPIX1',
                      '2CRPX4': 'CRPIX2',
                      '1CRVL4': 'CRVAL1',
                      '2CRVL4': 'CRVAL2',
                      '1CUNI4': 'CUNIT1',
                      '2CUNI4': 'CUNIT2',
                      '1CDLT4': 'CDELT1',
                      '2CDLT4': 'CDELT2',
                      '11PC4': 'PC1_1',
                      '12PC4': 'PC1_2',
                      '21PC4': 'PC2_1',
                      '22PC4': 'PC2_2'}
    mywcs = {}
    for oldkey, newkey in funny_keywords.items():
        mywcs[newkey] = hdu[1].header[oldkey] 
    mywcs = WCS(mywcs)

    # Check if in the galaxy plane -20 < b < 20
    b = Get_gal_lat(mywcs,datacube)
    if (float(b) > -20) and (float(b) < 20):
        Result[4] = 0 
    else:
        Result[4] = 1
    # Check through each of the identified frame sets to see if there is anything and save a figure is so

    for i in range(len(events)):
        #Find Coords of transient

        x, y = np.where(Eventmask[events[i],:,:] == 1)
        Coord = pix2coord(x[0],y[0],mywcs)

        # Generate a light curve from the transient masks
        LC = np.nansum(Maskdata*Eventmask[events[i]], axis = (1,2))

        fig = plt.figure(figsize=(10,6))
        # set up subplot grid
        gridspec.GridSpec(3,3)

        # large subplot
        plt.subplot2grid((3,3), (0,0), colspan=2, rowspan=3)
        plt.title('Masked light curve (JD '+str(round(time[events[i]]))+', RA '+str(round(Coord[0],1))+', DEC '+str(round(Coord[1],1))+')')
        plt.xlabel('Time (+'+str(time[0])+' BJD)')
        plt.ylabel('Flux')
        plt.plot(time - time[0], LC,'.')
        plt.axvspan(time[eventtime[i][0]],time[eventtime[i][1]], color = 'orange')
        # small subplot 1 Reference image plot
        plt.subplot2grid((3,3), (0,2))
        plt.title('Reference')
        plt.imshow(Maskdata[Framemin,:,:], origin='lower')
        # small subplot 2 Event mask
        plt.subplot2grid((3,3), (1,2))
        plt.title('Mask')
        plt.imshow(Eventmask[events[i]], origin='lower')
        # small subplot 3 Image of event
        plt.subplot2grid((3,3), (2,2))
        plt.title('Event')
        plt.imshow(Maskdata[events[i],:,:], origin='lower')
        # fit subplots and save fig
        fig.tight_layout()
        #fig.set_size_inches(w=11,h=7)
        plt.savefig(save+pixelfile.split('/')[-1].split('-')[0]+' '+str(i)+'.pdf', bbox_inches = 'tight');
        Result[3] += 1


    Result[0] = int(pixelfile.split('ktwo')[-1].split('-')[0])
    Result[1] = time[-1] - time[0]
    if (datacube.shape[1] > 1) and (datacube.shape[1] < 1):
        Result[2] = np.nansum((Conv < 2))
    else:
        Result[2] = np.nansum((Conv < 1))

    return Result
path = '/Volumes/TOSHIBA EXt/K2/c5/*/*/'
save = '/Users/ryanr/Documents/PhD/coding/Kepler/K2/c5/K2tranPix/'

Files = np.asarray(glob(path+'*.gz'))

Rez = []
rez = []
for filename in tqdm_notebook(Files):
    rez = K2tranPix(filename,save)
    Rez.append(rez)
    rez = []

