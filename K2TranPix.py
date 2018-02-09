
# coding: utf-8

# In[ ]:

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

from scipy.signal import convolve2d
from scipy.signal import deconvolve
from scipy.ndimage.filters import convolve

from astropy.io import fits
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS

from glob import glob
import os

from tqdm import tnrange, tqdm_notebook
import warnings
warnings.filterwarnings("ignore",category =RuntimeWarning)


import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

from scipy.ndimage.filters import convolve

from astropy.io import fits
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS

from glob import glob
import os

from tqdm import tnrange, tqdm_notebook
import warnings
warnings.filterwarnings("ignore",category =RuntimeWarning)

def MinThrustframe(data,thrust):
    mean = np.nanmean(data[thrust+1],axis = 0)
    std = np.nanstd((data[thrust+1] - mean), axis = (1,2))
    Framemin = np.where(std == np.nanmin(abs(std)))[0][0]
    return thrust[Framemin]+1

def DriftKiller(data,thrust):
    # The right value choice here is a bit ambiguous, though it seems that typical variations are <10.
    Drift = (abs(data[thrust+1]-data[thrust-1]) < 10)*1.0 
    Drift[Drift == 0] = np.nan
    j = 0
    for i in range(len(thrust)):
        data[j:thrust[i]] = data[j:thrust[i]]*Drift[i]
        j = thrust[i]
    return data

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
    return Maskv2

def EventSplitter(events,Times,Masks):
    Events = []
    times = []
    mask = []
    for i in range(len(events)):
        # Check if there are multiple transients
        Coincident = convolve(Masks[events[i]]*1, np.ones((3,3)), mode='constant', cval=0.0)
        positions = np.where(Coincident == Coincident.max())
        for p in range(len(positions[0])):
            eventmask = np.zeros((Masks.shape[1],Masks.shape[2]))
            eventmask[positions[0][p],positions[1][p]] = 1
            eventmask = convolve(eventmask,np.ones((3,3)),mode='constant', cval=0.0)
            Similar = np.where(np.nansum(Masks[Times[i][0]:Times[i][-1],:,:]*eventmask,axis = (1,2)) > 0)[0]
            if len(Similar) > 0:
                timerange = [Similar[0]+Times[i][0]-1,Similar[-1]+Times[i][0]+1]
                if len(timerange) > 1:
                    Events.append(events[i])
                    times.append(timerange)
                    mask.append(eventmask)

    return Events, times, mask

def Asteroid_fitter(Mask,Time,Data, plot = False):
    lc = np.nansum(Data*Mask,axis=(1,2))
    middle = np.where(np.nanmax(lc[Time[0]-1:Time[-1]+1]) == lc)[0][0]
    x = np.arange(middle-2,middle+2+1,1)
    x2 = np.arange(0,len(x),1)
    y = lc[[np.arange(middle-2,middle+2+1,1)]]
    p1, residual, _, _, _ = np.polyfit(x,y,2, full = True)
    p2 = np.poly1d(p1)
    maxpoly = np.where(np.nanmax(p2(x)) == p2(x))[0][0]
    if (residual < 5000) &  (abs(middle - x[maxpoly]) < 2):
        asteroid = True
        if plot == True:
            p2 = np.poly1d(p1)
            plt.figure()
            plt.plot(x,lc[x],'.')
            plt.plot(x,p2(x),'.')
            plt.ylabel('Counts')
            plt.xlabel('Time')
            plt.title('Residual = ' + str(residual))
            
    else:
        asteroid = False
        
    return asteroid 

def Smoothmax(interval,Lightcurve,qual):
    x = np.arange(interval[0],interval[1],1.)
    x[qual[interval[0]:interval[-1]]!=0] = np.nan 
    nbins = int(len(x)/5)
    y = np.copy(Lightcurve[interval[0]:interval[-1]])
    y[qual[interval[0]:interval[-1]]!=0] = np.nan
    
    if np.nansum(x) > 0:
        n, _ = np.histogram(x, bins=nbins,range=(np.nanmin(x),np.nanmax(x)))
        sy, _ = np.histogram(x, bins=nbins, weights=y,range=(np.nanmin(x),np.nanmax(x)))
        sy2, _ = np.histogram(x, bins=nbins, weights=y*y,range=(np.nanmin(x),np.nanmax(x)))
        mean = sy / n
        std = np.sqrt(sy2/n - mean*mean)

        xrange = np.linspace(np.nanmin(x),np.nanmax(x),len(x))
        y_smooth = np.interp(xrange, (_[1:] + _[:-1])/2, mean)
        y_smooh_error = np.interp(xrange, (_[1:] + _[:-1])/2, std)

        temp = np.copy(y)
        temp[y_smooh_error>10] =np.nan

        maxpos = np.where(temp == np.nanmax(temp))[0]+interval[0]
    else:
        maxpos = 0
    return maxpos

def ThrusterElim(Events,Times,Masks,Firings,Quality,qual,Data):
    temp = []
    temp2 = []
    temp3 = []
    asteroid = []
    asttime = []
    astmask = []
    for i in range(len(Events)):
        Range = Times[i][-1] - Times[i][0]
        if (Range > 0) & (Range/Data.shape[0] < 0.8) & (Times[i][0] > 5): 
            begining = Firings[(Firings >= Times[i][0]-3) & (Firings <= Times[i][0]+3)]
            if len(begining) == 0:
                begining = Quality[(Quality >= Times[i][0]-1) & (Quality <= Times[i][0]+1)]
            end = Firings[(Firings >= Times[i][-1]-3) & (Firings <= Times[i][-1]+3)]
            if len(end) == 0:
                end = Quality[(Quality >= Times[i][-1]-1) & (Quality <= Times[i][-1]+1)]
            eventthrust = Firings[(Firings >= Times[i][0]) & (Firings <= Times[i][-1])]
            
            if (~begining.any() & ~end.any()) & (len(eventthrust) < 3):
                
                if Asteroid_fitter(Masks[i],Times[i],Data):
                    asteroid.append(Events[i])
                    asttime.append(Times[i])
                    astmask.append(Masks[i])
                else:
                    temp.append(Events[i])
                    temp2.append(Times[i])
                    temp3.append(Masks[i])

            elif len(eventthrust) >= 3:
                if begining.shape[0] == 0:
                    begining = 0
                else:
                    begining = begining[0]   
                if end.shape[0] == 0:
                    end = Times[i][-1] + 10
                else:
                    end = end[0]
                LC = np.nansum(Data*Masks[i], axis = (1,2))
                maxloc = Smoothmax(Times[i],LC,qual)

                if ((maxloc > begining + 5) & (maxloc < end - 5)): 

                    premean = np.nanmean(LC[eventthrust-1]) 
                    poststd = np.nanstd(LC[eventthrust+1])
                    postmean = np.nanmean(LC[eventthrust+1])
                    if  premean < postmean+2*poststd:
                        print(Events[i])
                        temp.append(Events[i])
                        temp2.append(Times[i])
                        temp3.append(Masks[i])

    events = np.array(temp)
    eventtime = np.array(temp2)
    eventmask = np.array(temp3)
    return events, eventtime, eventmask, asteroid, asttime, astmask


def pix2coord(x,y,mywcs):
    wx, wy = mywcs.wcs_pix2world(x, y, 0)
    return np.array([float(wx), float(wy)])

def Get_gal_lat(mywcs,datacube):
    ra, dec = mywcs.wcs_pix2world(int(datacube.shape[1]/2), int(datacube.shape[2]/2), 0)
    b = SkyCoord(ra=float(ra)*u.degree, dec=float(dec)*u.degree, frame='icrs').galactic.b.degree
    return b




def K2tranPix(data, time, Qual, mywcs, pixelfile, save): # More efficient in checking frames
    try:
        datacube = data
        if datacube.shape[1] > 1 and datacube.shape[2] > 1:
            thrusters = np.where((Qual == 1048576) | (Qual == 1089568) | (Qual == 1056768) | (Qual == 1064960) | (Qual == 1081376) | (Qual == 10240) | (Qual == 32768) )[0]
            quality = np.where(Qual != 0)[0]
            #calculate the reference frame
            Framemin = FindMinFrame(datacube)
            # Apply object mask to data
            Mask = ObjectMask(datacube,Framemin)

            Maskdata = datacube*Mask
            #Maskdata[Maskdata<0] = 0 

            #Motion control
            #Conv = convolve2d(np.ma.masked_invalid(Mask).mask, np.ones((3,3)), mode="same")
            #Maskdata = Maskdata*(Conv < 2)

            # Make a mask for the object to use as a test to eliminate very bad pointings
            obj = np.ma.masked_invalid(Mask).mask
            objmed = np.nanmedian(datacube[thrusters+1]*obj,axis=(0))
            objstd = np.nanstd(datacube[thrusters+1]*obj,axis=(0))
            Maskdata[(np.nansum(datacube*obj,axis=(1,2)) < np.nansum(objmed-3*objstd)),:,:] = np.nan

            #overflow = ((convolve(Maskdata[(np.nansum(datacube*obj,axis=(1,2)) > np.nansum(objmed+2*objstd)),:,:],
                                  #np.ones((1,1,2)), mode='constant', cval=0.0)) >= 1)*1.0
            #overflow[overflow == 1] = np.nan
            #overflow[overflow == 0] = 1
            #Maskdata[(np.nansum(datacube*obj,axis=(1,2)) > np.nansum(objmed+2*objstd)),:,:] = overflow
            #Maskdata[Maskdata > 170000] = np.nan
            #Stdframe = np.ones(Maskdata.shape)
            framemask = np.zeros(Maskdata.shape)

            #Index = (np.nansum(datacube*obj,axis=(1,2))>np.nansum(objmed-3*objstd)) #((np.nanstd(Maskdata,axis=(1,2)) > np.nanmedian(stddist)) & ((Maskdata.shape[1]>1) & (Maskdata.shape[2]>1))) 
            #framemask[Index] = (Maskdata[Index]/(np.nanmedian(Maskdata[Index])+2*(np.nanstd(Maskdata[Index])))) >= 1
            framemask = ((Maskdata/abs(np.nanmedian(Maskdata, axis = (0))+3*(np.nanstd(Maskdata, axis = (0))))) >= 1)
            framemask[:,np.where(Maskdata > 170000)[1],np.where(Maskdata > 170000)[2]] = 0
            #Index = ((np.nanstd(Maskdata) > np.nanmedian(stddist)+np.nanstd(stddist)) & ((Maskdata.shape[1]==1) | (Maskdata.shape[2] == 1))) 

            # Identify if there is a sequence of consecutive or near consecutive frames that meet condtition 
            #Eventmask = (convolve(framemask,np.ones((5,3,3)),mode='constant', cval=0.0) >= 3)

            Eventmask = (convolve(framemask,np.ones((1,3,3)),mode='constant', cval=0.0))*1
            Eventmask = (convolve(Eventmask,np.ones((5,1,1)),mode='constant', cval=0.0) >= 4)

            #Eventmask = DriftKiller(Eventmask*Maskdata,thrusters) > 0
            #Eventmask[np.isnan(Eventmask)] = 0
            Index = np.where(np.nansum(Eventmask*1, axis = (1,2))>0)[0]
            events = []
            eventtime = []
            if len(Index) > 0:
                masklarge = Index[0] 
                masksize = np.nansum(Eventmask[Index[0]]*1,axis = (0,1))
            temp = []
            while len(Index) > 1:
                if (Eventmask[Index[0]]*Eventmask[Index[1]]).any():
                    temp = [Index[0],Index[1]]
                    if np.nansum(Eventmask[Index[1]]*1,axis = (0,1)) > masksize:
                        masklarge = Index[1]
                        masksize = np.nansum(Eventmask[Index[1]]*1,axis = (0,1))
                    else:
                        maskframe = Index[0]
                    Index = np.delete(Index,1)
                elif len(temp) == 2:
                    events.append(masklarge)
                    eventtime.append(temp)
                    Index = np.delete(Index,0)
                    temp = []
                    masklarge = Index[0]
                    maskframe = Index[0]
                    masksize = np.nansum(Eventmask[Index[0]]*1,axis = (0,1))
                else:
                    #events.append(Index[0])
                    #eventtime.append([Index[0]])
                    Index = np.delete(Index,0)
                    temp = []
                    masklarge = Index[0]
                    maskframe = Index[0]
                    masksize = np.nansum(Eventmask[Index[0]]*1,axis = (0,1))
            if len(Index) ==1:
                events.append(Index[0])
                if len(temp) > 0:
                    eventtime.append(temp)
                else:
                    eventtime.append([Index[0]])        
            events, eventtime, eventmask = EventSplitter(events,eventtime,Eventmask)     
            #eventtime = np.array(eventtime)
            events = np.array(events)
            eventmask = np.array(eventmask)
            print(events)
            # Eliminate events that begin/end within 2 cadences of a thruster fire
            events, eventtime, eventmask, asteroid, asttime, astmask = ThrusterElim(events,eventtime,eventmask,thrusters,quality,Qual,Maskdata)
            events = np.array(events)
            eventtime = np.array(eventtime)
            eventmask = np.array(eventmask)
            print(events)
            temp = []
            temp2 = []
            temp3 = []
            for i in range(len(eventtime)):
                if len(eventtime[i])>0:
                    t = np.nansum(Eventmask[eventtime[i][0]:eventtime[i][-1],:,:]*1,axis=(1,2)) > 0
                    if np.sum(t)/t.shape[0] > 0.6:
                        temp.append(eventtime[i][:])
                        temp2.append(events[i])
                        temp3.append(eventmask[i])
            eventtime = np.array(temp)
            events = np.array(temp2)
            eventmask = np.array(temp3)

            temp = []
            for i in range(len(events)):
                if len(np.where(datacube[eventtime[i,0]:eventtime[i,-1]]*eventmask[i] > 170000)[0]) == 0:
                    temp.append(i)
            eventtime = eventtime[temp]
            events = events[temp]
            eventmask = eventmask[temp]


            # Save asteroids
            ast = {}
            ast['File'] = pixelfile
            ast['Asteroids'] = asteroid
            ast['Time'] = asttime
            ast['Mask'] = astmask

            astsave = '/Users/ryanr/Documents/PhD/coding/Kepler/K2/' + pixelfile.split('-')[1].split('_')[0] + '/Asteroid/' + pixelfile.split('ktwo')[-1].split('-')[0]+'_Asteroid.npy'
            np.save(astsave,ast)
            # 1st col pixelfile, 2nd duration, 3rd col area, 4th col number of events, 5th 0 if in galaxy, 1 if outside
            Result = np.zeros(5)


            # Check if in the galaxy plane -20 < b < 20
            b = Get_gal_lat(mywcs,datacube)
            if (float(b) > -20) and (float(b) < 20):
                Result[4] = 0 
            else:
                Result[4] = 1
            # Check through each of the identified frame sets to see if there is anything and save a figure if so

            counter = 0
            for i in range(len(events)):
                # Check if there are multiple transients
                #Find Coords of transient
                position = np.where(eventmask[i])
                middle = int(np.ceil(np.nansum(eventmask[i])/2))
                Coord = pix2coord(position[0][middle],position[1][middle],mywcs)

                # Generate a light curve from the transient masks
                LC = np.nansum(Maskdata*eventmask[i], axis = (1,2))
                BG = Maskdata*~Eventmask[events[i]]
                BG[BG <= 0] =np.nan
                BGLC = np.nanmedian(BG, axis = (1,2))

                maxcolor = np.nanmax(Maskdata[eventtime[i][0]:eventtime[i][-1]]*eventmask[i])

                fig = plt.figure(figsize=(10,6))
                # set up subplot grid
                gridspec.GridSpec(3,3)

                # large subplot
                plt.subplot2grid((3,3), (0,0), colspan=2, rowspan=3)
                plt.title('Masked light curve (BJD '+str(round(time[eventtime[i][0]]-time[0],2))+', RA '+str(round(Coord[0],3))+', DEC '+str(round(Coord[1],3))+')')
                plt.xlabel('Time (+'+str(time[0])+' BJD)')
                plt.ylabel('Flux')
                plt.plot(time - time[0], LC,'.', label = 'Event LC')
                plt.plot(time - time[0], BGLC,'k.', label = 'Background LC')
                plt.axvspan(time[eventtime[i][0]]-time[0]-.5/24,time[eventtime[i][-1]]-time[0]+.5/24, color = 'orange', label = 'Event duration')
                plt.axvline(time[quality[0]]-time[0],color = 'red', linestyle='dashed',label = 'Quality', alpha = 0.5)
                for j in range(quality.shape[0]-1):
                    j = j+1 
                    plt.axvline(time[quality[j]]-time[0], linestyle='dashed', color = 'red', alpha = 0.5)
                # plot Thurster firings 
                plt.axvline(time[thrusters[0]]-time[0],color = 'red',label = 'Thruster', alpha = 0.5)
                for j in range(thrusters.shape[0]-1):
                    j = j+1 
                    plt.axvline(time[thrusters[j]]-time[0],color = 'red', alpha = 0.5)
                xmin = time[eventtime[i][0]]-time[0]-(eventtime[i][-1]-eventtime[i][0])/10
                xmax = time[eventtime[i][-1]]-time[0]+(eventtime[i][-1]-eventtime[i][0])/10
                if xmin < 0:
                    xmin = 0
                if xmax > time[-1] - time[0]:
                    xmax = time[-1] - time[0]
                plt.xlim(xmin,xmax) # originally 48 for some reason
                plt.ylim(0,np.nanmax(LC[eventtime[i][0]:eventtime[i][-1]])+0.1*np.nanmax(LC[eventtime[i][0]:eventtime[i][-1]]))
                plt.legend(loc = 1)
                # small subplot 1 Reference image plot
                plt.subplot2grid((3,3), (0,2))
                plt.title('Reference')
                plt.imshow(Maskdata[Framemin,:,:], origin='lower',vmin=0,vmax = maxcolor)
                plt.colorbar()
                # small subplot 2 Event mask
                plt.subplot2grid((3,3), (1,2))
                plt.title('Mask')
                plt.imshow(eventmask[i], origin='lower')
                plt.colorbar()
                # small subplot 3 Image of event
                plt.subplot2grid((3,3), (2,2))
                plt.title('Event')
                plt.imshow(Maskdata[np.where(Maskdata*eventmask[i]==np.nanmax(Maskdata[eventtime[i][0]:eventtime[i][-1]]*eventmask[i]))[0][0],:,:], origin='lower',vmin=0,vmax = maxcolor)
                plt.colorbar()
                if eventmask.shape[0] == 1 or eventmask.shape[1] == 1:
                    plt.xticks([-0.5,0,0.5],['',0,''])
                # fit subplots and save fig
                fig.tight_layout()
                #fig.set_size_inches(w=11,h=7)
                plt.savefig(save+pixelfile.split('/')[-1].split('-')[0]+'_'+str(counter)+'.pdf', bbox_inches = 'tight');
                Result[3] += 1
                counter += 1


            Result[0] = int(pixelfile.split('ktwo')[-1].split('-')[0])
            Result[1] = (time[-1] - time[0]) - 3*len(thrusters)/48 # Last term is for removing coincident times
            Result[2] = np.nansum(Mask*1)
        else:
            Result = np.ones(5)*np.nan
    except (OSError):
        Result = np.ones(5)*-1
    return Result

