
# coding: utf-8

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

def ThrustObjectMask(data,thrust):
    StartMask = np.ones((data.shape[1],data.shape[2]))
    for i in range(2):
        Start = data[thrust[:3]+1]*StartMask/(np.nanmedian(data[thrust[:3]+1]*StartMask, axis = (1,2))+np.nanstd(data[thrust[:3]+1]*StartMask, axis = (1,2)))[:,None,None]
        Start = Start >= 1
        temp = (np.nansum(Start*1, axis = 0) >=1)*1.0
        temp[temp>=1] = np.nan
        temp[temp<1] = 1
        StartMask = StartMask*temp


    EndMask = np.ones((data.shape[1],data.shape[2]))
    for i in range(2):
        End = data[thrust[-3:]+1]*EndMask/(np.nanmedian(data[thrust[-3:]+1]*EndMask, axis = (1,2))+np.nanstd(data[thrust[-3:]+1]*EndMask, axis = (1,2)))[:,None,None]
        End = End >= 1
        temp = (np.nansum(End*1, axis = 0) >=1)*1.0
        temp[temp>=1] = np.nan
        temp[temp<1] = 1
        EndMask = EndMask*temp
    
        
    Mask = np.nansum([np.ma.masked_invalid(StartMask).mask,np.ma.masked_invalid(EndMask).mask],axis=(0))*1.0
    Mask[Mask!=2] = 1
    Mask[Mask==2] = np.nan
    return Mask



def EventSplitter(events,Times,Masks,framemask):
    Events = []
    times = []
    mask = []
    for i in range(len(events)):
        # Check if there are multiple transients
        Coincident = Masks[events[i]]*framemask[events[i]]*1
        positions = np.where(Coincident == 1)
        if len(positions[0]) >1:
            for p in range(len(positions[0])):
                eventmask = np.zeros((Masks.shape[1],Masks.shape[2]))
                eventmask[positions[0][p],positions[1][p]] = 1
                eventmask = convolve(eventmask,np.ones((3,3)),mode='constant', cval=0.0)
                Similar = np.where((Masks[Times[i][0]:,:,:]*eventmask == eventmask).all(axis=(1,2)))[0]
                
                if len((np.diff(Similar)<5)) > 1:
                    
                    if len(np.where((np.diff(Similar)<5) == False)[0]) > 0:
                        SimEnd = np.where((np.diff(Similar)<5) == False)[0][0] 
                    else:
                        SimEnd = -1
                else:
                    SimEnd = 0
                #if len(np.where((np.diff(Similar)<5) == False)[0]) < 1:
                #    SimEnd = -1
                #else:
                #    SimEnd = np.where((np.diff(Similar)<5) == False)[0][0] - 1
                Similar = Similar[:SimEnd]
                if len(Similar) > 1:
                    timerange = [Similar[0]+Times[i][0]-1,Similar[-1]+Times[i][0]+1]
                    if len(timerange) > 1:
                        Events.append(events[i])
                        times.append(timerange)
                        mask.append(eventmask)
                
        else:
            Events.append(events[i])
            times.append(Times[i])
            mask.append(Masks[events[i]])
            

    return Events, times, mask

def Asteroid_fitter(Mask,Time,Data, plot = False):
    lc = np.nansum(Data*Mask,axis=(1,2))
    middle = np.where(np.nanmax(lc[Time[0]-1:Time[-1]+1]) == lc)[0][0]
    x = np.arange(middle-2,middle+2+1,1)
    x2 = np.arange(0,len(x),1)
    y = lc[[np.arange(middle-2,middle+2+1,1)]]
    p1, residual, _, _, _ = np.polyfit(x,y,2, full = True)
    p2 = np.poly1d(p1)
    AvLeft = np.nansum(abs(lc[Time[0]:Time[-1]] - p2(np.arange(Time[0],Time[-1]))))/(Time[-1]-Time[0])
    maxpoly = np.where(np.nanmax(p2(x)) == p2(x))[0][0]
    if (AvLeft < 200) &  (abs(middle - x[maxpoly]) < 2):
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

def ThrusterElim(Events,Times,Masks,Firings,Quality,qual,Data,Real_position):
    temp = []
    temp2 = []
    temp3 = []
    asteroid = []
    asttime = []
    astmask = []
    for i in range(len(Events)):
        Range = Times[i][-1] - Times[i][0]
        if (Range > 0) & (Range/Data.shape[0] < 0.8) & (Times[i][0] > 5): 
            if (Real_position*Masks[i]).any():
                print('T1')
            begining = Firings[(Firings >= Times[i][0]-3) & (Firings <= Times[i][0]+1)]
            if len(begining) == 0:
                begining = Quality[(Quality >= Times[i][0]-1) & (Quality <= Times[i][0]+1)]
            end = Firings[(Firings >= Times[i][-1]-1) & (Firings <= Times[i][-1]+3)]
            if len(end) == 0:
                end = Quality[(Quality >= Times[i][-1]-1) & (Quality <= Times[i][-1]+1)]
            eventthrust = Firings[(Firings >= Times[i][0]) & (Firings <= Times[i][-1])]
            
            if (~begining.any() & ~end.any()) & (len(eventthrust) < 3):
                
                if Asteroid_fitter(Masks[i],Times[i],Data):
                    if (Real_position*Masks[i]).any():
                        print('Asteroid')
                    asteroid.append(Events[i])
                    asttime.append(Times[i])
                    astmask.append(Masks[i])
                else:
                    temp.append(Events[i])
                    temp2.append(Times[i])
                    temp3.append(Masks[i])
                    if (Real_position*Masks[i]).any():
                        print('TSF')
            elif len(eventthrust) >= 3:
                if (Real_position*Masks[i]).any():
                    print('TL1')
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

                if ((maxloc > begining) & (maxloc < end)): 
                    postmean = np.nanmedian(LC[eventthrust[(qual[eventthrust+1]==0)]+1])
                    Outsidethrust = Firings[(Firings < Times[i][0]) | (Firings > Times[i][-1]+20)]
                    Outsidemean = np.nanmedian(LC[(Outsidethrust[(qual[Outsidethrust+1]==0)]+1)])
                    Outsidestd = np.nanstd(LC[(Outsidethrust[(qual[Outsidethrust+1]==0)]+1)])
                    if (Real_position*Masks[i]).any():
                        print(postmean)
                        print(Outsidemean)
                        print(Outsidestd)
                    if  postmean > Outsidemean+Outsidestd:
                        temp.append(Events[i])
                        temp2.append(Times[i])
                        temp3.append(Masks[i])
                        if (Real_position*Masks[i]).any():
                            print('TLF')

    events = np.array(temp)
    eventtime = np.array(temp2)
    eventmask = np.array(temp3)
    print(events)
    return events, eventtime, eventmask, asteroid, asttime, astmask


def pix2coord(x,y,mywcs):
    wx, wy = mywcs.wcs_pix2world(x, y, 0)
    return np.array([float(wx), float(wy)])

def Get_gal_lat(mywcs,datacube):
    ra, dec = mywcs.wcs_pix2world(int(datacube.shape[1]/2), int(datacube.shape[2]/2), 0)
    b = SkyCoord(ra=float(ra)*u.degree, dec=float(dec)*u.degree, frame='icrs').galactic.b.degree
    return b




def K2tranPix(data, time, Qual, mywcs, Position, Time): 
    try:
        datacube = data
        if datacube.shape[1] > 1 and datacube.shape[2] > 1:
            print('Run')
            thrusters = np.where((Qual == 1048576) | (Qual == 1089568) | (Qual == 1056768) | (Qual == 1064960) | (Qual == 1081376) | (Qual == 10240) | (Qual == 32768) )[0]
            quality = np.where(Qual != 0)[0]
            #calculate the reference frame
            Framemin = FindMinFrame(datacube)
            # Apply object mask to data
            Mask = ThrustObjectMask(datacube,thrusters)

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
            Real_position = np.zeros((data.shape[1],data.shape[2]))
            Real_position[Position[0],Position[1]] = 1


            Eventmask = (convolve(framemask,np.ones((1,3,3)),mode='constant', cval=0.0))*1
            Eventmask[~np.where((convolve(Eventmask,np.ones((5,1,1)),mode='constant', cval=0.0) >= 4))[0]] = 0
            Eventmask[Qual!=0,:,:] = False
            #print(np.where(np.nansum(Real_position*Eventmask,axis=(1,2))>0)[0])

            #Eventmask = DriftKiller(Eventmask*Maskdata,thrusters) > 0
            #Eventmask[np.isnan(Eventmask)] = 0
            Index = np.where(np.nansum(Eventmask*1, axis = (1,2))>0)[0]
            events = []
            eventtime = []
            while len(Index) > 1:
                #print('1')
                similar = Index[((Eventmask[Index[0]]*Eventmask[Index]) == Eventmask[Index[0]]).all(axis = (1,2))]
                
                #similar = similar[np.append(True, np.diff(similar)<5)]
                if len((np.diff(similar)<5)) > 1:
                    #print('2')
                    
                    if len(np.where((np.diff(similar)<5) == False)[0]) > 0:
                        simEnd = np.where((np.diff(similar)<5) == False)[0][0] 
                    else:
                        simEnd = -1
                else:
                    simEnd = 0

                if simEnd > 0:
                    similar = similar[:simEnd]
                else:
                    similar = np.array([similar[0]])

                if len(similar) > 1:
                    events.append(similar[0])
                    temp = [similar[0],similar[-1]]
                    eventtime.append(temp)
                    temp = []
                for number in similar:
                    if ((Eventmask[Index[0]]*Eventmask[number]) == Eventmask[number]).all():
                        Index = np.delete(Index, np.where(Index == number)[0])
            print(events)
            events, eventtime, eventmask = EventSplitter(events,eventtime,Eventmask,framemask)     
            #eventtime = np.array(eventtime)
            events = np.array(events)
            eventmask = np.array(eventmask)
            if (Real_position*eventmask).any():
                print('2')
                thing = events[np.where(np.nansum(Real_position*eventmask,axis=(1,2))>0)[0]]
                print(thing)
                #print(eventtime[thing])


            # Eliminate events that begin/end within 2 cadences of a thruster fire

            events, eventtime, eventmask, asteroid, asttime, astmask = ThrusterElim(events,eventtime,eventmask,thrusters,quality,Qual,Maskdata,Real_position)
            events = np.array(events)
            eventtime = np.array(eventtime)
            eventmask = np.array(eventmask)


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
                    else:
                        if (Real_position*eventmask).any():
                            print('Under dense')
                            print(np.sum(t)/t.shape[0])
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



            Real_position = np.zeros((data.shape[1],data.shape[2]))
            Real_position[Position[0],Position[1]] = 1
            detection = []
            if len(events) > 0:
                Overlap = (((Real_position*eventmask) == 1).any(axis = (1, 2))) & ((time[eventtime[:,0]] - time[Time]) < 4) & (Time < eventtime[:,0])

                detection = events[Overlap]

                false_detection = len(events[~Overlap])
            else:
                false_detection = 0.

            if len(detection) > 0:
                detect = 1.
            else:
                detect = 0.
            if np.nansum(obj*Real_position) > 0:
                Masked = 1.
            else:
                Masked = 0.

        else:
            detect = -1.
            false_detection = -1.
            Masked = -1.
    except (OSError):
        detect = -1.
        false_detection = -1.
        Masked = -1.
    return detect, false_detection, Masked

