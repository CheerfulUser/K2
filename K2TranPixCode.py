#!/pkg/linux/anaconda3/bin/python


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
        if len(positions[0]) > 1:
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
    if abs(Time[0] - Time[1]) < 4:
        x = np.arange(middle-1,middle+1+1,1)
    else:
        x = np.arange(middle-2,middle+2+1,1)
    if x[-1] > len(lc) - 1:
        x = x[x<len(lc)]
    x2 = np.arange(0,len(x),1)
    y = lc[x]
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
            begining = Firings[(Firings >= Times[i][0]-2) & (Firings <= Times[i][0]+1)]
            if len(begining) == 0:
                begining = Quality[(Quality == Times[i][0])] #& (Quality <= Times[i][0]+1)]
            end = Firings[(Firings >= Times[i][-1]-1) & (Firings <= Times[i][-1]+2)]
            if len(end) == 0:
                end = Quality[(Quality == Times[i][-1])] #& (Quality <= Times[i][-1]+1)]
            eventthrust = Firings[(Firings >= Times[i][0]) & (Firings <= Times[i][-1])]

            if (~begining.any() & ~end.any()) & (len(eventthrust) < 3):
                
                if Asteroid_fitter(Masks[i],Times[i],Data):
                    asteroid.append(Events[i])
                    asttime.append(Times[i])
                    astmask.append(Masks[i])
                else:
                    LC = np.nansum(Data[Times[i][0]:Times[i][-1]+3]*Masks[i], axis = (1,2))
                    if (np.where(np.nanmax(LC) == LC)[0] < Range).all():
                    
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

                if ((maxloc > begining).all() & (maxloc < end)).all(): 
                    premean = np.nanmean(LC[eventthrust-1]) 
                    poststd = np.nanstd(LC[eventthrust+1])
                    postmean = np.nanmedian(LC[eventthrust+1])
                    Outsidethrust = Firings[(Firings < Times[i][0]) | (Firings > Times[i][-1]+20)]
                    Outsidemean = np.nanmedian(LC[Outsidethrust+1])
                    Outsidestd = np.nanstd(LC[Outsidethrust+1])
                    if  postmean > Outsidemean+2*Outsidestd:
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

def Save_space(Save,File):
    try:
        
        if not os.path.exists(Save+File.split('K2/')[-1]):
            os.makedirs(Save+File.split('K2/')[-1].split('ktwo')[0])
    except FileExistsError:
        pass

def K2TranPixFig(Events,Eventtime,Eventmask,Data,Time,Frames,wcs,Save,File,Quality,Thrusters,Framemin,Datacube):
    for i in range(len(Events)):
            # Check if there are multiple transients
            #Find Coords of transient
            position = np.where(Eventmask[i])

            maxcolor = np.nanmax(Data[Eventtime[i][0]:Eventtime[i][-1],(Eventmask[i]==1)])
            
            Mid = np.where(Data[Eventtime[i][0]:Eventtime[i][-1],(Eventmask[i]==1)] == maxcolor)
            
            Coord = pix2coord(Mid[1],Mid[0],wcs)
            # Generate a light curve from the transient masks
            LC = np.nansum(Data*Eventmask[i], axis = (1,2))
            BG = Data*~Frames[Events[i]]
            BG[BG <= 0] =np.nan
            BGLC = np.nanmedian(BG, axis = (1,2))
            
            Obj = np.ma.masked_invalid(Data[Framemin]).mask
            ObjLC = np.nansum(Datacube*Obj,axis = (1,2))
            ObjLC = ObjLC*np.nanmax(LC)/np.nanmax(ObjLC)
            
            
            fig = plt.figure(figsize=(10,6))
            # set up subplot grid
            gridspec.GridSpec(2,3)

            # large subplot
            plt.subplot2grid((2,3), (0,0), colspan=2, rowspan=2)
            plt.title('Event light curve (BJD '+str(round(Time[Eventtime[i][0]]-Time[0],2))+', RA '+str(round(Coord[0],3))+', DEC '+str(round(Coord[1],3))+')')
            plt.xlabel('Time (+'+str(Time[0])+' BJD)')
            plt.ylabel('Counts')
            plt.plot(Time - Time[0], LC,'.', label = 'Event LC')
            plt.plot(Time - Time[0], BGLC,'k.', label = 'Background LC')
            plt.plot(Time - Time[0], ObjLC,'kx', label = 'Scalled object LC')
            plt.axvspan(Time[Eventtime[i][0]]-Time[0],Time[Eventtime[i][-1]]-Time[0], color = 'orange', label = 'Event duration')
            plt.axvline(Time[Quality[0]]-Time[0],color = 'red', linestyle='dashed',label = 'Quality', alpha = 0.5)
            for j in range(Quality.shape[0]-1):
                j = j+1 
                plt.axvline(Time[Quality[j]]-Time[0], linestyle='dashed', color = 'red', alpha = 0.5)
            # plot Thurster firings 
            plt.axvline(Time[Thrusters[0]]-Time[0],color = 'red',label = 'Thruster', alpha = 0.5)
            for j in range(Thrusters.shape[0]-1):
                j = j+1 
                plt.axvline(Time[Thrusters[j]]-Time[0],color = 'red', alpha = 0.5)
            xmin = Time[Eventtime[i][0]]-Time[0]-(Eventtime[i][-1]-Eventtime[i][0])/10
            xmax = Time[Eventtime[i][-1]]-Time[0]+(Eventtime[i][-1]-Eventtime[i][0])/10
            if xmin < 0:
                xmin = 0
            if xmax > Time[-1] - Time[0]:
                xmax = Time[-1] - Time[0]
            plt.xlim(xmin,xmax) # originally 48 for some reason
            plt.ylim(0,np.nanmax(LC[Eventtime[i][0]:Eventtime[i][-1]])+0.1*np.nanmax(LC[Eventtime[i][0]:Eventtime[i][-1]]))
            plt.legend(loc = 1)
            # small subplot 1 Reference image plot
            plt.subplot2grid((2,3), (0,2))
            plt.title('Reference')
            plt.imshow(Data[Framemin,:,:], origin='lower',vmin=0,vmax = maxcolor)
            current_cmap = plt.cm.get_cmap()
            current_cmap.set_bad(color='black')
            plt.colorbar(fraction=0.046, pad=0.04)
            plt.plot(position[1],position[0],'r.',ms = 15)
            # small subplot 2 Image of event
            plt.subplot2grid((2,3), (1,2))
            plt.title('Event')
            plt.imshow(Data[np.where(Data*Eventmask[i]==np.nanmax(Data[Eventtime[i][0]:Eventtime[i][-1]]*Eventmask[i]))[0][0],:,:], origin='lower',vmin=0,vmax = maxcolor)
            current_cmap = plt.cm.get_cmap()
            current_cmap.set_bad(color='black')
            plt.colorbar(fraction=0.046, pad=0.04)
            plt.plot(position[1],position[0],'r.',ms = 15)
            # fit subplots and save fig
            fig.tight_layout()
            #fig.set_size_inches(w=11,h=7)
            Save_space(Save+'Figures/',File)
            
            plt.savefig(Save+'Figures/'+File.split('/')[-1].split('-')[0]+'_'+str(i)+'.pdf', bbox_inches = 'tight')
            plt.close;

def K2TranPixGif(Events,Eventtime,Eventmask,Data,wcs,Save,File):
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=1, metadata=dict(artist='Me'), bitrate=1800)
    for i in range(len(Events)):
        position = np.where(Eventmask[i])
        
        maxcolor = np.nanmax(Data[Eventtime[i][0]:Eventtime[i][-1],(Eventmask[i] == 1)])

        xmin = Eventtime[i][0]-(Eventtime[i][1]-Eventtime[i][0])
        xmax = Eventtime[i][1]+(Eventtime[i][1]-Eventtime[i][0])
        if xmin < 0:
            xmin = 0
        if xmax > len(Data):
            xmax = len(Data)-1
        Section = Data[int(xmin):int(xmax),:,:]
        fig = plt.figure()
        fig.set_size_inches(6,6)
        ims = []
        for j in range(Section.shape[0]):
            im = plt.imshow(Section[j], origin='lower',vmin = 0, vmax = maxcolor, animated=True)
            plt.plot(position[1],position[0],'r.',ms = 15)
            ims.append([im])
        plt.title(File.split('/')[-1].split('-')[0]+' Event # '+str(i))
        ani = animation.ArtistAnimation(fig, ims, interval=300, blit=True, repeat = False)
        c = plt.colorbar(fraction=0.046, pad=0.04)
        c.set_label('Counts')
        
        Save_space(Save+'Figures/',File)
        ani.save(Save+'Figures/'+File.split('/')[-1].split('-')[0]+'_Event_#_'+str(i)+'.mp4',dpi=300)
        plt.close();




def K2TranPix(pixelfile,save): # More efficient in checking frames
    start = t.time()
    Save = save + pixelfile.split('-')[1].split('_')[0]
    try:
        hdu = fits.open(pixelfile)
        dat = hdu[1].data
        datacube = fits.ImageHDU(hdu[1].data.field('FLUX')[:]).data#np.copy(testdata)#
        if datacube.shape[1] > 1 and datacube.shape[2] > 1:
            time = dat["TIME"] + 2454833.0
            Qual = hdu[1].data.field('QUALITY')
            thrusters = np.where((Qual == 1048576) | (Qual == 1089568) | (Qual == 1056768) | (Qual == 1064960) | (Qual == 1081376) | (Qual == 10240) | (Qual == 32768) )[0]
            quality = np.where(Qual != 0)[0]
            #calculate the reference frame
            Framemin = FindMinFrame(datacube)
            # Apply object mask to data
            Mask = ThrustObjectMask(datacube,thrusters)
        
            Maskdata = datacube*Mask

            # Make a mask for the object to use as a test to eliminate very bad pointings
            obj = np.ma.masked_invalid(Mask).mask
            objmed = np.nanmedian(datacube[thrusters+1]*obj,axis=(0))
            objstd = np.nanstd(datacube[thrusters+1]*obj,axis=(0))
            Maskdata[(np.nansum(datacube*obj,axis=(1,2)) < np.nansum(objmed-3*objstd)),:,:] = np.nan

            framemask = np.zeros(Maskdata.shape)

            framemask = ((Maskdata/abs(np.nanmedian(Maskdata, axis = (0))+3*(np.nanstd(Maskdata, axis = (0))))) >= 1)
            framemask[:,np.where(Maskdata > 100000)[1],np.where(Maskdata > 100000)[2]] = 0

            # Identify if there is a sequence of consecutive or near consecutive frames that meet condtition 

            Eventmask_ref = (convolve(framemask,np.ones((1,3,3)),mode='constant', cval=0.0))*1
            Eventmask = np.copy(Eventmask_ref)
            Eventmask[~np.where((convolve(Eventmask_ref,np.ones((5,1,1)),mode='constant', cval=0.0) >= 4))[0]] = 0
            Eventmask[Qual!=0,:,:] = False
            Eventmask_ref[Qual!=0,:,:] = False

            Index = np.where(np.nansum(Eventmask*1, axis = (1,2))>0)[0]


            events = []
            eventtime = []
            while len(Index) > 1:

                similar = np.where(((Eventmask[Index[0]]*Eventmask_ref[Index[0]:]) == Eventmask[Index[0]]).all(axis = (1,2)))[0]+Index[0]

                if len((np.diff(similar)<5)) > 1:



                    if len(np.where((np.diff(similar)<5) == False)[0]) > 0:
                        simEnd = np.where((np.diff(similar)<5) == False)[0][0] 
                    else:
                        simEnd = -1
                else:
                    simEnd = 0
                if (simEnd > 0):
                    similar = similar[:simEnd]
                elif (simEnd == 0):
                    similar = np.array([similar[0]])

                if len(similar) > 1:

                    events.append(similar[0])
                    temp = [similar[0]-1,similar[-1]+1]
                    eventtime.append(temp)
                    temp = []
                template = Eventmask[Index[0]]
                for number in similar:
                    if (np.nansum(template*1-Eventmask[number]*1) == 0):
                        Index = np.delete(Index, np.where(Index == number)[0])

            events, eventtime, eventmask = EventSplitter(events,eventtime,Eventmask,framemask)  
        
            events = np.array(events)
            eventmask = np.array(eventmask)
            eventtime = np.array(eventtime)

            temp = []
            for i in range(len(events)):
                if len(np.where(datacube[eventtime[i][0]:eventtime[i][-1]]*eventmask[i] > 100000)[0]) == 0:
                    temp.append(i)
            eventtime = eventtime[temp]
            events = events[temp]
            eventmask = eventmask[temp]
            if len(eventmask) > 0:
                middle = (convolve(eventmask,np.ones((1,3,3))) == np.nanmax(convolve(eventmask,np.ones((1,3,3))))) & (convolve(eventmask,np.ones((1,3,3)),mode='constant', cval=0.0) == np.nanmax(convolve(eventmask,np.ones((1,3,3)),mode='constant', cval=0.0)))
                eventmask = eventmask*middle


            # Eliminate events that begin/end within 2 cadences of a thruster fire
            events, eventtime, eventmask = EventSplitter(events,eventtime,Eventmask,framemask)  
            events = np.array(events)
            eventmask = np.array(eventmask)
            if len(eventmask) > 0:
                middle = (convolve(eventmask,np.ones((1,3,3))) == np.nanmax(convolve(eventmask,np.ones((1,3,3))))) & (convolve(eventmask,np.ones((1,3,3)),mode='constant', cval=0.0) == np.nanmax(convolve(eventmask,np.ones((1,3,3)),mode='constant', cval=0.0)))
                eventmask = eventmask*middle


            # Eliminate events that begin/end within 2 cadences of a thruster fire
            events, eventtime, eventmask, asteroid, asttime, astmask = ThrusterElim(events,eventtime,eventmask,thrusters,quality,Qual,Maskdata)
            events = np.array(events)
            eventtime = np.array(eventtime)
            eventmask = np.array(eventmask)


            coincident = []
            i = 0
            while i < len(events):
                coincident = ((eventtime[:,0] >= eventtime[i,0]-1) & (eventtime[:,0] <= eventtime[i,0]+1) & (eventtime[:,1] >= eventtime[i,1]-1) & (eventtime[:,1] <= eventtime[i,1]+1))       
                if sum(coincident*1) > 1:
                    newmask = (np.nansum(eventmask[coincident],axis = (0)) > 0)*1 

                    events = np.delete(events,np.where(coincident[1:])[0])
                    eventtime = np.delete(eventtime,np.where(coincident[1:])[0], axis = (0))
                    eventmask = np.delete(eventmask,np.where(coincident[1:])[0], axis = (0))
                    eventmask[i] = newmask

                i +=1
            
            # Save asteroids
            ast = {}
            ast['File'] = pixelfile
            ast['Asteroids'] = asteroid
            ast['Time'] = asttime
            ast['Mask'] = astmask

            astsave = Save + '/Asteroid/' + pixelfile.split('ktwo')[-1].split('-')[0]+'_Asteroid.npy'
            Save_space(Save + '/Asteroid/',pixelfile)
            np.save(astsave,ast)
            
            
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
            
            # Print the figures

            K2TranPixFig(events,eventtime,eventmask,Maskdata,time,Eventmask,mywcs,Save,pixelfile,quality,thrusters,Framemin,datacube)
            K2TranPixGif(events,eventtime,eventmask,Maskdata,mywcs,Save,pixelfile)
            
            
            Result[0] = int(pixelfile.split('ktwo')[-1].split('-')[0])
            Result[1] = (time[-1] - time[0]) - 3*len(thrusters)/48 # Last term is for removing coincident times
            Result[2] = np.nansum(Mask*1)
            Result[3] = 1*len(events)
            
        else:
            Result = np.ones(5)*np.nan
    except (OSError):
        Result = np.ones(5)*-1
    
    stop = t.time()
    
    print('Beep')
    print('Time taken=%f' %(stop-start))