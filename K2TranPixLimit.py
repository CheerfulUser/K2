import imageio
import numpy as np
import pandas as pd

from scipy.ndimage.filters import convolve
from scipy.interpolate import interp1d
from astropy.stats import sigma_clip

from astropy.io import fits
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS

from xml.parsers.expat import ExpatError
from astroquery.exceptions import TableParseError
from astropy import coordinates
import astropy.units as u
from astropy.visualization import (SqrtStretch, ImageNormalize)


import csv

from glob import glob
import os
import time as t


import warnings
warnings.filterwarnings("ignore",category = RuntimeWarning)
warnings.filterwarnings("ignore",category = UserWarning)

import traceback


def ThrustObjectMask(data,thrust):
    """
    Creates a sceince target mask through standard deviation cuts on an average of stable frames. 
    To avoid masking out an event, a comparison of two masks, at the begining and end of the campaign are used.
    Points that appear in both masks are used in the final mask.
    This method has issues when the pointing completely breaks down during the campaign, such as in C10.
    """
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




def Motion_correction(Data,Mask,Thrusters,Dist):
    """
    Atempts to correct for telescope motion between individual thruster firings.
    A spline is first fitted to the stable points, and subtracted from the data.
    Next a cubic is fitted into a thruster firing interval and subtracted from the
    original data. 
    There is a check on the second derivative to ientify points that appear to not
    follow the general trend and so should not be used in fitting the cubic.

    This method still has some issues, for instance it doesn't seem to work on 
    C03 or C10.
    """
    Corrected = np.zeros((Data.shape[0],Data.shape[1],Data.shape[2]))
    fit = np.zeros(len(Data))
    X = np.where(Mask == 1)[0]
    Y = np.where(Mask == 1)[1]
    for j in range(len(X)):
        temp = np.copy(Data[:,X[j],Y[j]])
        zz = np.arange(0,len(Data))
        AvSplineind = []
        for i in range(len(Thrusters)-1):
            beep = []
            beep = Dist[Thrusters[i]+1:Thrusters[i+1]-1]
            if (beep < 0.3).any():
                datrange = Data[Thrusters[i]+1:Thrusters[i+1]-1,X[j],Y[j]]
                val = Data[np.where(beep == np.nanmin(beep))[0][0]+Thrusters[i]+1,X[j],Y[j]]
                if val < np.nanmedian(datrange) + 2*np.nanstd(datrange):
                    AvSplineind.append(np.where(beep == np.nanmin(beep))[0][0]+Thrusters[i]+1)
        AvSplineind = np.array(AvSplineind)

        if len(AvSplineind) > 1:
            AvSplinepoints = np.copy(Data[AvSplineind,X[j],Y[j]])
            Splinef = interp1d(AvSplineind, AvSplinepoints, kind='linear', fill_value=np.nan, bounds_error = False)
            Spline = Splinef(zz)

            for i in range(len(Thrusters)-1):

                if abs(Thrusters[i]-Thrusters[i+1]) > 5:
                    try:
                        Section = np.copy(Data[Thrusters[i]+2:Thrusters[i+1],X[j],Y[j]]) - Spline[Thrusters[i]+2:Thrusters[i+1]]
                        temp2 = np.copy(Section)
                        #temp2[Spline[Thrusters[i]+2:Thrusters[i+1]] == -1e10] = np.nan
                        x = np.arange(0,len(Section))
                        #limit =np.nanmedian(np.diff(np.diff(Section)))+2.5*np.nanstd(np.diff(np.diff(Section)))
                        #yo = np.where(np.diff(np.diff(Section))>limit)[0]
                        '''
                        if len(yo)/2 == int(len(yo)/2):
                            z = 0
                            while z + 1 < len(yo):
                                yoarr = np.arange(yo[z],yo[z+1])
                                temp2[yoarr] = np.nan
                                yo = np.delete(yo,[0,1])
                        else:
                            z = 0
                            while z + 2 < len(yo):
                                yoarr = np.arange(yo[z],yo[z+1])
                                temp2[yoarr] = np.nan
                                yo = np.delete(yo,[0,1])
                        if len(yo) == 1:
                            temp2[yo] = np.nan
                        '''
                        ind = np.where(~np.isnan(temp2))[0]

                        if (len(x[ind]) > 3) & (len(x[ind])/len(x) > 0.6):
                            polyfit, resid, _, _, _  = np.polyfit(x[ind], Section[ind], 3, full = True)
                            p3 = np.poly1d(polyfit)

                            if np.abs(resid/len(x[ind])) < 10:
                                temp[x+Thrusters[i]+2] = np.copy(Data[Thrusters[i]+2:Thrusters[i+1],X[j],Y[j]]) - p3(x) 
                        # This should kill all instances of uncorrected data due to drift systematically being > 0.3 pix
                        if (np.isnan(Spline[Thrusters[i]+2:Thrusters[i+1]])).all():
                            temp[x+Thrusters[i]+2] = np.nan
                    except RuntimeError:
                        pass

        Corrected[:,X[j],Y[j]] = temp                    
    return Corrected

def pix2coord(x,y,mywcs):
    """
    Calculates RA and DEC from the pixel coordinates
    """
    wx, wy = mywcs.wcs_pix2world(x, y, 0)
    return np.array([float(wx), float(wy)])

def Get_gal_lat(mywcs,datacube):
    """
    Calculates galactic latitude, which is used as a flag for event rates.
    """
    ra, dec = mywcs.wcs_pix2world(int(datacube.shape[1]/2), int(datacube.shape[2]/2), 0)
    b = SkyCoord(ra=float(ra)*u.degree, dec=float(dec)*u.degree, frame='icrs').galactic.b.degree
    return b



def Local_Gal_Check(Datacube,Limit,WCS,File,Save):
    """
    Checks every pixel for coincidence with the Pan-STARRS galaxy cataloge. 
    """
    Database_location = '/avatar/ryanr/Data/Catalog/PS/' 
    Campaign = File.split('-')[1].split('_')[0].split('c')[1]
    valid_pix = np.nansum(abs(Datacube),axis=(0)) > 0
    if np.nansum(valid_pix) > 0:
	    Y, X = np.where(valid_pix)

	    result_table = pd.read_csv(Database_location + 'Gal_c' + Campaign + '_Cheerful.csv').values
	    keys = pd.read_csv(Database_location + 'Gal_c' + Campaign + '_Cheerful.csv').key
	    header = []
	    for k in keys:
	        header.append(k)
	    header.append('Maglim')
	    footprint = WCS.calc_footprint()
	    padding = 4/3600
	    min_ra = np.nanmin(footprint[:,0]) - padding
	    max_ra = np.nanmax(footprint[:,0]) + padding
	    min_dec = np.nanmin(footprint[:,1]) - padding
	    max_dec = np.nanmax(footprint[:,1]) + padding

	    ind1 = np.where((result_table[:,1] >= min_ra) & (result_table[:,1] <= max_ra))
	    temp = result_table[ind1,:]
	    ind2 = np.where((temp[:,1] >= min_dec) & (temp[:,2] <= max_dec))
	    gals = temp[ind2,:]

	    for i in range(len(X)):
	        coord = pix2coord(X[i],Y[i],WCS)

	        c = coordinates.SkyCoord(ra=coord[0], dec=coord[1],unit=(u.deg, u.deg), frame='icrs')
	        ra = c.ra.deg
	        dec = c.dec.deg
	        dist = np.sqrt((gals[:,1] - ra)**2 + (gals[:,2] - dec)**2)
	        radius = 2/3600 # Convert arcsec to deg 
	        short_list = gals[dist <= radius]
	        for item in short_list:
	            CVSstring = []
	            for ele in item:
	                CVSstring.append(ele)
	            CVSstring.append(Limit[Y[i],X[i]])

	            Save_space(Save+'/Gals/')
	            Path = Save + '/Gals/' + File.split('/')[-1].split('-')[0] + '_Gs.csv'
	            
	            if os.path.isfile(Path):
	                with open(Path, 'a') as csvfile:
	                    spamwriter = csv.writer(csvfile, delimiter=',')
	                    spamwriter.writerow(CVSstring)
	            else:
	                with open(Path, 'w') as csvfile:
	                    spamwriter = csv.writer(csvfile, delimiter=',')
	                    spamwriter.writerow(header)
	                    spamwriter.writerow(CVSstring)
		else:
			print('Corrupt file')
    return


def Save_space(Save):
    """
    Creates a pathm if it doesn't already exist.
    """
    try:
        if not os.path.exists(Save):
            os.makedirs(Save)
    except FileExistsError:
        pass





def Long_events_limit(Data, Time, Mask, Dist, Save, File):
    '''
    Saves the magnitude limit per pixel for events longer than 2 days.

    Data - 3d fux array
    Time - 1d time array
    Mask - 2d masked pixel array
    Dist - 1d displacement array
    Save - Save directory
    File - TPF name
    '''
    sub = np.zeros(Data[0].shape)
    limit = np.zeros(Data[0].shape)
    good_frames = np.where(Dist < 0.3)[0]

    dim1,dim2 = Data[0].shape
    for i in range(dim1):
        for j in range(dim2):

            lc = np.copy(Data[:,i,j])#[good_frames,i,j]
            #lc[lc < 0] = 0

            condition = np.nanmedian(lc) + np.nanstd(lc)
            diff = np.diff(Time[lc < condition])
            ind = np.where(lc < condition)[0]
            lc2 = np.copy(lc)
            for k in range(len(diff)):
                if diff[k] < 1:
                    section = np.copy(lc[ind[k]:ind[k+1]])
                    
                    section[section > condition] = np.nan
                    lc2[ind[k]:ind[k+1]] = section
                    

            if np.isnan(Mask[i,j]):
                sub[i,j] = abs((np.nanmean(lc2) - np.nanmedian(lc2)))
                
                    
            elif ~np.isnan(Mask[i,j]):
                if np.nanmedian(lc2) != 0:
                    sub[i,j] = abs(1-(np.nanmean(lc2) / np.nanmedian(lc2)))
                else:
                    sub[i,j] = abs(1-(np.nanmean(lc2) / 1e-15))
                

                
    
    cutbkg = np.nanmedian(sub*Mask) + 2*np.nanstd(sub*Mask)
    ob = np.ma.masked_invalid(Mask).mask
    cutobj = np.nanmedian(sub*ob) + 2*np.nanstd(sub*ob)
    ob = ob*1.
    ob[ob == 0] = np.nan

    limit = np.zeros((2,sub.shape[0],sub.shape[1]))
    
    limit[0,sub*Mask>=cutbkg] = sub[sub*Mask>=cutbkg]
    limit[0,sub*Mask<cutbkg] = cutbkg
    
    limit[1,sub*ob>=cutobj] = sub[sub*ob>=cutobj]
    limit[1,sub*ob<cutobj] = cutobj

    limitfile = np.zeros((3,limit.shape[1],limit.shape[2]))
    limitfile[0] = limit[0]
    limitfile[1] = limit[1]
    limitfile[2] = np.nanstd(sub*Mask)
 	
    Limitsave = Save + '/Limit/' + File.split('ktwo')[-1].split('-')[0]+'_VLimit'
    Save_space(Save + '/Limit/')
    np.save(Limitsave,limitfile)
    return





def K2TranPix_limit(pixelfile,save): 
    """
    Main code yo. Runs an assortment of functions to detect events in Kepler TPFs.
    """
    Save = save + pixelfile.split('-')[1].split('_')[0]
    try:
        hdu = fits.open(pixelfile)
        if len(hdu) > 1:
            dat = hdu[1].data
        else:
            print('Broken file ', pixelfile)
            return
        datacube = fits.ImageHDU(hdu[1].data.field('FLUX')[:]).data#np.copy(testdata)#
        if datacube.shape[1] > 1 and datacube.shape[2] > 1:
            ind = np.where(np.isfinite(datacube[0]))
            for i in range(len(ind)):
                lc = datacube[:,ind[0][i],ind[1][i]]
                datacube[sigma_clip(lc,sigma=5.).mask,ind[0][i],ind[1][i]] = np.nan

            time = dat["TIME"] + 2454833.0
            Qual = hdu[1].data.field('QUALITY')
            thrusters = np.where((Qual == 1048576) | (Qual == 1089568) | (Qual == 1056768) | (Qual == 1064960) | (Qual == 1081376) | (Qual == 10240) | (Qual == 32768) | (Qual == 1097760) | (Qual == 1048580) | (Qual == 1081348))[0]
            thrusters = np.insert(thrusters,0,-1)
            thrusters = np.append(thrusters,len(datacube)-2)
            quality = np.where(Qual != 0)[0]
            
            xdrif = dat['pos_corr1']
            ydrif = dat['pos_corr2']
            distdrif = np.sqrt(xdrif**2 + ydrif**2)
            goodthrust = thrusters[np.where(distdrif[thrusters]<0.2)]
            #calculate the reference frame
            if len(goodthrust) > 4:
                Framemin = goodthrust[3]+1
            elif len(goodthrust) > 0:
                Framemin = goodthrust[0]+1
            else:
                Framemin = 100 # Arbitrarily chosen, Data is probably screwed anway if there are no thruster firings.
            # Apply object mask to data
            Mask = ThrustObjectMask(datacube,goodthrust)

            #Maskdata, ast = First_pass(np.copy(datacube),Qual,quality,thrusters,pixelfile)
            Maskdata = np.copy(datacube)
            allMask = np.ones((datacube.shape[1],datacube.shape[2]))
            Maskdata = Motion_correction(Maskdata,allMask,thrusters,distdrif)

            # Make a mask for the object to use as a test to eliminate very bad pointings
            obj = np.ma.masked_invalid(Mask).mask
            objmed = np.nanmedian(datacube[thrusters+1]*obj,axis=(0))
            objstd = np.nanstd(datacube[thrusters+1]*obj,axis=(0))

            framemask = np.zeros(Maskdata.shape)

            limit = abs(np.nanmedian(Maskdata[Qual == 0], axis = (0))+3*(np.nanstd(Maskdata[Qual == 0], axis = (0))))
            limit[limit<22] = 22
            
            limitfile = np.zeros((2,limit.shape[0],limit.shape[1]))
            limitfile[0] = limit
            limitfile[1] = np.nanstd(Maskdata[Qual == 0], axis = (0))

            Limitsave = Save + '/Limit/' + pixelfile.split('ktwo')[-1].split('-')[0]+'_Limit'
            Save_space(Save + '/Limit/')
            np.savez(Limitsave,limitfile)
            
            
            # Create an array that saves the total area of mask and time. 
            # 1st col pixelfile, 2nd duration, 3rd col area, 4th col number of events, 5th 0 if in galaxy, 1 if outside
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
            
            # Save thrusts and quality flags and time 
            Fieldprop = {}
            Fieldprop['File'] = pixelfile
            Fieldprop['Thruster'] = thrusters
            Fieldprop['Quality'] = quality
            Fieldprop['Duration'] = len(time)
            Fieldprop['Gal_lat'] = Get_gal_lat(mywcs,datacube)
            
            Fieldsave = Save + '/Field/' + pixelfile.split('ktwo')[-1].split('-')[0]+'_Field'
            Save_space(Save + '/Field/')
            np.savez(Fieldsave, Fieldprop)

            Local_Gal_Check(Maskdata, limit, mywcs, pixelfile, Save)

            Long_events_limit(Maskdata, time, Mask, distdrif, Save, pixelfile)
        	
        else:
            print('Small ', pixelfile)
    except (OSError):
        print(pixelfile)
        traceback.print_exc()
        pass  