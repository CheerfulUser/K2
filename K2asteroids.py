import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
#import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

from scipy.ndimage.filters import convolve
from scipy.interpolate import interp1d
from astropy.stats import sigma_clip

from astropy.io import fits
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS

from scipy import ndimage as ndi
from skimage.morphology import watershed
from skimage.feature import peak_local_max

from astroquery.simbad import Simbad
from astroquery.ned import Ned
from astroquery.ned.core import RemoteServiceError
from xml.parsers.expat import ExpatError
from astroquery.exceptions import TableParseError
from astropy import coordinates
import astropy.units as u
from astropy.visualization import (SqrtStretch, ImageNormalize)

from astropy import convolution 

from pywt import wavedec
from scipy.signal import find_peaks
from scipy.interpolate import PchipInterpolator
from scipy.signal import savgol_filter


import operator
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

from requests.exceptions import ConnectionError
from requests.exceptions import ReadTimeout
from requests.exceptions import ChunkedEncodingError

import csv

from glob import glob
import os
import time as t

from marth_asteroid import *

import warnings
#warnings.filterwarnings("ignore",category = RuntimeWarning)
#warnings.filterwarnings("ignore",category = UserWarning)

warnings.filterwarnings("ignore")

import traceback


def Clip_cube(Data):
	"""
	Performs a sigma clip on the light curves of all pixels in the data.
	Data - 3d array with time and pixel values
	"""
	data = np.copy(Data)
	frame = np.nanmedian(Data,axis=(0))
	frame[frame==0] = np.nan
	ind = np.where(np.isfinite(frame))
	for i in range(len(ind)):
		lc = data[:,ind[0][i],ind[1][i]]
		data[sigma_clip(lc,sigma=5.).mask,ind[0][i],ind[1][i]] = np.nan
	return data


def ObjectMask(Data,Dist):
	"""
	Creates a sceince target mask through standard deviation cuts on an average of stable frames. 
	To avoid masking out an event, a comparison of two masks, at the begining and end of the campaign are used.
	Points that appear in both masks are used in the final mask.
	This method has issues when the pointing completely breaks down during the campaign, such as in C10.
	"""
	data = Data[Dist < 0.1]
	if len(data) < 6:
		data = Data[Dist < 0.2]

	StartMask = np.ones((data.shape[1],data.shape[2]))
	for i in range(2):
		Start = data[:3]*StartMask/(np.nanmedian(data[:3]*StartMask, 
				axis = (1,2)))[:,None,None]
		Start = Start >= 1
		temp = (np.nansum(Start*1, axis = 0) >=1)*1.0
		temp[temp>=1] = np.nan
		temp[temp<1] = 1
		StartMask = StartMask*temp


	EndMask = np.ones((data.shape[1],data.shape[2]))
	for i in range(2):
		End = data[-3:]*EndMask/(np.nanmedian(data[-3:]*EndMask, 
			  axis = (1,2)))[:,None,None]
		End = End >= 1
		temp = (np.nansum(End*1, axis = 0) >=1)*1.0
		temp[temp>=1] = np.nan
		temp[temp<1] = 1
		EndMask = EndMask*temp
	
		
	Mask = np.nansum([np.ma.masked_invalid(StartMask).mask,np.ma.masked_invalid(EndMask).mask],axis=(0))*1.0
	Mask[Mask!=2] = 1
	Mask[Mask==2] = np.nan
	return Mask

def Event_ID(Sigmask, Significance, Minlength, Smoothing = True):
	"""
	Identifies events in a datacube, with a primary input of a boolean array for where pixels are 3std above background.
	Event duration is calculated by differencing the positions of False values in the boolean array.
	The event mask is saved as a tuple.
	"""
	binary = Sigmask >= Significance


	tarr = np.copy(binary)
	summed_binary = np.nansum(binary,axis=0)
	leng = 10
	X = np.where(summed_binary >= Minlength)[0]
	Y = np.where(summed_binary >= Minlength)[1]

	if Smoothing:
		for i in range(leng-2):
			kern = np.zeros((leng, 1, 1))
			kern[[0, -1]] = 1
			tarr[convolve(tarr*1, kern) > 1] = True
			leng -= 1

	events = []
	eventtime = []
	eventtime = []
	eventmask = []

	for i in range(len(X)):
		temp = np.insert(tarr[:,X[i],Y[i]],0,False) # add a false value to the start of the array
		testf = np.diff(np.where(~temp)[0])
		indf = np.where(~temp)[0]
		testf[testf == 1] = 0
		testf = np.append(testf,0)


		if len(indf[testf>Minlength]) > 0:
			for j in range(len(indf[testf>Minlength])):
				if abs((indf[testf>Minlength][j] + testf[testf>Minlength][j]-1) - indf[testf>Minlength][j]) < 48: # Condition on events shorter than a day 
					start = indf[testf>Minlength][j]
					end = (indf[testf>Minlength][j] + testf[testf>Minlength][j]-1)
					if np.nansum(binary[start:end,X[i],Y[i]]) / abs(end-start) > 0.5:
						events.append(indf[testf>Minlength][j])
						eventtime.append([indf[testf>Minlength][j], (indf[testf>Minlength][j] + testf[testf>Minlength][j]-1)])
						masky = [np.array(X[i]), np.array(Y[i])]
						eventmask.append(masky)	
				else:
					events.append(indf[testf>Minlength][j])
					eventtime.append([indf[testf>Minlength][j], (indf[testf>Minlength][j] + testf[testf>Minlength][j]-1)])
					masky = [np.array(X[i]), np.array(Y[i])]
					eventmask.append(masky)

	events = np.array(events)
	eventtime = np.array(eventtime)
	return events, eventtime, eventmask



def Asteroid_fitter(Mask,Time,Data, plot = False):
	"""
	Simple method to remove asteroids. This opperates under the assumption that asteroids travel through the field, 
	and pixel, at a uniform velocity, thus creating a parabolic light curve. If the event fits a parabola, then it 
	is assumed to be an asteroid.
	This method doesn't work for asteroids that change their direction of motion.
	"""
	lc = np.nansum(Data*Mask,axis=(1,2))
	middle = np.where(np.nanmax(lc[Time[0]-1:Time[-1]+1]) == lc)[0][0]
	if abs(Time[0] - Time[1]) < 4:
		x = np.arange(middle-1,middle+1+1,1)
	else:
		x = np.arange(middle-2,middle+2+1,1)
	if x[-1] > len(lc) - 1:
		x = x[x<len(lc)]
	x2 = np.arange(0,len(x),1)
	if np.nanmedian(lc[x]) >0:
		y = lc[x]/np.nanmedian(lc[x])
		p1, residual, _, _, _ = np.polyfit(x,y,2, full = True)
		p2 = np.poly1d(p1)
		AvLeft = residual/len(x)#np.nansum(abs(lc[Time[0]:Time[-1]]/np.nanmedian(lc[x]) - p2(np.arange(Time[0],Time[-1]))))/(Time[-1]-Time[0])
		maxpoly = np.where(np.nanmax(p2(x)) == p2(x))[0][0]
		if (AvLeft < 10) &  (abs(middle - x[maxpoly]) < 2):
			asteroid = True
			if plot == True:
				p2 = np.poly1d(p1)
				plt.figure()
				plt.plot(x,y,'.',label='Event LC')
				plt.plot(x,p2(x),'kx',label='Parabola fit')
				plt.axvspan(Time[0],Time[1], color = 'orange',alpha=0.5, label = 'Event duration')
				plt.ylabel('Counts')
				plt.xlabel('Time (frames)')
				plt.legend()
				plt.title('Residual = {}'.format(AvLeft[0]))
				plt.savefig('test{}.pdf'.format(np.round(AvLeft[0])))
		else:
			asteroid = False
	else:
		asteroid = False

	return asteroid

def Smoothmax(interval,Lightcurve,qual):
	"""
	Calculates the time for the maximum value of a light curve, after smoothing the data.
	"""
	nanind = np.where(qual[interval[0]:interval[1]]!=0)[0]
	x = np.arange(interval[0],interval[1],1.)
	x[nanind] = np.nan 
	#print(x)
	nbins = int(len(x)/5)
	if nbins > 0:
		y = np.copy(Lightcurve[interval[0]:interval[1]])
		y[nanind] = np.nan
		
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
	else:
		maxpos = -100

	return maxpos


def Vet_brightness(Event, Eventtime, Eventmask, Data, Quality,pixelfile):
	good_ind = np.zeros(len(Event))
	
	for i in range(len(Eventtime)):
		mask = np.zeros(Data.shape[1:])
		mask[Eventmask[i]] = 1
		mask = mask > 0
		
		LC = Lightcurve(Data,mask)
		outside_mask = np.ones(len(LC))
		lower = Eventtime[i][0] - 2*48
		upper = Eventtime[i][1] + 10*48
		if lower < 0:
			lower = 0
		if upper > len(LC):
			upper = -1

		outside_mask[lower:upper] = 0
		outside_mask = outside_mask > 0

		median = np.nanmedian(LC[outside_mask])
		std = np.nanstd(LC[outside_mask])
		event_max = Smoothmax(Eventtime[i],LC,Quality)
		if type(event_max) == int:
			bright = np.round((LC[event_max]-median)/std,1)
			if bright > 3:
				good_ind[i] = 1

		elif len(event_max) > 0:
			bright = np.round((LC[event_max[0]]-median)/std,1)
		
			if bright > 3:
				good_ind[i] = 1
	
	good_ind = good_ind > 0
	event_bright = Event[good_ind] 
	eventtime_bright = Eventtime[good_ind] 
	mask_ind = np.where(~good_ind)[0]
	for i in range(len(mask_ind)):
		rev = len(mask_ind) -1 - i
		del Eventmask[mask_ind[rev]]

	if len(Eventmask) != len(event_bright):
		raise ValueError('Arrays are different lengths, check whats happening in {}'.format(pixelfile))

	return event_bright, eventtime_bright, Eventmask



def Identify_event(Data, Position):
	"""
	Uses an iterrative process to find spacially seperated masks in the object mask.
	"""
	
	p1 = Position[0]
	p2 = Position[1]
	
	events = np.copy(Data)
	masktemp = np.zeros_like(events)

	masktemp[p1,p2] = 1
	size = 0
	while np.nansum(masktemp) != size:
		size = np.nansum(masktemp)
		conv = ((convolve(masktemp*1,np.ones((3,3)),mode='constant', cval=0.0)) > 0)*1.0
		masktemp[(conv*events) > 0] = 1
	
	x,y = np.where(masktemp)
	return [x,y]

def Match_events(Events, Eventtime, Eventmask, Data):
	"""
	Matches flagged pixels that have concurrent event times and are nieghbouring.
	"""
	eventpos = np.zeros((len(Events),Data.shape[1],Data.shape[2]))

	for i in range(len(Eventmask)):
		eventpos[i,Eventmask[i][0],Eventmask[i][1]] = 1 

	for i in range(len(Events)):
		if (Events[i] >= 0):
			duration = Eventtime[i,1]- Eventtime[i,0]
			start = Eventtime[i,0] - duration
			if start < 0:
				start = 0
			end = Eventtime[i,1] + 2*duration
			if end >= len(Data) - 1:
				end = len(Data) - 1
			
			t_coinc = np.where((Eventtime[:,0] >= start) & (Eventtime[:,1] <= end))[0]

			if len(t_coinc) > 1:
				summed_events = np.nansum(eventpos[t_coinc,:,:], axis = 0)
				combined_mask = Identify_event(summed_events,Eventmask[i])
				
				newmask = np.zeros_like(Data[0])
				newmask[combined_mask] = 1

				ind = []
				for t in t_coinc:
					if np.nansum(newmask * eventpos[t]) > 0:
						ind += [t]
				ind = np.array(ind,dtype='int')
				
				if len(ind[ind>0]) > 0:
					Eventmask[i] = combined_mask
					new_start = np.nanmin(Eventtime[ind[ind>0],0])
					new_end = np.nanmax(Eventtime[ind[ind>0],1])
					temp = Events[i]
					Events[ind[:]] = -10 # flag for identification later
					Events[i] = temp
					temp = [new_start,new_end]
					Eventtime[ind[:]] = -10
					Eventtime[i] = temp

			
	eh = np.where(Events > 0)[0]			
	Events = Events[eh]
	Eventtime = Eventtime[eh]
	Eventmask = np.array(Eventmask)[eh].tolist()
	return Events, Eventtime, Eventmask


def Get_all_resets(Data, Quality):
	allflux = np.nansum(Data,axis=0)
	ind1, ind2 = np.where(np.nanmax(allflux) == allflux)

	nonan = Data[np.isfinite(Data[:,ind1[0],ind2[0]]),ind1[0],ind2[0]]
	nonaninds = np.where(np.isfinite(Data[:,ind1[0],ind2[0]]))[0]
	coeffs = wavedec(nonan, 'db2',level=20)
	eh = abs(coeffs[-1][:-1])# / nonan[::2]
	eh_x = np.arange(0,len(coeffs[-1])*2,2)
	peaks = find_peaks(eh,distance=4,prominence=np.mean(eh))[0]

	peaks = nonaninds[eh_x[peaks]]

	realt = np.where((Quality==1048576) | (Quality==524288) | 
					 (Quality==1081376) | (Quality==1056768))[0]
	o = realt.copy()
	print('quality thrusters: ',len(realt))
	print('wavelet thrusters: ',len(peaks))
	for p in peaks:
		if ~np.isclose(p, realt, atol=3).any():
			realt = np.append(realt, p)

	realt = np.sort(realt)
	print('diff: ',len(realt)-len(o))
	
	return realt

def Regress_fit(Data, Fit = True):
	ind = np.where(np.isfinite(Data))[0]
	x = np.arange(0,len(Data))
	x = x[ind]
	y = Data[ind]
	if len(y) >= 3:
		x = x[:, np.newaxis]
		y = y[:, np.newaxis]

		polynomial_features= PolynomialFeatures(degree=3)
		x_poly = polynomial_features.fit_transform(x)

		model = LinearRegression()
		model.fit(x_poly, y)
		y_poly_pred = model.predict(x_poly)

		sort_axis = operator.itemgetter(0)
		sorted_zip = sorted(zip(x,y_poly_pred), key=sort_axis)
		x, y_poly_pred = zip(*sorted_zip)

		r2 = r2_score(y,y_poly_pred)

		mod = np.zeros_like(Data[ind])
		xx = np.zeros_like(Data[ind])
		for i in range(len(mod)):
			mod[i] = y_poly_pred[i][0]
			xx[i] = x[i][0]
		temp = interp1d(xx,mod,bounds_error = False,fill_value='extrapolate')
		mod = temp(np.arange(0,len(Data)))
		
	else:
		mod = np.zeros_like(Data)
		mod[:] = np.nan
	fit = Data.copy()

	fit = np.squeeze(mod)
	
	if Fit:
		return fit
	else:
		return r2

def Median_clip(data, sigma=3):
	med = np.nanmedian(data)
	std = np.nanstd(data)
	mask = data > (med + sigma * std)
	return mask

def Correct_motion(Data, Distance, Thrust):
	data = Data.copy()
	data[Thrust] = np.nan
	data[Thrust[:-1]+1] = np.nan
	#data[Thrust[:-1]+2] = np.nan
	data[Thrust[1:]-1] = np.nan
	X, Y = np.where(np.nansum(data,axis=0) != 0)
	
	fitting = data.copy()
	spline = data.copy()
	x = np.arange(data.shape[0])
	for j in range(len(X)):
		trend = []
		for i in range(len(Thrust)-1):
			section = data[Thrust[i]:Thrust[i+1]-1,X[j],Y[j]].copy()
			nonan = np.where(np.isfinite(section))[0]
			if len(nonan) > 0:

				nanmask = Median_clip(section, sigma=2)#sigma_clip(section,sigma=2,masked=True,maxiters=1).mask
				if np.nansum(nanmask) > 3:
					nanmask = Median_clip(section, sigma=3)#sigma_clip(section,sigma=3,masked=True,maxiters=1).mask

				section[nanmask] = np.nan


				fitting[Thrust[i]:Thrust[i+1]-1,X[j],Y[j]] = Regress_fit(section)

				d = Distance[Thrust[i]:Thrust[i+1]-1]
				ind = np.where(np.isfinite(section))[0]
				
					
				if (d[ind] <= 0.2).any():
					gi = np.where(d[ind] <= 0.2)[0]
					p = np.average(ind[gi],weights=1-d[ind[gi]])
					val = np.average(section[ind[gi]],weights=1-d[ind[gi]])
					trend += [[p+Thrust[i],val]]
				elif (np.abs(np.nanmedian(section)) <= 5) & \
					 (np.nansum(np.abs(section)) > 0) & \
					 (d[ind] < 1).any():
					gi = np.where(d[ind] < 1)[0]
					p = np.average(ind[gi],weights=1-d[ind[gi]])
					val = np.average(section[ind[gi]],weights=1-d[ind[gi]])
					trend += [[p+Thrust[i],val]]

					

		trend = np.array(trend)
		trend = trend[trend[:,0].argsort()]
		#return trend
		if len(trend) > 2:
			spl = PchipInterpolator(trend[:,0], trend[:,1],extrapolate=False)
			#spl = interp1d(trend[:,0], trend[:,1], kind = 'linear',bounds_error=False)
			x = np.arange(data.shape[0])
			spl = spl(x)

			spline[:,X[j],Y[j]] = spl
		else:
			spline[:,X[j],Y[j]] = np.nan

	data = data - fitting + spline
	
	return data#, fitting, spline


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

def Identify_masks(Obj):
	"""
	Uses an iterrative process to find spacially seperated masks in the object mask.
	"""
	objsub = np.copy(Obj)
	Objmasks = []

	mask1 = np.zeros((Obj.shape))
	if np.nansum(objsub) > 0:
		mask1[np.where(objsub==1)[0][0],np.where(objsub==1)[1][0]] = 1
		while np.nansum(objsub) > 0:

			conv = ((convolve(mask1*1,np.ones((3,3)),mode='constant', cval=0.0)) > 0)*1.0
			objsub = objsub - mask1
			objsub[objsub < 0] = 0

			if np.nansum(conv*objsub) > 0:
				
				mask1 = mask1 + (conv * objsub)
				mask1 = (mask1 > 0)*1
			else:
				
				Objmasks.append(mask1)
				mask1 = np.zeros((Obj.shape))
				if np.nansum(objsub) > 0:
					mask1[np.where(objsub==1)[0][0],np.where(objsub==1)[1][0]] = 1
	return Objmasks

def Watershed_object_sep(obj):
	"""
	Uses the watershed method to identify components in the object mask. 
	"""
	obj = obj*1
	distance = ndi.distance_transform_edt(obj)
	local_maxi = peak_local_max(distance, indices=False, footprint=np.ones((3, 3)),labels=obj)
	markers = ndi.label(local_maxi)[0]
	labels = watershed(-distance, markers, mask=obj)

	temp = np.nanmax(labels)
	Objmasks = []
	for i in range(temp):
		i += 1
		#print(i)
		Objmasks += [(labels == i)*1]

	Objmasks = np.array(Objmasks)
	return Objmasks

def Database_event_check(Data, Eventtime, Eventmask, WCS):
	"""
	Checks Ned and Simbad to check the event position against known objects.
	"""
	Objects = []
	Objtype = []
	for I in range(len(Eventtime)):
		mask = np.zeros((Data.shape[1], Data.shape[2]))
		mask[Eventmask[I][0], Eventmask[I][1]] = 1
		maxcolor = np.nanmax(
			Data[Eventtime[I][0]:Eventtime[I][-1]]*(mask == 1))

		Mid = np.where(Data[Eventtime[I][0]:Eventtime[I][-1]]
					   * (mask == 1) == maxcolor)
		if len(Mid[0]) == 1:
			Coord = pix2coord(Mid[1], Mid[0], WCS)
		elif len(Mid[0]) > 1:
			Coord = pix2coord(Mid[1][0], Mid[0][0], WCS)

		c = SkyCoord(ra=Coord[0], dec=Coord[1],
					 unit=(u.deg, u.deg), frame='icrs')

		Ob = 'Unknown'
		objtype = 'Unknown'
		try:
			result_table = Ned.query_region(
				c, radius=4*u.arcsec, equinox='J2000')
			Ob = np.asarray(result_table['Object Name'])[0].decode("utf-8")
			objtype = result_table['Type'][0].decode("utf-8")

			if '*' in objtype:
				objtype = objtype.replace('*', 'Star')
			if '!' in objtype:
				objtype = objtype.replace('!', 'Gal')  # Galactic sources
			if objtype == 'G':
				try:
					result_table = Simbad.query_region(c, radius=4*u.arcsec)
					if len(result_table.colnames) > 0:
						objtype = objtype + 'Simbad'
				except (AttributeError, ExpatError, TableParseError, ValueError, 
					EOFError, IndexError, ConnectionError,ReadTimeout,ChunkedEncodingError) as e:
					#print('Simbad fail event')
					pass

		except (RemoteServiceError, ExpatError, TableParseError, ValueError, 
			EOFError, IndexError, ConnectionError,ReadTimeout, ChunkedEncodingError) as e:
			#print('Ned fail event')
			try:
				result_table = Simbad.query_region(c, radius=4*u.arcsec)
				if len(result_table.colnames) > 0:
					Ob = np.asarray(result_table['MAIN_ID'])[0].decode("utf-8")
					objtype = 'Simbad'
			except (AttributeError, ExpatError, TableParseError, ValueError, 
				EOFError, IndexError, ConnectionError,ReadTimeout,ChunkedEncodingError) as e:
				#print('Simbad fail event')
				pass
		Objects.append(Ob)
		Objtype.append(objtype)

	return Objects, Objtype


def Database_check_mask(Datacube, Dist, Masks, WCS):
	"""
	Checks Ned and Simbad to find the object name and type in the mask.
	This uses the mask set created by Identify_masks.
	"""
	Objects = []
	Objtype = []
	av = np.nanmedian(Datacube[Dist <= 0.2],axis = 0)
	for I in range(len(Masks)):

		Mid = np.where(av*Masks[I] == np.nanmax(av*Masks[I]))
		if len(Mid[0]) == 1:
			Coord = pix2coord(Mid[1], Mid[0], WCS)
		elif len(Mid[0]) > 1:
			Coord = pix2coord(Mid[1][0], Mid[0][0], WCS)

		c = coordinates.SkyCoord(
			ra=Coord[0], dec=Coord[1], unit=(u.deg, u.deg), frame='icrs')
		Ob = 'Unknown'
		objtype = 'Unknown'
		try:
			result_table = Ned.query_region(
				c, radius=6*u.arcsec, equinox='J2000')
			Ob = np.asarray(result_table['Object Name'])[0].decode("utf-8")
			objtype = result_table['Type'][0].decode("utf-8")

			if '*' in objtype:
				objtype = objtype.replace('*', 'Star')
			if '!' in objtype:
				objtype = objtype.replace('!', 'Gal')  # Galactic sources
			if objtype == 'G':
				try:
					result_table = Simbad.query_region(c, radius=6*u.arcsec)
					if len(result_table.colnames) > 0:
						objtype = objtype + 'Simbad'
				except (AttributeError, ExpatError, TableParseError, ValueError, 
					EOFError, IndexError, ConnectionError,ReadTimeout,ChunkedEncodingError) as e:
					#print('Simbad fail mask')
					pass

		except (RemoteServiceError, ExpatError, TableParseError, ValueError, 
			EOFError, IndexError, ConnectionError,ReadTimeout,ChunkedEncodingError) as e:
			#print('Ned fail mask')
			try:
				result_table = Simbad.query_region(c, radius=6*u.arcsec)
				if len(result_table.colnames) > 0:
					Ob = np.asarray(result_table['MAIN_ID'])[0].decode("utf-8")
					objtype = 'Simbad'

			except (AttributeError,ExpatError,TableParseError,ValueError,
				EOFError,IndexError,ConnectionError,ReadTimeout,ChunkedEncodingError) as e:
				#print('Simbad fail mask')
				pass
		Objects.append(Ob)
		Objtype.append(objtype)

	return Objects, Objtype



def Near_which_mask(Eventmask,Objmasks,Data):
	"""
	Finds which mask in the object mask an event is near. The value assigned to Near_mask 
	is the index of Objmask that corresponds to the event. If not mask is near, value is nan.
	"""
	Near_mask = np.ones(len(Eventmask),dtype=int)*-1
	mask = np.zeros((len(Eventmask),Data.shape[1],Data.shape[2]))
	for i in range(len(Eventmask)):
		mask[i,Eventmask[i][0],Eventmask[i][1]] = 1

	for i in range(len(Objmasks)):
		near_mask = ((convolve(Objmasks[i]*1,np.ones((3,3)),mode='constant', cval=0.0)) > 0)*1
		isnear = near_mask*mask
		Near_mask[np.where(isnear==1)[0]] = int(i)
	return Near_mask

def In_which_mask(Eventmask,Objmasks,Data):
	"""
	Finds which mask in the object mask an event is in. The value assigned to In_mask 
	is the index of Objmask that corresponds to the event. If not mask is near, value is -1.
	"""
	In_mask = np.ones(len(Eventmask),dtype=int)*-1
	mask = np.zeros((len(Eventmask),Data.shape[1],Data.shape[2]))
	for i in range(len(Eventmask)):
		mask[i,Eventmask[i][0],Eventmask[i][1]] = 1

	for i in range(len(Objmasks)):
		in_mask = (mask*Objmasks[i]).any()
		isin = in_mask*mask
		In_mask[np.where(isin==1)[0]] = int(i)
	return In_mask


def Save_space(Save):
	"""
	Creates a pathm if it doesn't already exist.
	"""
	try:
		if not os.path.exists(Save):
			os.makedirs(Save)
	except FileExistsError:
		pass

def Save_environment(Eventtime,maxcolor,Source,SourceType,Save):
	"""
	Creates paths to save event data, based on brightness and duration.
	"""
	if Eventtime[-1] - Eventtime[0] >= 48:
		if maxcolor <= 24:
			if ':' in Source:
				Cat = Source.split(':')[0]
				directory = Save+'/Figures/Long/Faint/' + Cat + '/' + SourceType.split(Cat + ': ')[-1] + '/'
			else:
				directory = Save+'/Figures/Long/Faint/' + SourceType + '/'

		else:
			if ':' in Source:
				Cat = Source.split(':')[0]
				directory = Save+'/Figures/Long/Bright/' + Cat + '/' + SourceType.split(Cat + ': ')[-1] + '/'
			else:
				directory = Save+'/Figures/Long/Bright/' + SourceType + '/'

	else:
		if maxcolor <= 24:
			if ':' in Source:
				Cat = Source.split(':')[0]
				directory = Save+'/Figures/Short/Faint/' + Cat + '/' + SourceType.split(Cat + ': ')[-1] + '/'
			else:
				directory = Save+'/Figures/Short/Faint/' + SourceType + '/'

		else:
			if ':' in Source:
				Cat = Source.split(':')[0]
				directory = Save+'/Figures/Short/Bright/' + Cat + '/' + SourceType.split(Cat + ': ')[-1] + '/'
			else:
				directory = Save+'/Figures/Short/Bright/' + SourceType + '/'
	Save_space(directory)
	return directory


def Types_masks(Events, Eventtime,Eventmask, Objmasks, ObjName, ObjType, Data, Dist, wcs):
	Source = []; SourceType = []; Maskobj = []
	if len(Events) > 0:
		Source, SourceType = Database_event_check(Data,Eventtime,Eventmask,wcs)

		Mask = ObjectMask(Data,Dist)

		Near = Near_which_mask(Eventmask,Objmasks,Data)
		In = In_which_mask(Eventmask,Objmasks,Data)
		
		Maskobj = np.zeros((len(Events),Data.shape[1],Data.shape[2])) # for plotting masked object reference
		if len(Objmasks) > 0:
			if len(np.where(Objmasks[:,int(Data.shape[1]/2),int(Data.shape[2]/2)] == 1)[0]) > 0:
				CentralMask = np.where(Objmasks[:,int(Data.shape[1]/2),int(Data.shape[2]/2)] == 1)[0]
			elif len(np.where(Objmasks[:,int(Data.shape[1]/2),int(Data.shape[2]/2)] == 1)[0]) > 1:
				CentralMask = np.where(Objmasks[:,int(Data.shape[1]/2),int(Data.shape[2]/2)] == 1)[0][0]
			else:
				CentralMask = -1
			if CentralMask == -1:
				Maskobj[:] = Mask
			else:
				Maskobj[:] = Objmasks[CentralMask]

			for ind in np.where(Near != -1)[0]:
				Source[ind] = 'Near: ' + ObjName[Near[ind]]
				SourceType[ind] = 'Near: ' + ObjType[Near[ind]]
				Maskobj[ind] = Objmasks[Near[ind]]
			for ind in np.where(In != -1)[0]:
				Source[ind] = 'In: ' + ObjName[In[ind]]
				SourceType[ind] = 'In: ' + ObjType[In[ind]]
				Maskobj[ind] = Objmasks[In[ind]]
		else:
			Maskobj[:] = Mask


		Source, SourceType = Probable_host(Eventtime,Eventmask,Source,SourceType,Objmasks,ObjName,ObjType,Data)
		
	return Source, SourceType, Maskobj


def Lightcurve(Data, Mask, Normalise = False):
	if type(Mask) == list:
		mask = np.zeros((Data.shape[1],Data.shape[2]))
		mask[Mask[0],Mask[1]] = 1
		Mask = mask*1.0
	else:
		Mask = Mask * 1.0
	Mask[Mask == 0.0] = np.nan
	LC = np.nansum(Data*Mask, axis = (1,2))
	for k in range(len(LC)):
		if np.isnan(Data[k]*Mask).all(): # np.isnan(np.sum(Data[k]*Mask)) & (np.nansum(Data[k]*Mask) == 0):
			LC[k] = np.nan
	LC[LC == 0] = np.nan
	if Normalise:
		LC = LC / np.nanmedian(LC)
	return LC

def Thumbnail(LC,BGLC,Eventtime,Time,Xlim,Ylim,Eventnum,File,Direct):
	fig = plt.figure()
	if Eventtime[-1] < len(Time):
		plt.axvspan(Time[Eventtime[0]]-Time[0],Time[Eventtime[-1]]-Time[0], color = 'orange')
	else:
		plt.axvspan(Time[Eventtime[0]]-Time[0],Time[-1]-Time[0], color = 'orange')
	plt.plot(Time - Time[0], LC,'.', label = 'Event LC')
	plt.plot(Time - Time[0], BGLC,'k.', label = 'Background LC')
	plt.xlim(Xlim[0],Xlim[1])
	plt.ylim(Ylim[0],Ylim[1])
	plt.tick_params(axis='x',which='both', labelbottom=False)	
	plt.tick_params(axis='y',which='both', labelleft=False)
	
	xfigsizeTN=1.5
	yfigsizeTN=1.5
	fig.set_size_inches(xfigsizeTN,yfigsizeTN)
	#Save_space(Save + '/Figures/Thumb/')
	#plt.savefig(Save + '/Figures/Thumb/'+ File.split('/')[-1].split('-')[0]+'_'+str(Eventnum)+'.png', bbox_inches = 'tight')  
	plt.savefig(Direct+ 'TN-' + File.split('/')[-1].split('-')[0]+'_'+str(Eventnum)+'.png', bbox_inches = 'tight')
	plt.close();

def Im_lims(dim,ev):
	"""
	Calculates the axis range for frames that have axis >20 pixels.
	"""
	if (ev - 9) < 0:
		xmin = 0
		xmax = ev + (20 - ev)

	elif (ev + 9) > dim:
		xmin = ev - (20 - (20 - ev))
		xmax = dim

	else:
		xmin = ev - 8
		xmax = ev + 8

	xlims = [xmin - 0.5, xmax - 0.5]
	return xlims


def Fig_cut(Datacube,Eventmask):
	"""
	Returns the figure limits for large frames that have an axis >20 pixels.
	"""
	x = Datacube.shape[1] - 0.5
	y = Datacube.shape[2] - 0.5

	if len(Eventmask[0]) == 1:
		if (x > 20) | (y > 20):

			if (x > 20) & (y <= 20):

				xlims  = Im_lims(x,Eventmask[0][0])

				ylims = [-0.5,y]
			elif (x <= 20) & (y > 20):
				ylims  = Im_lims(y,Eventmask[1][0])
				xlims = [-0.5,x]	   

			elif (x > 20) & (y > 20):
				xlims  = Im_lims(x,Eventmask[0][0])
				ylims  = Im_lims(y,Eventmask[1][0])
		else:
			xlims = [-0.5,x]
			ylims = [-0.5,y]
	else:
		xlims = [-0.5,x]
		ylims = [-0.5,y]
	return xlims, ylims


def Cutout(Data,Position):
	"""
	Limit the imshow dimensions to 20 square.
	Inputs:
	-------
	Data		- 3d array
	Position	- list

	Output:
	-------
	cutout_dims - 2x2 array 
	"""
	cutout_dims = np.array([[0, Data.shape[1]],[0, Data.shape[2]]])
	for i in range(2):
		if (Data.shape[i] > 19):
			dim0 = [Position[i][0] - 6, Position[i][0] + 6]

			bound = [(dim0[0] < 0), (dim0[1] > Data.shape[1])]

			if any(bound):
				if bound[0]:
					diff = abs(dim0[0])
					dim0[0] = 0
					dim0[1] += diff

				if bound[1]:
					diff = abs(dim0[1] - Data.shape[1])
					dim0[1] = Data.shape[1]
					dim0[0] -= diff

			cutout_dims[i,0] = dim0[0]
			cutout_dims[i,1] = dim0[1]
	return cutout_dims - 0.5

def K2TranPixFig(Events,Eventtime,Eventmask,Data,Time,
	Frames,wcs,Save,File,Quality,Thrusters,Datacube,
	Source,SourceType,ObjMask,Short=True):
	"""
	Makes the main K2:BS pipeline figure. Contains light curve with diagnostics, alongside event info.
	"""
	for i in range(len(Events)):
		mask = np.zeros((Data.shape[1],Data.shape[2]))
		mask[Eventmask[i][0],Eventmask[i][1]] = 1
		
		if np.isnan(Time[Eventtime[i][1]]):
			Eventtime[i][1] = Eventtime[i][1] -1
		
		#Find Coords of transient
		position = np.where(mask)
		if len(position[0]) == 0:
			print(Broken)
		Mid = ([position[0][0]],[position[1][0]])
		maxcolor = -1000 # Set a bad value for error identification
		for j in range(len(position[0])):
			lcpos = np.copy(Data[Eventtime[i][0]:Eventtime[i][-1],position[0][j],position[1][j]])
			nonanind = np.isfinite(lcpos)
			temp = sorted(lcpos[nonanind].flatten())
			temp = np.array(temp)
			if len(temp) > 10:
				temp  = temp[-3] # get 3rd brightest point
			#elif len(temp) > 0:
			#	temp  = temp[-1] # get 3rd brightest point
			else:
				temp = 0
			if temp > maxcolor:
				maxcolor = temp
				Mid = ([position[0][j]],[position[1][j]])

		if len(Mid[0]) == 1:
			Coord = pix2coord(Mid[1],Mid[0],wcs)
		elif len(Mid[0]) > 1:
			Coord = pix2coord(Mid[1][0],Mid[0][0],wcs)

		test = np.ma.masked_invalid(Data).mask*1
		wide = convolve(test,np.ones((1,3,3))) > 0
		bgmask = -(wide+mask) + 1
		bgmask[bgmask==0] = np.nan
		background = Data*bgmask
		level = np.nanmedian(background,axis=(1,2))
		BG = Data*~Frames[Events[i]]
		BG[BG <= 0] = np.nan
		BGLC = level
		# Generate a light curve from the transient masks
		LC = Lightcurve(Data, mask)

		Obj = ObjMask[i]
		ObjLC = Lightcurve(Datacube, Obj)
		ObjLC = ObjLC/np.nanmedian(ObjLC)*np.nanmedian(LC)

		OrigLC = Lightcurve(Datacube, mask)
		
		tt = Time - np.floor(Time[0])
		
		fig = plt.figure(figsize=(10,6))
		# set up subplot grid
		gridspec.GridSpec(2,3)
		plt.suptitle('EPIC ID: ' + File.split('ktwo')[-1].split('_')[0] + '\nSource: '+ Source[i] + ' (' + SourceType[i] + ')')
		# large subplot
		plt.subplot2grid((2,3), (0,0), colspan=2, rowspan=2)
		plt.title('Event light curve ('+str(round(Coord[0],3))+', '+str(round(Coord[1],3))+')')
		plt.xlabel('Time (+'+str(np.floor(Time[0]))+' BJD)')
		plt.ylabel('Counts')
		if Eventtime[i][1] > len(Time):
			Eventtime[i][1] = len(Time) - 1
		plt.axvspan(tt[Eventtime[i,0]],tt[Eventtime[i,1]], color = 'orange',alpha=0.5, label = 'Event duration')
		if (Eventtime[i][1] - Eventtime[i][0]) < 48:
			plt.axvline(tt[Quality[0]],color = 'red', linestyle='dashed',label = 'Quality', alpha = 0.5)
			for j in range(Quality.shape[0]-1):
				j = j+1 
				plt.axvline(tt[Quality[j]], linestyle='dashed', color = 'red', alpha = 0.5, rasterized=True)
			# plot Thurster firings 
			plt.axvline(tt[Thrusters[0]],color = 'red',label = 'Thruster', alpha = 0.5, rasterized=True)
			for j in range(Thrusters.shape[0]-1):
				j = j+1 
				plt.axvline(tt[Thrusters[j]],color = 'red', alpha = 0.5, rasterized=True)
			
		

		plt.plot(tt, BGLC,'k.', alpha = 0.5, label = 'Background LC', rasterized=True)
		plt.plot(tt, ObjLC,'kx', alpha = 0.5, label = 'Scaled object LC', rasterized=True)
		plt.plot(tt, OrigLC,'m+', alpha = 0.5, label = 'Original data', rasterized=True)
		plt.plot(tt, LC,'.', label = 'Event LC',alpha=0.7, rasterized=True)
		if not Short:
			Six_LC, ind = SixMedian(LC)
			plt.plot(tt[ind], Six_LC,'m.', label = '6hr average',alpha=1, rasterized=True)
		
		width = Eventtime[i,1]-Eventtime[i,0]
		
		temp1 = Eventtime[i][0] - int(2*width)
		
		if temp1 < 0:
			temp1 = 0
		temp2 = Eventtime[i][1] + int(4*width)

		if temp2 > len(tt)-1:
			temp2 = len(tt)-1
		xmin = tt[temp1]
		xmax = tt[temp2]
		
		if np.isfinite(xmin) & np.isfinite(xmax):
			plt.xlim(xmin,xmax) 
		else:
			print('min',xmin)
			print('max',xmax)
			print('why limits broken in ',File)
		
		if width < 50:
			limit_lc = Savgol(LC,width,10)
		else:
			limit_lc = Savgol(LC,48,10)

		ind1 = np.where(np.nanmin(abs(xmin - tt)) == 
								  abs(xmin - tt))[0][0]
		ind2 = np.where(np.nanmin(abs(xmax - tt)) == 
								  abs(xmax - tt))[0][0]
		
		ymin = np.nanmin(limit_lc[ind1:ind2])				
		ymax = np.nanmax(limit_lc[ind1:ind2])
		
		ymin -= 0.1*ymin
		ymax += 0.1*ymax

		if np.isfinite(ymin) & np.isfinite(ymax):
			plt.ylim(ymin,ymax)
		plt.legend()#loc = 1)
		plt.minorticks_on()
		plt.tick_params(axis='both',which='both',direction='in')
		ylims, xlims = Fig_cut(Datacube,Mid)

		# small subplot 1 Reference image plot
		ax = plt.subplot2grid((2,3), (0,2))
		plt.title('Reference')
		plt.imshow(np.nanmedian(Data,axis=0), origin='lower',vmin=0,vmax = maxcolor)
		plt.xlim(xlims[0],xlims[1])
		plt.ylim(ylims[0],ylims[1])
		current_cmap = plt.cm.get_cmap()
		current_cmap.set_bad(color='black')
		plt.colorbar(fraction=0.046, pad=0.04)
		plt.plot(position[1],position[0],'r.',ms = 15, rasterized=True)
		plt.minorticks_on()
		ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
		ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
		plt.tick_params(axis='both',which='both',direction='in')
		# small subplot 2 Image of event
		ax = plt.subplot2grid((2,3), (1,2))
		plt.title('Event')
		plt.imshow(Data[np.where(Data*mask==np.nanmax(Data[Eventtime[i][0]:Eventtime[i][-1]]*mask))[0][0],:,:], origin='lower',vmin=0,vmax = maxcolor)
		plt.xlim(xlims[0],xlims[1])
		plt.ylim(ylims[0],ylims[1])
		current_cmap = plt.cm.get_cmap()
		current_cmap.set_bad(color='black')
		plt.colorbar(fraction=0.046, pad=0.04)
		plt.plot(position[1],position[0],'r.',ms = 12, rasterized=True)
		plt.minorticks_on()
		ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
		ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
		plt.tick_params(axis='both',which='both',direction='in')
		
		if Short:
			directory = Save_environment(Eventtime[i],maxcolor,Source[i],SourceType[i],Save)
			plt.savefig(directory+File.split('/')[-1].split('-')[0]+'_'+str(i)+'.pdf', bbox_inches = 'tight')
		else:
			directory = Long_save_environment(maxcolor,Source[i],SourceType[i],Save)
			plt.savefig(directory+File.split('/')[-1].split('-')[0]+'_L'+str(i)+'.pdf', bbox_inches = 'tight')

		plt.close()
		Thumbnail(LC,BGLC,Eventtime[i],Time,[xmin,xmax],[ymin,ymax],i,File,directory);
	return

def K2TranPixZoo(Events,Eventtime,Eventmask,Source,SourceType,Data,Time,wcs,Save,File,Short=True):
	"""
	Iteratively gmakes Zooniverse videos for events. Videos are made from frames which are saved in a corresponding Frame directory.
	"""
	saves = []
	for i in range(len(Events)):
		mask = np.zeros((Data.shape[1],Data.shape[2]))
		mask[Eventmask[i][0],Eventmask[i][1]] = 1
		position = np.where(mask)
		Mid = ([position[0][0]],[position[1][0]])
		maxcolor = 0 # Set a bad value for error identification
		for j in range(len(position[0])):
			lcpos = np.copy(Data[Eventtime[i][0]:Eventtime[i][-1],position[0][j],position[1][j]])
			nonanind = np.isfinite(lcpos)
			temp = sorted(lcpos[nonanind].flatten())
			temp = np.array(temp)
			if len(temp) > 10:
				temp  = temp[-3] # get 3rd brightest point
			else:
				temp  = 0#temp[-1] # get 3rd brightest point

			if temp > maxcolor:
				maxcolor = temp
				Mid = ([position[0][j]],[position[1][j]])

		width = Eventtime[i,1]-Eventtime[i,0]
		tt = Time - np.floor(Time[0])
		
		LC = Lightcurve(Data, mask)
		
		if width < 50:
			limit_lc = Savgol(LC,width,10)
		else:
			limit_lc = Savgol(LC,48,10)
		
		if Short:
			xmin = Eventtime[i][0] - int(2*width)
			xmax = Eventtime[i][1] + int(2*width)

			if xmin < 0:
				xmin = 0
			if xmax > len(Time) - 1:
				xmax = len(Time) - 1 
			if ~np.isfinite(xmin):
				xmin = 0
			if ~np.isfinite(xmax):
				xmax = len(Time) - 1 
			
			step = int((xmax - xmin)*.05) # Make a step so that only 5% of the frames are produced 
			if step <= 0:
				step = 1
		else:
			xmin = 0
			xmax = len(Data) -1
			step = int((xmax - xmin)*.1)

		Section = np.arange(int(xmin),int(xmax),step)
		nanframe_ind = np.where(np.nansum(Data,axis=(1,2)) == 0)[0]
		for s in range(len(Section)):
			while (Section[s] in nanframe_ind) & (Section[s] < len(Data)-1):
				Section[s] += 1

		if Short:
			FrameSave = Save + '/Figures/Frames/' + File.split('/')[-1].split('-')[0] + '/Event_' + str(int(i)) + '/'
		else:
			FrameSave = Save + '/Figures/Frames/' + File.split('/')[-1].split('-')[0] + '/Event_L' + str(int(i)) + '/'

		Save_space(FrameSave)

		ylims, xlims = Fig_cut(Data,Mid)

		ymin = np.nanmin(limit_lc[xmin:xmax])				
		ymax = np.nanmax(limit_lc[xmin:xmax])
						
		ymin -= 0.1*ymin
		ymax += 0.1*ymax
		# Create an ImageNormalize object using a SqrtStretch object
		norm = ImageNormalize(vmin=ymin/len(position[0]), vmax=maxcolor, stretch=SqrtStretch())

		height = 1100/2
		width = 2200/2
		my_dpi = 100
		for j in range(len(Section)):
			filename = FrameSave + 'Frame_' + str(int(j)).zfill(4)+".png"
			
			fig = plt.figure(figsize=(width/my_dpi,height/my_dpi),dpi=my_dpi)
			plt.subplot(1, 2, 1)
			plt.title('Event light curve')
			plt.axvspan(tt[Eventtime[i,0]],tt[Eventtime[i,1]],color='orange',alpha = 0.5)
			plt.plot(tt, LC,'k.')
			
			
			plt.ylim(ymin,ymax)
			plt.xlim(tt[xmin],tt[xmax])

			plt.ylabel('Counts')
			plt.xlabel('Time (days)')
			plt.axvline(tt[Section[j]],color='red',lw=2)
			plt.tick_params(axis='both',which='both',direction='in')

			plt.subplot(1,2,2)
			plt.title('Kepler image')
			Data[np.isnan(Data)] = 0
			plt.imshow(Data[Section[j]],origin='lower',cmap='gray',norm=norm)
			current_cmap = plt.cm.get_cmap()
			current_cmap.set_bad(color='black')
			#plt.colorbar()
			ylims, xlims = Fig_cut(Data,Mid)
			plt.xlim(xlims[0],xlims[1])
			plt.ylim(ylims[0],ylims[1])
			plt.ylabel('Row')
			plt.xlabel('Column')
			plt.plot(position[1],position[0],'r.',ms = 15)
			fig.tight_layout()
			
			ax = fig.gca()
			ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
			ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
			plt.tick_params(axis='both',which='both',direction='in')

			plt.savefig(filename,dpi=100)
			plt.close();

		framerate = (len(Section))/5
		if Short:
			directory = Save_environment(Eventtime[i],maxcolor,Source[i],SourceType[i],Save)
			ffmpegcall = ('ffmpeg -y -nostats -loglevel 8 -f image2 -framerate ' + str(framerate) 
						  + ' -i ' + FrameSave + 'Frame_%04d.png -vcodec libx264 -pix_fmt yuv420p ' 
						  + directory + 'Zoo-' + File.split('/')[-1].split('-')[0] + '_' + str(i) + '.mp4')
		else:
			directory = Long_save_environment(maxcolor,Source[i],SourceType[i],Save)
			ffmpegcall = ('ffmpeg -y -nostats -loglevel 8 -f image2 -framerate ' + str(framerate) 
						  + ' -i ' + FrameSave + 'Frame_%04d.png -vcodec libx264 -pix_fmt yuv420p ' 
						  + directory + 'Zoo-' + File.split('/')[-1].split('-')[0] + '_L' + str(i) + '.mp4')

		

		
		os.system(ffmpegcall);

		saves.append('./Figures' + directory.split('Figures')[-1] + 'Zoo-' + File.split('/')[-1].split('-')[0] + '_' + str(i) + '.mp4')


	return saves

def Write_event(Pixelfile, Eventtime, Eventmask, Source, Sourcetype, Zoo_Save, Data, Quality, WCS, hdu, Path):
	"""
	Saves the event and field properties to a csv file.
	"""
	feild = Pixelfile.split('-')[1].split('_')[0]
	ID = Pixelfile.split('ktwo')[1].split('-')[0]
	try:
		rank_brightness = Rank_brightness(Eventtime,Eventmask,Data,Quality)
	except IndexError:
		error_str = 'event lengths broken in {}'.format(Pixelfile)
		raise IndexError(error_str)
	rank_duration = Rank_duration(Eventtime)
	rank_mask = Rank_mask(Eventmask,Data)
	rank_host = Rank_host(Sourcetype)

	rank_total = rank_brightness + rank_duration + rank_mask + rank_host 
	for i in range(len(Eventtime)):
		mask = np.zeros((Data.shape[1],Data.shape[2]))
		mask[Eventmask[i][0],Eventmask[i][1]] = 1
		start = Eventtime[i][0]
		duration = Eventtime[i][1] - Eventtime[i][0]
		maxlc = np.nanmax(Lightcurve(Data[Eventtime[i][0]:Eventtime[i][-1]], mask))

		#Find Coords of transient
		position = np.where(mask)
		if len(position[0]) == 0:
			print(Broken)
		Mid = ([position[0][0]],[position[1][0]])
		maxcolor = -1000 # Set a bad value for error identification
		for j in range(len(position[0])):
			lcpos = np.copy(Data[Eventtime[i][0]:Eventtime[i][-1],position[0][j],position[1][j]])
			lcpos[np.isnan(lcpos)] = 0.0
			temp = sorted(lcpos.flatten())
			temp = np.array(temp)
			if len(temp) > 10:
				temp  = temp[-3] # get 3rd brightest point
			else:
				temp  = temp[-1] # get 3rd brightest point
			if temp > maxcolor:
				maxcolor = temp
				Mid = ([position[0][j]],[position[1][j]])

		if len(Mid[0]) == 1:
			Coord = pix2coord(Mid[1],Mid[0],WCS)
		elif len(Mid[0]) > 1:
			Coord = pix2coord(Mid[1][0],Mid[0][0],WCS)


		size = np.nansum(Eventmask[i])
		Zoo_fig = Zoo_Save[i]
		CVSstring = [str(feild), str(ID), str(i), Sourcetype[i], str(start), 
					 str(duration), str(maxlc), str(size), str(Coord[0]), str(Coord[1]), 
					 Source[i], str(hdu[0].header['CHANNEL']), str(hdu[0].header['MODULE']), 
					 str(hdu[0].header['OUTPUT']), str(rank_brightness[i]), str(rank_duration[i]),
					 rank_mask[i],rank_host[i], rank_total[i], Zoo_fig]				
		if os.path.isfile(Path + '/Events.csv'):
			with open(Path + '/Events.csv', 'a') as csvfile:
				spamwriter = csv.writer(csvfile, delimiter=',')
				spamwriter.writerow(CVSstring)
		else:
			with open(Path + '/Events.csv', 'w') as csvfile:
				spamwriter = csv.writer(csvfile, delimiter=',')
				spamwriter.writerow(['Field', 'EPIC', '#Event number', '!Host type', '#Start', 
									 'Duration', 'Counts', '#Size','#RA','#DEC',
									 '#Host', '#Channel', '#Module', 
									 '#Output', '#Rank brightness', '#Rank duration',
									 '#Rank mask', '#Rank host','#Rank total', '#Zoofig'])
				spamwriter.writerow(CVSstring)
	return
			
def Probable_host(Eventtime,Eventmask,Source,SourceType,Objmasks,ObjName,ObjType,Data):
	"""
	Identifies if the event is likely associated with a bright science target. 
	This is calculated based on if neighbouring pixels feature the same or greater brightness.
	"""
	for i in range(len(Eventtime)):
		if 'Near' not in SourceType[i]:
			mask = np.zeros((Data.shape[1],Data.shape[2]))
			mask[Eventmask[i][0],Eventmask[i][1]] = 1
			maxlc = np.nanmax(np.nansum(Data[Eventtime[i][0]:Eventtime[i][-1]]*(mask==1),axis=(1,2)))
			maxcolor = np.nanmax(Data[Eventtime[i][0]:Eventtime[i][-1]]*(mask==1))
			maxframe = Data[np.where(Data*mask==np.nanmax(Data[Eventtime[i][0]:Eventtime[i][-1]]*mask))[0][0]]	   
			conv = (convolve(mask,np.ones((3,3)),mode='constant', cval=0.0) > 0) - mask
			if len(np.where(maxframe*conv >= maxcolor)[0]) > 1:
				Mid = np.where(Data[Eventtime[i][0]:Eventtime[i][-1]]*(mask==1) == maxcolor)
				if len(Objmasks) > 0:
					if len(Mid[0]) == 1:
						distance = np.sqrt((np.where(Objmasks==1)[1] - Mid[1])**2 + (np.where(Objmasks==1)[2] - Mid[2])**2)
					elif len(Mid[0]) > 1:
						distance = np.sqrt((np.where(Objmasks==1)[1] - Mid[1][0])**2 + (np.where(Objmasks==1)[2] - Mid[2][0])**2)
					try:
						minind = np.where((np.nanmin(distance) == distance))[0][0]
						minind = np.where(Objmasks==1)[0][minind]
						SourceType[i] = 'Prob: ' + ObjType[minind]
						Source[i] = 'Prob: ' + ObjName[minind]
					except ValueError:
						pass
	return Source, SourceType

def SixMedian(LC):
	'''
	Creates a lightcurve using a 6 hour median average. 
	Returns:
	lc6 - the 6 hour averaged light curve
	x   - the time indecies of the light curve positions to index the time array on.
	'''
	lc6 = []
	x = []
	for i in range(int(len(LC)/12)):
		if np.isnan(LC[i*12:(i*12)+12]).all():
			lc6.append(np.nan)
			x.append(i*12+6)
		else:
			lc6.append(np.nanmedian(LC[i*12:(i*12)+12]))
			x.append(i*12+6)
	lc6 = np.array(lc6)
	x = np.array(x)
	return lc6, x


def Savgol(Data,Width,Sigma=3):
	lc = Data.copy()
	lc[Median_clip(lc,Sigma)] =np.nan
	x = np.arange(0,len(lc))
	ind = np.isfinite(lc)
	fun = interp1d(x[ind],lc[ind],bounds_error = False)
	nonan = fun(x)
	if Width/2 == int(Width/2):
		Width += 1
	sm = savgol_filter(nonan,Width,2,mode='nearest')
	return sm

def Long_smooth_limit(Data,Dist):
	sm_dist = Savgol(Dist,5*48)
	dist_ind = np.where(sm_dist > 0.4)[0]

	X,Y = np.where(np.isfinite(np.nansum(Data,axis=0)))

	smoothed = np.zeros_like(Data)
	limit = np.zeros_like(Data[0]) * np.nan
	for i in range(len(X)):
		lc = Data[:,X[i],Y[i]].copy()
		lc[Dist > .3] = np.nan
		ind = np.isfinite(lc)
		if len(lc[ind]) > 10:
			width = 5*48 - 1
			sm = Savgol(lc,width)
			smoothed[:,X[i],Y[i]] = sm

			pea = find_peaks(sm,distance=4*width)[0]
			maxpeak = pea[np.where(sm[pea] == np.nanmax(sm[pea]))[0]]
			totalmax = np.where(np.nanmax(sm) == sm)[0]
			
			ind = np.arange(0,len(smoothed))
			m_s = ind < int(maxpeak-2*width)
			m_e = ind > int(maxpeak+4*width)
			mean_start = np.nanmean(sm[m_s]) 
			mean_end   = np.nanmean(sm[m_e]) 

			std_start = np.nanstd(sm[m_s])
			std_end = np.nanstd(sm[m_e])
			if (maxpeak not in dist_ind) & (totalmax not in dist_ind):
				if np.isfinite(mean_start) & np.isfinite(mean_end):
					if mean_end > (mean_start + 3*std_start):
						
						mean_outside = mean_start
						std_outside = std_start
					else:
						mean_outside = np.nanmean(sm[m_s+m_e])
						std_outside = np.nanstd(sm[m_s+m_e])
				else:
					mean_outside = np.nanmean(sm[m_s+m_e])
					std_outside = np.nanstd(sm[m_s+m_e])
				limit[X[i],Y[i]] = mean_outside +3*std_outside
			else:
				limit[X[i],Y[i]] = np.nan#sm[maxpeak]
		else:
			smoothed[:,X[i],Y[i]] = np.nan
	#limit[limit<22] = 22
	return smoothed, limit

def Vet_long(Events, Eventtime, Eventmask, Sig, Data):
	good_ind = []
	for i in range(len(Events)):
		eh = Sig[Eventtime[i,0]:Eventtime[i,1],Eventmask[i][0],Eventmask[i][1]]
		r2 = Regress_fit(eh, Fit = False)
		
		totalmax = np.where(np.nanmax(eh) == eh)[0]
		lc = Data[:,Eventmask[i][0],Eventmask[i][1]].copy()
		lc = Savgol(lc,49)
		allmax = np.where(np.nanmax(lc) == lc)[0]
		max_in_event = (allmax > Eventtime[i,0]) & (allmax < Eventtime[i,1])
		if (r2 > 0.95) & (totalmax < len(eh) - 10).all() & max_in_event.all():
			good_ind += [i]
	good_ind = np.array(good_ind)
	
	if len(good_ind) > 0:
		Events = Events[good_ind]
		Eventtime = Eventtime[good_ind]
		Eventmask = np.array(Eventmask)[good_ind].tolist()
	else:
		Events = np.array([])
		Eventtime = np.array([])
		Eventmask = []
	return Events, Eventtime, Eventmask


def Long_save_environment(maxcolor,Source,SourceType,Save):
	'''
	Makes a save pathway for the long event search. All events will be saved under a VLong directory.
	'''
	if maxcolor <= 24:
		if ':' in Source:
			Cat = Source.split(':')[0]
			directory = Save+'/Figures/VLong/Faint/' + Cat + '/' + SourceType.split(Cat + ': ')[-1] + '/'
		else:
			directory = Save+'/Figures/VLong/Faint/' + SourceType + '/'

	else:
		if ':' in Source:
			Cat = Source.split(':')[0]
			directory = Save+'/Figures/VLong/Bright/' + Cat + '/' + SourceType.split(Cat + ': ')[-1] + '/'
		else:
			directory = Save+'/Figures/VLong/Bright/' + SourceType + '/'

	Save_space(directory)
	return directory


def Kill_bright(Data, Limit):
	temp = Data > 100000
	kernal = np.ones((1,3,3))
	conv = convolve(temp,kernal,mode='constant', cval=0.0) > 0
	inds = np.where(np.nansum(conv,axis=0) > 0)
	Limit[inds] = np.nan
	return Limit


	
def Find_short_events(Data, Time, Dist, File, Save, Objmasks, ObjName, 
					  ObjType, wcs, Orig, Quality, Thrusters, hdu,xdrift,ydrift):
	
	framemask = np.zeros_like(Data)

	med = np.nanmedian(Data[Quality == 0], axis = (0))
	med[med < 0] = 0

	limit = med+3*(np.nanstd(Data[Quality == 0], axis = (0)))
	#limit[limit<22] = 22
	limit = Kill_bright(Data, limit) 
	
	Limitsave = Save + '/Limit/' + File.split('ktwo')[-1].split('-')[0]+'_Limit'
	Save_space(Save + '/Limit/')
	np.save(Limitsave,limit)
	
	
	framemask = (Data/limit)

	# Identify if there is a sequence of consecutive or near consecutive frames that meet condtition 
	Eventmask = np.copy(framemask)
	Eventmask[Quality!=0,:,:] = 0

	events, eventtime, eventmask = Event_ID(Eventmask, 1, 5)
	events, eventtime, eventmask = Match_events(events,eventtime,eventmask,Data)
	events, eventtime, eventmask = Vet_peaks(events, eventtime, eventmask, Data)
	
	for i in range(len(events)):
		mask = np.zeros((Data.shape[1],Data.shape[2]))
		mask[eventmask[i][0],eventmask[i][1]] = 1
		section = Data[eventtime[i][0]:eventtime[i][-1]] * mask
		t,x,y = np.where(np.nanmax(section) == section)
		name = File.split('ktwo')[-1].split('-')[0]+ '_' + str(i)
		Track_Asteroid(x,y,t,Data,Time,xdrift,ydrift,wcs,Save,name)
	
	return print(File, '# of short events: ', len(events))


def Find_long_events(Data, Time, Dist, File, Save, Objmasks, ObjName, 
				ObjType, wcs, Orig, Quality, Thrusters, hdu):
	smoothed, limit = Long_smooth_limit(Data.copy(),Dist.copy())
	limit = Kill_bright(Data, limit) 

	Limitsave = Save + '/Limit/' + File.split('ktwo')[-1].split('-')[0]+'_VLimit'
	Save_space(Save + '/Limit/')
	np.save(Limitsave,limit)
	
	framemask = (smoothed/limit)
	events = []
	eventtime = []
	eventmask = []
	events, eventtime, eventmask = Event_ID(framemask, 1, 10*48,Smoothing=False)
	events, eventtime, eventmask = Vet_long(events, eventtime, eventmask, framemask, Data)
	events, eventtime, eventmask = Match_events(events,eventtime,eventmask,Data.copy())
	
	print(len(events))
	Source, SourceType, Maskobj = Types_masks(events, eventtime, eventmask, Objmasks, 
											  ObjName, ObjType, Data, Dist, wcs)
	
	quality = np.where(Quality != 0)[0]
	if len(events) > 0:
		K2TranPixFig(events.copy(),eventtime.copy(),eventmask.copy(),Data,Time,
					(framemask.copy() >= 0),wcs,Save,File,quality,Thrusters,Orig,
					Source,SourceType,Maskobj, Short=False)
		
		Zoo_saves = K2TranPixZoo(events.copy(),eventtime.copy(),eventmask.copy(),Source.copy(),
									SourceType.copy(),Data.copy(),Time,wcs,Save,File,Short=False)
		
		Write_event(File,eventtime.copy(),eventmask.copy(),Source,
					SourceType,Zoo_saves,Data.copy(),Quality,wcs,hdu,Save)
	
	return print(File, '# of long events: ', len(events))


def Rank_brightness(Eventtime,Eventmask,Data,Quality):
	Rank = np.zeros(len(Eventtime))
	for i in range(len(Eventtime)):

		mask = np.zeros((Data.shape[1],Data.shape[2]))
		mask[Eventmask[i][0],Eventmask[i][1]] = 1
		mask = mask > 0
		
		LC = Lightcurve(Data,mask)
		outside_mask = np.ones(len(LC))
		lower = Eventtime[i][0] - 2*48
		upper = Eventtime[i][1] + 10*48
		if lower < 0:
			lower = 0
		if upper > len(LC):
			upper = -1

		outside_mask[lower:upper] = 0
		outside_mask = outside_mask > 0

		median = np.nanmedian(LC[outside_mask])
		std = np.nanstd(LC[outside_mask])
		event_max = Smoothmax(Eventtime[i],LC,Quality)
		if type(event_max) == int:
			Rank[i] = np.round((LC[event_max]-median)/std,1)
			
		elif len(event_max) > 0:
			Rank[i] = np.round((LC[event_max[0]]-median)/std,1)
			
		else:
			Rank[i] = 0
	
	return Rank

def Rank_duration(Eventtime):
	Rank = np.zeros(len(Eventtime))
	Rank = np.round((Eventtime[:,1] - Eventtime[:,0])/48,1)
	
	return Rank

def Rank_mask(Eventmask,Data):
	Rank = np.zeros(len(Eventmask))
	for i in range(len(Eventmask)):
		mask = np.zeros((Data.shape[1],Data.shape[2]))
		mask[Eventmask[i][0],Eventmask[i][1]] = 1
		mask = mask > 0
		Rank[i] = np.round(np.nansum(mask)/3,1)
	Rank[Rank > 1] = 1
	return Rank

import pandas as pd

def Rank_host(Type):
	rank_system = pd.read_csv('Type_weights.csv').values
	Rank = np.zeros(len(Type))
	for i in range(len(Type)):
		for j in range(len(rank_system)):
			if rank_system[j,0] in Type[i]:
				Rank[i] = int(rank_system[j,1])
	return Rank

def Vet_peaks(Events, Eventtime, Eventmask, Data, Smoothdata = True):
	good_ind = []
	lcs = []
	peaks = []
	for i in Eventmask:
		lcs += [Lightcurve(Data,i)]
	lcs = np.array(lcs)
	
	for e in range(len(lcs)):
		x = np.arange(0,len(lcs[e]))
		ind = np.isfinite(lcs[e])
		fun = interp1d(x[ind],lcs[e][ind],bounds_error = False,fill_value='extrapolate')
		nonan = fun(x)
		width = ((Eventtime[e,1] - Eventtime[e,0]) * 3) 
		if (width/2) == int(width/2):
			width -= 1
		if Smoothdata:
			sm = savgol_filter(nonan,width,2,mode='nearest')
		else:
			sm = nonan
		pea = find_peaks(sm)[0]
		
		ev_ind = np.where((pea >= Eventtime[e,0]) & (pea <= Eventtime[e,1]))[0]
		max_ind = np.where(np.nanmax(sm[pea])== sm[pea])[0]
		
		if (ev_ind == max_ind).any():
			stat_ind = pea[pea!=pea[max_ind]]  
			std = np.nanstd(sm[stat_ind])
			med = np.nanmedian(sm[stat_ind])
			sig = (sm[pea[max_ind]] - med) / std
			if sig >= 5:
				if (width * 10) > 48 * 10:
					check_range = width * 10
				else:
					check_range = 48 * 10
				tstart = Eventtime[e,0] - check_range
				if tstart < 0:
					tstart = 0 
				tend = Eventtime[e,1] + check_range
				if tend >= len(Data):
					tend = len(Data) - 1
				near_peaks = pea[(pea < Eventtime[e,0]) & (pea >= tstart)
								 | (pea > Eventtime[e,1]) & (pea <= tend)]
				if ((sm[near_peaks] / sm[pea[max_ind]]) < 0.7).all(): 
					good_ind += [e]
	good_ind = np.array(good_ind)
	
	if len(good_ind) > 0:
		Events = Events[good_ind]
		Eventtime = Eventtime[good_ind]
		Eventmask = np.array(Eventmask)[good_ind].tolist()
	else:
		Events = np.array([])
		Eventtime = np.array([])
		Eventmask = []
	return Events, Eventtime, Eventmask


# Testing this function 




def K2TranPix(pixelfile,save): 
	"""
	Main code yo. Runs an assortment of functions to detect events in Kepler TPFs.
	"""
	Save = save + pixelfile.split('-')[1].split('_')[0]
	try:
		hdu = fits.open(pixelfile)
	except OSError:
		print('OSError ',pixelfile)
		return

	if len(hdu) > 1:
		dat = hdu[1].data
	else:
		print('Broken file ', pixelfile)
		return
	datacube = fits.ImageHDU(hdu[1].data.field('FLUX')[:]).data
	if datacube.shape[1] > 1 and datacube.shape[2] > 1:
		datacube = Clip_cube(datacube)
		time = dat["TIME"] + 2454833.0
		nonanind = np.where(np.isfinite(time))[0]
		Qual = hdu[1].data.field('QUALITY')
		
		time = time[nonanind]
		datacube = datacube[nonanind,:,:]
		Qual = Qual[nonanind]
		
		thrusters = np.where((Qual & 1048576) > 0)[0]#Get_all_resets(datacube, Qual)
			
		xdrif = dat['pos_corr1']
		ydrif = dat['pos_corr2']	
		distdrif = np.sqrt(xdrif**2 + ydrif**2)
		distdrif = distdrif[nonanind]

		if len(distdrif) != len(datacube):
			err_string = 'Distance arr is too short for {file}: len = {leng}'.format(file = pixelfile, leng = len(distdrif))
			raise ValueError(err_string) 
		
		
		Maskdata = Correct_motion(datacube, distdrif, thrusters)
		
		# Save asteroids
		#astsave = Save + '/Asteroid/' + pixelfile.split('ktwo')[-1].split('-')[0]+'_Asteroid'
		#Save_space(Save + '/Asteroid/')
		#np.savez(astsave,ast)
		
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


		# Find all spatially seperate objects in the event mask.
		Mask = ObjectMask(Maskdata,distdrif)
		obj = np.ma.masked_invalid(Mask).mask
		Objmasks = Watershed_object_sep(obj)
		if len(Objmasks.shape) < 3:
			Objmasks = np.zeros((1,datacube.shape[1],datacube.shape[2]))
		ObjName, ObjType = Database_check_mask(datacube,distdrif,Objmasks,mywcs)
		
		Find_short_events(Maskdata, time, distdrif, pixelfile, Save, Objmasks, ObjName, 
						 ObjType, mywcs, datacube, Qual, thrusters, hdu,xdrif,ydrif)		

	else:
		print('Small ', pixelfile)

