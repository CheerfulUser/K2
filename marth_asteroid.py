import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage.filters import convolve
import time
from scipy import signal
from scipy.signal import find_peaks, peak_prominences
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from scipy import ndimage as ndi
import pandas as pd
from tqdm import tqdm
from astropy.utils.data import download_file
import matplotlib.gridspec as gridspec
from astropy.io import fits
import os




def Initial(x,y,t,flux):	
	'''
	Given an initial position and time, returns the pixel at that time which is most likely to be the centre of the asteroid by correlating the derivatives of the neighbouring pixels and selecting the pixel with the largest spike	
	'''
	print('x',x)
	print('y',y)
	print('t',t)
	pixcurve = flux[int(t-30):int(t+30),int(x),int(y)]
	pixcurve = pixcurve/np.nanmedian(pixcurve)
	pcdiff = np.diff(pixcurve)
	control = signal.correlate(pcdiff,pcdiff)	
	centre = np.where(control == np.max(control))

	maxdif1 = []
	corrarr = []
	for i in range(int(x-1),int(x+1)):
		for j in range(int(y-1),int(y+1)):
			normflux = flux[int(t-30):int(t+30),i,j]/np.median(flux[int(t-30):int(t+30),i,j])
			corr = signal.correlate(pcdiff,np.diff(normflux))
			corrarr += [(i,j,np.max(corr))]
			cormax = np.where(corr == np.max(corr))
			tshift = centre[0] - cormax[0]
			if tshift == 0:
				maxdif1 += [np.max(corr)]

	corrarr = np.array(corrarr)			
	ind = np.argmax(maxdif1)
	ind1 = np.where(maxdif1[ind] == corrarr[:,2])			

	pos1 = int(corrarr[ind1,0])
	pos2 = int(corrarr[ind1,1])
	t1 = np.argmax(flux[int(t-3):int(t+4),pos1,pos2]) + t - 3

	return pos1, pos2, t1


def asteroid(pos1, pos2, time, flux):
	'''
	Given a position and time where the asteroid occurs, finds other points which get bright when the asteroid passes through
	'''
	astarray = [(time, pos1, pos2)]
	
	pixcurve = flux[time-30:time+30,pos1,pos2]
	pixcurve = pixcurve/np.nanmedian(pixcurve)
	pcdiff = np.diff(pixcurve)
	control = signal.correlate(pcdiff,pcdiff)	
	centre = np.where(control == np.max(control))
	
	for i in range(flux.shape[1]):
		for j in range(flux.shape[2]):
			normflux = flux[time-30:time+30,i,j]/np.median(flux[time-30:time+30,i,j])
			corr = signal.correlate(pcdiff,np.diff(normflux))
			cormax = np.where(corr == np.max(corr))
			tshift = centre[0] - cormax[0]
			if (tshift < 0) or (tshift > 0):
				peaks,_ = find_peaks(corr)
				prominence = peak_prominences(corr,peaks)
				proms = np.sort(prominence[0])
				promlen = proms.shape[0]
				
				if proms[promlen-1] >=  10*proms[promlen-2]:

					start = time + tshift[0] - 3
					end = time + tshift[0] + 4
										
					ftind = np.argmax(flux[start:end,i,j]) + time + tshift - 3
					
					astarray += [(ftind, i, j)]
					
	astarray = np.array(astarray)
	return astarray




def Single_pixel(ast1,flux):
	'''
	Given the array of pixels containing the asteroid, selects one pixel for each frame based on the maximum of the derivative
	'''

	ast2 = [] #creating empty list

	for i in np.unique(ast1[:,0]): #finding the different time indices
		ind = np.where(ast1[:,0] == i) #creating an array of all points with this time index
		start = i-3
		end= i+3
		maxdif = []
		for j in range(len(ind[0])): #running through the elements of ind
			#find the maximum derivative of the light curve in each of these pixels near the time
			maxdif += [np.max(np.diff(flux[start:end, ast1[ind][j][1], ast1[ind][j][2]]))]
			#finding the pixel for which this is greatest
		pix = np.argmax(maxdif)
		#appending this pixel to the list
		ast2 += [(i,ast1[ind][pix][1],ast1[ind][pix][2])]

	#converting the list to an array		
	ast2 = np.array(ast2)
	#print(ast2)
	
	return ast2



def Watershed_object_sep(obj):
	'''
	Uses the watershed method to identify components in the object mask.
	'''
	obj = obj*1
	distance = ndi.distance_transform_edt(obj)
	local_maxi = peak_local_max(
		distance, indices=False, footprint=np.ones((3, 3)), labels=obj)
	markers = ndi.label(local_maxi)[0]
	labels = watershed(-distance, markers, mask=obj)

	temp = np.nanmax(labels)
	Objmasks = []
	for i in range(temp):
		i += 1
		# print(i)
		Objmasks += [(labels == i)*1]

	Objmasks = np.array(Objmasks)
	return Objmasks




def Flux_Asteroid(pos1, pos2, ast2, flux):
	'''
	Uses the watershed mask to select points from the asteroid array which fall within a reasonable path
	'''
	asteroid_mask = np.zeros_like(flux[0])
	asteroid_mask[ast2[:,1],ast2[:,2]] = 1

	asteroids = convolve(asteroid_mask,np.ones((3,3)),mode='constant', cval=0.0)

	masks = Watershed_object_sep(asteroids)

	ind = np.where(masks[:,pos1,pos2] == 1)[0]

	ast_mask2 = np.zeros_like(flux)
	ast_mask2[ast2[:,0],ast2[:,1],ast2[:,2]] = 1
	
	ast3 = []
	astar = []
	
	for i in ast2[:,0]:
		if np.any(ast_mask2[i]*masks[ind[0]] == 1):
			ast = np.where(ast_mask2[i]*masks[ind[0]] == 1)
			ast3 += [(i, ast[0][0], ast[1][0])]
		
	ast3 = np.array(ast3)
	
	return ast3, masks[ind[0]]




def Heuristic_scene_model(Flux, Time, Frameno, Dist_matrix):
	'''
	Corrects for the drifting of Kepler and subtracts the stationary image from a given frame
	
	made by Gully

	'''
	this_dist = Dist_matrix[Frameno, :]
	sort_indices = np.argsort(this_dist)
	### Tuning parameters
	# Make sure that the kept cadences are at least within a minimal proximity
	minimal_proximity = 0.1 #pixels
	# Make sure that the difference image is composed of frames that are a set
	# distance away in time.  (limit self subtraction)
	minimal_time_difference = 20.0 #days
	
	time_ind = np.where(abs(Time[Frameno] - Time) > minimal_time_difference)[0]
	subset_distance = Dist_matrix[Frameno, time_ind]
	distance_ind = np.where(subset_distance <= minimal_proximity)[0]
	
	#Safety net: our scene model is not satisfiable!
	if len(distance_ind) <= 10: #any(( this_dist[subset] > minimal_proximity) |  
								#(np.abs(Time[subset] - Time[Frameno]) < minimal_time_difference)):
		median_frame = Flux[Frameno] *np.NaN
		std_frame = Flux[Frameno] *np.NaN
	else:
		#median_frame, std_frame = Weighted_arrays(Flux[time_ind[distance_ind]], 1/subset_distance[distance_ind])
		
		median_frame = np.nanmedian(Flux[time_ind[distance_ind],:,:], axis=(0))
		std_frame = np.nanstd(Flux[time_ind[distance_ind],:,:], axis=(0))
	return median_frame, std_frame



def Flux_ImSub_Asteroid(ast1, flux, time, masks,dist_matrix):
	'''
	Given the array of pixels containing the asteroid, selects one pixel for each frame 
	by taking the maximum of the normalised image subtraction within the watershed mask
	'''

	ast4 = [] #creating empty list
	Im2 = []

	mintime = np.min(np.unique(ast1[:,0]))
	maxtime = np.max(np.unique(ast1[:,0]))

	for i in range(mintime, maxtime+1): #finding the different time indices
	
		Im, Std = Heuristic_scene_model(flux,time,i,dist_matrix)
		Significance = (flux[i]-Im)/Std
		
		astsigs = Significance*masks
		
		ast = np.where(np.nanmax(astsigs) == astsigs)
		ast4 += [(i,ast[0][0],ast[1][0])]
		
		Im2 += [flux[i,ast[0][0],ast[1][0]]-Im[ast[0][0],ast[1][0]]]

	#converting the list to an array		
	ast4 = np.array(ast4)
	return ast4



def Sub_Asteroid(mintime, maxtime, time, flux,dist_matrix):
	'''
	Finds pixels likely to contain the asteroid by selecting the maximum of the normalised image subtraction in each frame	
	'''
	Astarray = []
	sig = []
	ts = []
	
	for i in range(mintime, maxtime+1):
		Im, Std = Heuristic_scene_model(flux,time,i,dist_matrix)
		Significance = (flux[i]-Im)/Std
		ast = np.where(np.nanmax(Significance) == Significance)
		Astarray += [(i,ast[0][0],ast[1][0])]
		sig += [np.nanmax(Significance)]
		ts += [i]
		
	ts = np.array(ts)
	sig = np.array(sig)
	
	ImFl = [ts,sig]
	
	Astarray = np.array(Astarray)
	return Astarray, ImFl



def _query_solar_system_objects(ra, dec, times, radius=0.1, location='kepler',
								cache=False):
	"""Returns a list of asteroids/comets given a position and time.
	This function relies on The Virtual Observatory Sky Body Tracker (SkyBot)
	service which can be found at http://vo.imcce.fr/webservices/skybot/
	 Geert's magic code

	Parameters
	----------
	ra : float
		Right Ascension in degrees.
	dec : float
		Declination in degrees.
	times : array of float
		Times in Julian Date.
	radius : float
		Search radius in degrees.
	location : str
		Spacecraft location. Options include `'kepler'` and `'tess'`.
	cache : bool
		Whether to cache the search result. Default is True.
	Returns
	-------
	result : `pandas.DataFrame`
		DataFrame containing the list of known solar system objects at the
		requested time and location.
	"""
	if (location.lower() == 'kepler') or (location.lower() == 'k2'):
		location = 'C55'
	elif location.lower() == 'tess':
		location = 'C57'

	url = 'http://vo.imcce.fr/webservices/skybot/skybotconesearch_query.php?'
	url += '-mime=text&'
	url += '-ra={}&'.format(ra)
	url += '-dec={}&'.format(dec)
	url += '-bd={}&'.format(radius)
	url += '-loc={}&'.format(location)

	df = None
	times = np.atleast_1d(times)
	for time in tqdm(times, desc='Querying for SSOs'):
		url_queried = url + 'EPOCH={}'.format(time)
		response = download_file(url_queried, cache=cache)
		if open(response).read(10) == '# Flag: -1':  # error code detected?
			raise IOError("SkyBot Solar System query failed.\n"
						  "URL used:\n" + url_queried + "\n"
						  "Response received:\n" + open(response).read())
		res = pd.read_csv(response, delimiter='|', skiprows=2)
		if len(res) > 0:
			res['epoch'] = time
			res.rename({'# Num ':'Num', ' Name ':'Name', ' Class ':'Class', ' Mv ':'Mv'}, inplace=True, axis='columns')
			res = res[['Num', 'Name', 'Class', 'Mv', 'epoch']].reset_index(drop=True)
			if df is None:
				df = res
			else:
				df = df.append(res)
	if df is not None:
		df.reset_index(drop=True)
	return df

def LC(astpos, flux):
	'''
	astpos is an array of indices
	flux is the flux array
	t is the time given as the initial time relative to the flux array's indexing
	'''

	from scipy.ndimage.filters import convolve
	
	astmask = np.zeros((flux.shape[0], flux.shape[1], flux.shape[2]))
	astmask[astpos[:,0], astpos[:,1], astpos[:,2]] = 1
	
	kernel = np.ones((1,3,3))
	astmaskcon = convolve(astmask, kernel)/convolve(astmask, kernel)
	
	lcarr = flux*astmaskcon
	
	lc = []

	for i in astpos[:,0]:
			lc += [np.nansum(lcarr[i])]
	
	#LC = lc[astpos[0,0]:astpos[0,0] + astpos.shape[0]]
	
	return lc, astmaskcon

def Significance(t, flux, time,xdrif,ydrif):
	
	Xdiff = xdrif
	Ydiff = ydrif
	dist_matrix = np.sqrt( (Xdiff - Xdiff[:, np.newaxis])**2 + (Ydiff - Ydiff[:, np.newaxis])**2 )
	
	Sig_Array = []
	Im_Array = []
	
	if t>50:
		for i in range(t-50, t+50):
			Im, Std = Heuristic_scene_model(flux,time,i,dist_matrix)
			Significance = (flux[i]-Im)/Std
			Im_Array += [(Im)]
			Sig_Array += [(Significance)]
			T = t - 50
	else:
		for i in range(0, t+50):
			Im, Std = Heuristic_scene_model(flux,time,i,dist_matrix)
			Significance = (flux[i]-Im)/Std
			Im_Array += [(Im)]
			Sig_Array += [(Significance)]
			T = 0
	
	Sig_Array = np.array(Sig_Array)
	Im_Array = np.array(Im_Array)
	
	Im_array_big = np.zeros((flux.shape[0], flux.shape[1], flux.shape[2]))
	if t>50:
		Im_array_big[t-50:t+50] = Im_Array
	else:
		Im_array_big[0:t+50] = Im_Array
	
	return Sig_Array, Im_Array, T


def create_dataframe(Ast_Fl, Ast_ImFl, Ast_Im, masks,time,WCS): 
	
	flt = []
	flr = []
	fld = []
	
	for i in np.unique(Ast_Fl[:,0]):
		flt += [time[i]]
	for i in range(Ast_ImFl.shape[0]):
		if (i < Ast_Fl.shape[0]):
			r, d = WCS.all_pix2world(Ast_Fl[i,2],Ast_Fl[i,1],0)
			flr += [r]
			fld += [d]
		else:
			flr += [0]
			fld += [0]
			flt += [0]
		
	
	fllc, maskfl = LC(Ast_Fl, flux)
	
	n = Ast_ImFl.shape[0] - Ast_Fl.shape[0]
	
	fllc = np.append(fllc, np.zeros(n))
	
	if len(masks)>0:
		imflt = []
		imflr = []
		imfld = []
		for i in range(Ast_ImFl.shape[0]):
			r, d = WCS.all_pix2world(Ast_ImFl[i,2],Ast_ImFl[i,1],0)
			imflr += [r]
			imfld += [d]
		for i in np.unique(Ast_ImFl[:,0]):
			imflt += [time[i]]
		
		imfllc, maskimfl = LC(Ast_ImFl, flux)
	
	imt = []
	imr = []
	imd = []
	for i in range(Ast_Im.shape[0]):
		r, d = WCS.all_pix2world(Ast_Im[i,2],Ast_Im[i,1],0)
		imr += [r]
		imd += [d]
	for i in np.unique(Ast_Im[:,0]):
		imt += [time[i]]
	
	imlc, maskim = LC(Ast_Im, flux)
	
	if len(masks>0):
		lcdf = pd.DataFrame({'Time 1': flt,
							 'RA 1': flr,
							 'Dec 1': fld,
							 'Flux (correlation method)':fllc, 
							 'Time 2': imflt,
							 'RA 2': imflr,
							 'Dec 2': imfld,
							 'Flux (correlation/image subtraction method)':imfllc,
							 'Time 3': imt,
							 'RA 3': imr,
							 'Dec 3': imd,
							 'Subtracted Flux (image subtraction method)':imlc})
	else:
		lcdf = pd.DataFrame({'Time 1': flt,
							 'RA 1': flr,
							 'Dec 1': fld,
							 'Flux (correlation method)':fllc, 
							 'Time 3': imt,
							 'RA 3': imr,
							 'Dec 3': imd,
							 'Flux (image subtraction method)':imlc})
	
	
	
	return lcdf


def Save_space(Save):
	try:
		if not os.path.exists(Save):
			os.makedirs(Save)
	except FileExistsError:
		pass

def Asteroid_move(Data, Ast_Fl, Ast_ImFl, Ast_Im, ID, Save):
	"""
	Makes a movie of the asteroid tracking mask.

	Inputs
	------
	Data : array
		flux array
	Ast_Fl, Ast_ImFl, Ast_Im: arrays
		arrays indices of the asteroid positions and times in the flux array
	ID : str
		unique identifier for the target
	Save : str
		save location of the final video

	"""
	
	Masktime1 = Ast_Fl[:,0]
	Masktime2 = Ast_ImFl[:,0]
	Masktime3 = Ast_Im[:,0]
	t = Ast_Fl[0,0]
	
	height = 1100/2
	width = 2200/2
	my_dpi = 100
	

	FrameSave = Save + '/' + ID + '/'
	Save_space(FrameSave)
	Section = Data[Masktime3[0]:Masktime3[-1],:,:]
	for i in range(len(Section)):
		filename = FrameSave + 'Frame_' + str(int(i)).zfill(4)+".png"
		
		fig = plt.figure(figsize=(width/my_dpi,height/my_dpi),dpi=my_dpi)
		
		plt.subplot(1, 2, 1)
		plt.title('Asteroid light curves')
		
		lc1, astlc1 = LC(Ast_Fl, flux)
		plt.plot((Masktime1[:-1]-Masktime1[0])/2,lc1[:-1],'r-',lw=2, label = 'Flux (correlation method)')
		plt.plot((Masktime1[:-1]-Masktime1[0])/2,lc1[:-1],'x')
		
		lc2, astlc2 = LC(Ast_ImFl, flux)
		plt.plot((Masktime2[:-1]-Masktime2[0])/2,lc2[:-1],'g-',lw=2, label = 'Flux (correlation/subtraction)')
		plt.plot((Masktime2[:-1]-Masktime2[0])/2,lc2[:-1],'x')
		
		lc3, astlc3 = LC(Ast_Im, flux)
		plt.plot((Masktime3[:-1]-Masktime3[0])/2,lc3[:-1],'b-',lw=2, label = 'Flux (subtraction method)')
		plt.plot((Masktime3[:-1]-Masktime3[0])/2,lc3[:-1],'x')
		
		plt.axvline((Masktime3[i]-Masktime3[0])/2, color ='red')  
		plt.legend()
		plt.ylabel('Counts')
		plt.xlabel('Time (hours)');
		
		plt.subplot(1,2,2)
		plt.title('Kepler image')
		plt.imshow(Section[i,:,:], origin='lower',vmin=0,vmax=1000)
		plt.plot(np.where(astlc3[i+t] == 1)[1],np.where(astlc3[i+t] == 1)[0],'b.')   
		plt.plot(np.where(astlc2[i+t] == 1)[1],np.where(astlc2[i+t] == 1)[0],'g.')   
		plt.plot(np.where(astlc1[i+t] == 1)[1],np.where(astlc1[i+t] == 1)[0],'r.') 
		plt.colorbar(fraction=0.046, pad=0.04)
		plt.savefig(filename)
		
	framerate = (len(Section))/5
	directory = Save

	ffmpegcall = ('ffmpeg -y -nostats -loglevel 8 -f image2 -framerate ' + 
				str(framerate) + ' -i ' + FrameSave + 'Frame_%04d.png -vcodec libx264 -pix_fmt yuv420p ' 
				+ directory + 'Ast-' + ID + '.mp4')
	os.system(ffmpegcall);

#Given an input position, time and flux, returns arrays of asteroid position/time and their light curves using 3 methods

def Track_Asteroid(x,y,t,flux,time,xdrift,ydrift,WCS,save,name):
	print('name',name)
	pos1, pos2, time = Initial(x,y,t,flux)
	Xdiff = xdrift
	Ydiff = ydrift 
	dist_matrix = np.sqrt( (Xdiff - Xdiff[:, np.newaxis])**2 + (Ydiff - Ydiff[:, np.newaxis])**2 )
	ast1 = asteroid(pos1, pos2, time, flux)
	ast2 = Single_pixel(ast1, flux)
	Ast_Fl, masks = Flux_Asteroid(pos1, pos2, ast2, flux)
	print('past initial')
	if len(masks>0):
		Ast_ImFl = Flux_ImSub_Asteroid(ast1, flux, masks,dist_matrix)
	
	Ast_Im, Lc_Im = Sub_Asteroid(np.min(np.unique(ast1[:,0])),np.max(np.unique(ast1[:,0])),flux,dist_matrix)
	
	Lc_Fl = np.array((Ast_Fl[:,0],flux[Ast_Fl[:,0],Ast_Fl[:,1],Ast_Fl[:,2]]))
	
	if len(masks>0):
		Lc_ImFl = np.array((Ast_ImFl[:,0],flux[Ast_ImFl[:,0],Ast_ImFl[:,1],Ast_ImFl[:,2]]))
		
	#find asteroid position 
	#r, d = WCS.all_pix2world(pos2,pos1,0)
	#t = tpf.astropy_time.jd[time]
	#plug stuff into Geert's code
	#obj = _query_solar_system_objects(r,d,t,radius=10/60**2)
	
	if len(masks>0):
		print('saving')
		Fldf = create_dataframe(Ast_Fl, Ast_ImFl, Ast_Im, masks,time,WCS)
		Fldf.to_csv(save+name)
		Asteroid_move(flux, Ast_Fl, Ast_ImFl, Ast_Im, name, 'ast_videos')
	
	return Ast_Fl, Ast_ImFl, Ast_Im, masks