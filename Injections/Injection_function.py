import lightkurve as lk
import pandas as pd
import numpy as np

from astropy.convolution import convolve, Gaussian2DKernel, Tophat2DKernel
from astropy.modeling.models import Gaussian2D

def KSN2015K(Magnitude,Start,Time):
    K = np.zeros(len(Time))
    Raw = np.loadtxt('/Users/ryanr/Documents/PhD/coding/Kepler/pipelines/Injections/SN/KSN2015K_v4_ap5.txt')
    Eventstart = np.where(Raw[:,0]==2400.008166)[0][0]
    Eventend = np.where(Raw[:,0]==2430.985510)[0][0]
    # Columns of Raw are, Time, Counts, Background subtracted counts
    Counts = Raw[:,2] - Raw[1,2]
    #m1 = -5/2*np.log10(Counts)+25.47
    LCmax = np.nanmin(m1)
    Counts2 = Counts*10**(2/5*(LCmax - Magnitude))
    Counts2 = Counts2[Eventstart:Eventend]
    if len(Counts2) > len(K[Start:]):
        K[Start:] += Counts2[:(len(K)-Start)]
    elif len(K[Start:]) > len(Counts2):
        K[Start:Start+len(Counts2)] += Counts2
    else:
        K[Start:] += Counts2
    
    return K 

def KSN2017K(Magnitude,Start,Time):
    K = np.zeros(len(Time))
    Raw = pd.read_csv('/Users/ryanr/Documents/PhD/coding/Kepler/Data/SNReference/SN2017jgi_ap5_v2.csv').values
    Eventstart = 0 #np.where(Raw[:,0]==2400.008166)[0][0]
    Eventend = -1 #np.where(Raw[:,0]==2430.985510)[0][0]
    # Columns of Raw are, Time, Counts, Background subtracted counts
    Counts = Raw[:,1] - np.nanmedian(Raw[100:200,1])
    m1 = -5/2*np.log10(Counts)+25.47
    LCmax = np.nanmin(m1)
    Counts2 = Counts*10**(2/5*(LCmax - Magnitude))
    Counts2 = Counts2[Eventstart:Eventend]
    if len(Counts2) > len(K[Start:]):
        K[Start:] += Counts2[:(len(K)-Start)]
    elif len(K[Start:]) > len(Counts2):
        K[Start:Start+len(Counts2)] += Counts2
    else:
        K[Start:] += Counts2
    K[K<0] = 0
    return K 

def KSN2015E(Magnitude,Start,Time):
    K = np.zeros(len(Time))
    Raw = np.loadtxt('/Users/ryanr/Documents/PhD/coding/Kepler/Data/SNReference/KSN2015e_ap3_v2.txt')
    Eventstart = 0 #np.where(Raw[:,0]==2400.008166)[0][0]
    Eventend = -1 #np.where(Raw[:,0]==2430.985510)[0][0]
    # Columns of Raw are, Time, Counts, Background subtracted counts
    Counts = Raw[:,1]
    Counts[Counts == 0] = np.nan
    Counts = Counts - np.nanmedian(Raw[200:300,1])
    m1 = -5/2*np.log10(Counts)+25.47
    LCmax = np.nanmin(m1)
    Counts2 = Counts*10**(2/5*(LCmax - Magnitude))
    Counts2 = Counts2[Eventstart:Eventend]
    if len(Counts2) > len(K[Start:]):
        K[Start:] += Counts2[:(len(K)-Start)]
    elif len(K[Start:]) > len(Counts2):
        K[Start:Start+len(Counts2)] += Counts2
    else:
        K[Start:] += Counts2
    K[K<0] = 0
    return K 


def Blur_seed(TPF,Model,Start, Mag):
    x = np.arange(-0, TPF.flux.shape[1])
    y = np.arange(0, TPF.flux.shape[2])
    x, y = np.meshgrid(x, y)

    seed = [np.random.rand()*TPF.flux.shape[1],np.random.rand()*TPF.flux.shape[2]]  
    if 'K' in Model:
        model_lc = KSN2017K(Mag, Start, TPF.time)
    elif 'E' in Model:
        model_lc = KSN2015E(Mag, Start, TPF.time)
    
    Seeded = np.copy(TPF.flux)
    for i in range(len(TPF.time)):
        gauss = Gaussian2D(model_lc[i], seed[0], seed[1], 1, 1)

        data_2D = gauss(x, y)
        Seeded[i] += data_2D
    return Seeded

