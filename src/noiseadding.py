import os
import glob
import torch
import numpy as np
# from skimage import io, transform
from torchvision import transforms
from torchvision.utils import make_grid
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np
import pylops

# import glob
import cv2 as cv2
import numpy as np
# import matplotlib.pyplot as plt
import random
import math
import pylops
from pylops.utils.wavelets import ricker
from skimage import io
from skimage import color
from skimage.restoration import denoise_nl_means, estimate_sigma

import numpy as np
from numpy.fft import fft, fftfreq, ifft

from scipy import ndimage as nd
from scipy.fft import fft, ifft
from scipy import fftpack

from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import time
import cv2
from math import sqrt
from math import exp
from matplotlib.colors import LogNorm

from numpy import sqrt, newaxis, integer
from numpy.fft import irfft, rfftfreq
from numpy.random import default_rng, Generator, RandomState
from numpy import sum as npsum



class add_color_noise(object):
    def powerlaw_psd_gaussian(self,exponent, size, fmin, random_state):
        # Make sure size is a list so we can iterate it and assign to it.
        try:
            size = list(size)
        except TypeError:
            size = [size]

        # The number of samples in each time series
        samples = size[-1]

        # Calculate Frequencies (we asume a sample rate of one)
        # Use fft functions for real output (-> hermitian spectrum)
        f = rfftfreq(samples)

        # Validate / normalise fmin
        if 0 <= fmin <= 0.5:
            fmin = max(fmin, 1./samples) # Low frequency cutoff
        else:
            raise ValueError("fmin must be chosen between 0 and 0.5.")

        # Build scaling factors for all frequencies
        s_scale = f    
        ix   = npsum(s_scale < fmin)   # Index of the cutoff
        if ix and ix < len(s_scale):
            s_scale[:ix] = s_scale[ix]
        s_scale = s_scale**(-exponent/2.)

        # Calculate theoretical output standard deviation from scaling
        w      = s_scale[1:].copy()
        w[-1] *= (1 + (samples % 2)) / 2. # correct f = +-0.5
        sigma = 2 * sqrt(npsum(w**2)) / samples

        # Adjust size to generate one Fourier component per frequency
        size[-1] = len(f)

        # Add empty dimension(s) to broadcast s_scale along last
        # dimension of generated random power + phase (below)
        dims_to_add = len(size) - 1
        s_scale     = s_scale[(newaxis,) * dims_to_add + (Ellipsis,)]

        # prepare random number generator
        normal_dist = self._get_normal_distribution(random_state)

        # Generate scaled random power + phase
        sr = normal_dist(scale=s_scale, size=size)
        si = normal_dist(scale=s_scale, size=size)

        # If the signal length is even, frequencies +/- 0.5 are equal
        # so the coefficient must be real.
        if not (samples % 2):
            si[..., -1] = 0
            sr[..., -1] *= sqrt(2)    # Fix magnitude

        # Regardless of signal length, the DC component must be real
        si[..., 0] = 0
        sr[..., 0] *= sqrt(2)    # Fix magnitude

        # Combine power + corrected phase to Fourier components
        s  = sr + 1J * si

        # Transform to real time series & scale to unit variance
        y = irfft(s, n=samples, axis=-1) / sigma

        return y
    def _get_normal_distribution(self,random_state):
        normal_dist = None
        if isinstance(random_state, (integer, int)) or random_state is None:
            random_state = default_rng(random_state)
            normal_dist = random_state.normal
        elif isinstance(random_state, (Generator, RandomState)):
            normal_dist = random_state.normal
        else:
            raise ValueError(
                "random_state must be one of integer, numpy.random.Generator, "
                "numpy.random.Randomstate"
            )
        return normal_dist
   
    def __call__(self, sample):
        noisescale=0.005 
        fmin=0         
        input, target = sample['input'], sample['target']
        img=input.squeeze()
        im=[]
        exponent=random.choice([1, 2])        
        noisescale=random.choice(range(1, 10, 1))*noisescale
        im.append(self.powerlaw_psd_gaussian(exponent, img.shape, fmin, None))
        y=np.array(np.array(im)).squeeze()*noisescale
        input +=np.expand_dims(y, axis=2)       
        return {'input': input, 'target': target}
    
class add_bandpassed_noise(object):
#     print(BasicNoise) 
        # USEFUL FUNCTIONS
    def band_limited_noise(self,min_freq, max_freq, np_seed_rnd, samples=1024, samplerate=1):
        freqs = np.fft.rfftfreq(samples, d=1/samplerate)
        f = np.zeros(samples)
        idx = np.where(np.logical_and(freqs >= min_freq, freqs <= max_freq))[0]
        f[idx] = 1
        return self.fftnoise(f, np_seed_rnd)
    def col_array_noise(self,datashape, np_seed_rnd, mnfreq=2, mxfreq=120, fs=500):
        noise_mod = np.zeros(datashape).T
        for i in range(len(noise_mod)):
            noise_mod[i,:] = self.band_limited_noise(mnfreq, mxfreq, np_seed_rnd, datashape[0], fs)
        return (noise_mod/np.mean(abs(noise_mod))).T
    def make_data(self,d, noiserange, nfreqrange, dt=500, nrels=500):
        noisy_data_list = []

    #     nsc = 0
        for i in range(nrels):  
            nsc=np.random.choice(np.arange(noiserange[0],
                                           noiserange[1],
                                           0.01)
                                )

            # Make bandpassed noise
            noise = self.col_array_noise(d.shape, 
                                    np.random.RandomState(seed=0), 
                                    mnfreq=nfreqrange[0], 
                                    mxfreq=nfreqrange[1],
                                    fs=1/dt
                                   )
            dn = d+(noise*nsc)
            noisy_data_list.append(np.expand_dims(np.expand_dims(dn,0),3))
        return noisy_data_list
    def fftnoise(self,f, np_seed_rnd):
        f = np.array(f, dtype='complex')
        Np = (len(f) - 1) // 2
        phases = np_seed_rnd.rand(Np) * 2 * np.pi
        phases = np.cos(phases) + 1j * np.sin(phases)
        f[1:Np+1] *= phases
        f[-1:-1-Np:-1] = np.conj(f[1:Np+1])
        return np.fft.ifft(f).real
    def __call__(self, sample):
        f_low=1
        f_high=10   
        input, target = sample['input'], sample['target']
        img=input.squeeze()
        dt = 0.002
        freqrange = [random.choice(range(f_low, f_low+5, 1)),random.choice(range(f_high, f_high+5, 1))]
        y=np.array(self.make_data(np.zeros_like(img), 
                                    [0.02,0.35], 
                                    freqrange, 
                                    dt=dt, 
                                    nrels=1)).squeeze()*0.5 
        input +=np.expand_dims(y, axis=2)     
        return {'input': input, 'target': target} 

class add_blurnoise(object):
    def __call__(self, sample):
        input, target = sample['input'], sample['target']
        img=input.squeeze()
        ksizemax=3
        imshape = img.shape
        kenellist=odd(1,ksizemax)                 
        ksizesize=random.choice(kenellist)
        ksize = (ksizesize, ksizesize)
        input = np.expand_dims( cv2.blur(img, ksize, cv2.BORDER_DEFAULT),axis=2)
        return {'input': input, 'target': target} 

class add_rainnoise(object):  
    def generate_random_lines(self,imshape,slant,drop_length,rain_type):
        drops=[]
        area=imshape[0]*imshape[1]
        no_of_drops=area//600
    #     print(no_of_drops)
        if rain_type.lower()=='drizzle':
            no_of_drops=area//770
            drop_length=10
        elif rain_type.lower()=='heavy':
            drop_length=30
        elif rain_type.lower()=='torrential':
            no_of_drops=area//3500
            drop_length=60
        for i in range(no_of_drops): ## If You want heavy rain, try increasing this
            if slant<0:
                x= np.random.randint(slant,imshape[1])
            else:
                x= np.random.randint(0,imshape[1]-slant)
            y= np.random.randint(0,imshape[0]-drop_length)
    #         print(slant,(x,y))
            drops.append((x,y))
        return drops,drop_length
    def rain_process(self,image,slant,drop_length,drop_color,drop_width,rain_drops):
        imshape = image.shape 
        image_t= image.copy()
        for rain_drop in rain_drops:
            cv2.line(image_t,(rain_drop[0],rain_drop[1]),(rain_drop[0]+slant,rain_drop[1]+drop_length),drop_color,drop_width)
        image= cv2.blur(image_t,(1,1)) ## rainy view are blurry
        return image
    def __call__(self, sample):
        input, target = sample['input'], sample['target']
        img=input.squeeze()
        slant=-1
        drop_length=20
        drop_width=1
        drop_color=(1) ## (200,200,200) a shade of gray
        slant_extreme=slant        
        raintype=['drizzle','heavy','torrential']       
        imshape = img.shape
        slant= np.random.randint(-10,10) ##generate random slant if no slant value is given           
        rain_type=random.choice(raintype)
        rain_drops,drop_length= self.generate_random_lines(imshape,slant,drop_length,rain_type)              
        drop_length=random.choice(range(drop_length, drop_length+10, 1))
        drop_width=random.choice(range(drop_width, drop_width+2, 1))
        drop_color=(random.choice(range(drop_color, drop_color+1, 1)))       
        noise=self.rain_process(img,slant_extreme,drop_length,drop_color,drop_width,rain_drops)
        input=np.expand_dims(noise,axis=2)
        return {'input': input, 'target': target} 
 
class add_gaussnoise(object):   
    def __call__(self, sample):
        noiselevel=3
        input, target = sample['input'], sample['target']
        img=input.squeeze()
        noise =  np.random.normal(loc=0, scale=random.choice(range(1, 20, 1)), size=img.shape)*0.01*noiselevel
        input+=np.expand_dims(noise,axis=2)
        return {'input': input, 'target': target}     
    
class add_spnoise(object):
     def __call__(self, sample):
        input, target = sample['input'], sample['target']
        img=input.squeeze()
        percentage=10
        row,col = img.shape
        s_vs_p = 0.50
        amount =percentage/100
        out = np.copy(img)
      # Salt mode
#             print(img.shape)
        num_salt = np.ceil(amount * img.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
              for i in img.shape]
        out[coords] = 1
      # Pepper mode
        num_pepper = np.ceil(amount* img.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
              for i in img.shape]
        out[coords] = 0       
        input=np.expand_dims(out,axis=2)
        return {'input': input, 'target': target}

class add_specklenoise(object):
    def __call__(self, sample):
        input, target = sample['input'], sample['target']
        img=input.squeeze()
        noiselevel=0.2   
        row,col = img.shape
        gauss = np.random.randn(row,col)
        gauss = gauss.reshape(row,col)        
        noisy = img + img * gauss*random.choice(range(1, 50, 1))*0.1*noiselevel
        input=np.expand_dims(noisy,axis=2)
        return {'input': input, 'target': target}
    
class add_noise_FFT(object):
    def distance(self,point1,point2):
        return sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)

    def gaussianLP(self,D0,imgShape):
        base = np.zeros(imgShape[:2])
        rows, cols = imgShape[:2]
        center = (rows/2,cols/2)
        for x in range(cols):
            for y in range(rows):
                base[y,x] = exp(((-self.distance((y,x),center)**2)/(2*(D0**2))))
        return base

    def gaussianHP(self,D0,imgShape):
        base = np.zeros(imgShape[:2])
        rows, cols = imgShape[:2]
        center = (rows/2,cols/2)
        for x in range(cols):
            for y in range(rows):
                base[y,x] = 1 - exp(((-self.distance((y,x),center)**2)/(2*(D0**2))))
        return base
    
    def __call__(self, sample):

        input, target = sample['input'], sample['target']
        img=input.squeeze()                 
        masktype="adds&pnoise"
        keep_fraction=0.3
        noiselevel=0.3
        f_lowpas=80
        f_high=10
        percentage=5
        im_fft = fftpack.fft2(img)
        if masktype=="crossfilter":     
            im_fft2 = im_fft.copy()           
            # Define the fraction of coefficients (in each direction) we keep
            keep_fraction = keep_fraction
            r, c = im_fft2.shape
            im_fft2[int(r*keep_fraction):int(r*(1-keep_fraction))] = 0
            # # Similarly with the columns:
            im_fft2[:, int(c*keep_fraction):int(c*(1-keep_fraction))] = 0
            # eal part for display.
            im_new = fftpack.ifft2(im_fft2).real   

        elif masktype=="addrandomnoise":
            im_fft2 = im_fft.copy() 
            im_fft2=im_fft2+np.random.normal(loc=0, scale=random.choice(range(1, 50, 1)), size=np.abs(im_fft2).shape)*noiselevel          
            im_new = fftpack.ifft2(im_fft2).real

        elif masktype=="adds&pnoise":
            im_fft2 = im_fft.copy() 
            speclist=[]
            img=im_fft2
            percentage=5
            row,col = img.shape
            s_vs_p = 0.5
            amount =percentage/100
            out = np.copy(img)
            num_salt = np.ceil(amount * img.size * s_vs_p)
            coords = [np.random.randint(0, i - 1, int(num_salt))
                  for i in img.shape]
            out[coords] = 1
            num_pepper = np.ceil(amount* img.size * (1. - s_vs_p))
            coords = [np.random.randint(0, i - 1, int(num_pepper))
                  for i in img.shape]
            out[coords] = 0
            im_new=fftpack.ifft2(out).real
        elif masktype=="lowpassfilter": 
            center = np.fft.fftshift(im_fft)
            d0=random.choice(range(f_lowpas, f_lowpas+200, 1))
#             print(d0)
            LowPassCenter = center * self.gaussianLP(d0,img.shape)
            im_fft2 = np.fft.ifftshift(LowPassCenter)             
            im_new = fftpack.ifft2(im_fft2).real
        elif masktype=="highpassfilter":    
            center = np.fft.fftshift(im_fft)
            d0=random.choice(range(f_high, f_high+10, 1))
#             print(d0)
            HighPassCenter = center * self.gaussianHP(d0,img.shape)
            im_fft2 = np.fft.ifftshift(HighPassCenter) 
            im_new = fftpack.ifft2(im_fft2).real 
    
        input=np.expand_dims(im_new,axis=2)
        return {'input': input, 'target': target}  
        
class add_linearnoise(object):
    def __call__(self, sample):
        
        v=100
        tsample=0.01
        slope=50
        ampli=20
        
        input, target = sample['input'], sample['target']
        img=input.squeeze()
        
        
#         print(input.shape,target.shape)
        f= random.choice(range(5, 50, 1))
            # f=30
        #     print(f)
        par = {"ox": 0, "dx": 1000/(img.shape[1]), "nx": img.shape[1]/2, "ot": 0, "dt": 0.002, "nt": img.shape[0], "f0":f}     
        
        tsample=random.choice(range(1, 20, 1))*tsample
        t0 = list(np.arange(-5, 5.0, tsample))     
        theta = [random.choice(range(slope-50, slope+50, 1))]*len(t0)     
        amp = [random.choice(range(-ampli, ampli, 1))*0.1]*len(t0)

        # create axis
        taxis, taxis2, xaxis, yaxis = pylops.utils.seismicevents.makeaxis(par)

        # create wavelet
        wav = ricker(taxis[:41], f0=par["f0"])[0]
        y = (
            pylops.utils.seismicevents.linear2d(xaxis, taxis, v, t0, theta, amp, wav)[1]
        )
        y = np.hstack([np.flip(y.T, axis=1)[:,:], y.T])
        input +=np.expand_dims(y, axis=2)
       
        return {'input': input, 'target': target}

def odd(l,u):
    return([a for a in range(l,u) if a%2 != 0])

def get_first_break_dataset(rootdir="/data/maksim/data/",
                       noise_transforms=[]):
    transforms_ = []
    transforms_ += [add_color_noise(),add_bandpassed_noise(),add_blurnoise(),add_rainnoise(),add_gaussnoise(),add_spnoise(),add_specklenoise(),add_linearnoise(),add_noise_FFT()]
    transforms_ += [FlipChannels(), ToTensor()]
#     
    print(transforms_)
    return FirstBreakLoader(rootdir, transform=transforms.Compose(transforms_))
    
    
firstbreak_dataset = get_dataset('firstbreak')
dataloader = DataLoader(firstbreak_dataset, batch_size=batch_size, shuffle=True, num_workers=workers)
sample = iter(dataloader).next()
for i in range(20):
    show(sample, idx=i)