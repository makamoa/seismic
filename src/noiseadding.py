import random
import pylops
from pylops.utils.wavelets import ricker
import numpy as np
from numpy.fft import fft, fftfreq, ifft
from scipy import fftpack

import cv2
from math import sqrt
from math import exp

from numpy import sqrt, newaxis, integer
from numpy.fft import irfft, rfftfreq
from numpy.random import default_rng, Generator, RandomState
from numpy import sum as npsum

def show(sample, idx=0, axes=None):
    input = sample['input'][idx]
    target = sample['target'][idx]
#     print(input.shape)
    if axes is None:
        fig, axes = plt.subplots(1,2, figsize=[20,10])
    axes[0].imshow(input.numpy().transpose((1, 2, 0)),cmap="seismic")
    axes[1].imshow(target.numpy().transpose((1, 2, 0)),cmap="seismic")

# show(sample, idx=0)
class add_color_noise(object):
    def __init__(self, noisescale=0.005): 
        self.noisescale=noisescale 
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
        noisescale=self.noisescale 
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
    def __init__(self, f_low=1,f_high=10,noisescale=0.5): 
        self.f_low=f_low
        self.f_high=f_high
        self.noisescale=noisescale
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
        f_low=self.f_low
        f_high=self.f_high  
        noisescale=self.noisescale 
        input, target = sample['input'], sample['target']
        img=input.squeeze()
        dt = 0.002
        freqrange = [random.choice(range(f_low, f_low+5, 1)),random.choice(range(f_high, f_high+5, 1))]
        y=np.array(self.make_data(np.zeros_like(img), 
                                    [0.02,0.35], 
                                    freqrange, 
                                    dt=dt, 
                                    nrels=1)).squeeze()*noisescale 
        input +=np.expand_dims(y, axis=2)     
        return {'input': input, 'target': target} 

class add_blurnoise(object):
    def __init__(self, ksizemax=3): 
        self.ksizemax=ksizemax        
    def __call__(self, sample):
        input, target = sample['input'], sample['target']
        img=input.squeeze()
        ksizemax=self.ksizemax
        imshape = img.shape
        kenellist=odd(1,ksizemax)                 
        ksizesize=random.choice(kenellist)
        ksize = (ksizesize, ksizesize)
        input = np.expand_dims( cv2.blur(img, ksize, cv2.BORDER_DEFAULT),axis=2)
        return {'input': input, 'target': target} 

class add_rainnoise(object): 
    def __init__(self, drop_length=20,drop_width=1,drop_color=(1)): 
        self.drop_length=drop_length
        self.drop_width=drop_width
        self.drop_color=drop_color
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
        drop_length=self.drop_length
        drop_width=self.drop_width
        drop_color=self.drop_color ## (200,200,200) a shade of gray
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
    def __init__(self, noiselevel=3): 
        self.noiselevel=noiselevel
    def __call__(self, sample):
        noiselevel=self.noiselevel
        input, target = sample['input'], sample['target']
        img=input.squeeze()
        noise =  np.random.normal(loc=0, scale=random.choice(range(1, 20, 1)), size=img.shape)*0.01*noiselevel
        input+=np.expand_dims(noise,axis=2)
        return {'input': input, 'target': target}     
    
class add_spnoise(object):
    def __init__(self, percentage=10): 
        self.percentage=percentage
    def __call__(self, sample):
        input, target = sample['input'], sample['target']
        img=input.squeeze()
        percentage=self.percentage
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
    def __init__(self, noiselevel=0.2): 
        self.noiselevel=noiselevel
    def __call__(self, sample):
        input, target = sample['input'], sample['target']
        img=input.squeeze()
        noiselevel=self.noiselevel   
        row,col = img.shape
        gauss = np.random.randn(row,col)
        gauss = gauss.reshape(row,col)        
        noisy = img + img * gauss*random.choice(range(1, 50, 1))*0.1*noiselevel
        input=np.expand_dims(noisy,axis=2)
        return {'input': input, 'target': target}
    
class add_noise_FFT(object):
    def __init__(self, masktype="crossfilter",keep_fraction=0.3,noiselevel=0.3,f_lowpas=80,f_high=10,percentage=5): 
        self.masktype=masktype
        self.keep_fraction=keep_fraction
        self.noiselevel=noiselevel
        self.f_lowpas=f_lowpas
        self.f_high=f_high
        self.percentage=percentage
        
        
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
        masktype=self.masktype
        keep_fraction=self.keep_fraction
        noiselevel=self.noiselevel
        f_lowpas=self.f_lowpas
        f_high=self.f_high
        percentage=self.percentage
        
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
 

class add_hpyer_noise(object):
    def __init__(self, v=100,tsample=2,slope=50,ampli=20): 
        self.velocity=v
        self.tsample=tsample
        self.slope=slope
        self.ampli=ampli
    def __call__(self, sample):        
        v=self.velocity
        tsample=self.tsample
        slope=self.slope
        ampli=self.ampli
        dt = 0.002
        input, target = sample['input'], sample['target']
        img=input.squeeze()              
#         f= random.choice(range(5, 50, 1))
#         par = {"ox": 0, "dx": 2000/(img.shape[1]), "nx": img.shape[1]/2, "ot": 0, "dt": 0.002, "nt": img.shape[0], "f0":f}            
    

#         print(dd.shape)

        num_events=random.choice(range(5, 10, 1))
    #     print(num_events)   
        z=np.sort(np.random.randint(low=50, high=6000, size=(num_events)))
        v=np.sort(np.random.randint(low=100, high=2500, size=(num_events)))
    #     print(z,v)
        for j in range(num_events):   
            f = random.choice(range(30, 50, 1))
            #print(f)
            length = 0.4        
            x_axis = np.arange(start=0,
                           stop=1000,
                           step=1000/(image_list[i].shape[1]))
            y_axis = np.arange(start=0,
                               stop=1,
                               step=50)

            rx, ry = np.meshgrid(x_axis, y_axis)
            rz=0
            recs = np.vstack([rx.flatten(),ry.flatten(), np.ones_like(rx.flatten())*rz]).T

            # Compute the traveltimes between a source and all the receivers

            tts = compute_tts([500,0,z[j]], recs, v[j])
            t_wav = np.arange(-length / 2, (length - dt) / 2, dt)
            wav = (1.0 - 2.0 * (np.pi ** 2) * (f ** 2) * (t_wav ** 2)) * np.exp(-(np.pi ** 2) * (f ** 2) * (t_wav ** 2))

            x= pylops.waveeqprocessing.marchenko.directwave(wav, tts, image_list[i].shape[0], dt, nfft=None, dist=None, kind='2d', derivative=True)
#             print(x.shape,dd.shape)
#             dd=dd+x        
#         image_RGB.append(dd)   
#     image_RGB=np.array(np.array(image_RGB)).squeeze()    

            input +=np.expand_dims(x, axis=2)
       
            return {'input': input, 'target': target}
       

class add_linearnoise(object):
    def __init__(self, v=100,tsample=2,slope=50,ampli=20):
        self.velocity=v
        self.tsample=tsample
        self.slope=slope
        self.ampli=ampli
    def __call__(self, sample):        
        v=self.velocity
        tsample=self.tsample
        slope=self.slope
        ampli=self.ampli
        
        
        input, target = sample['input'], sample['target']
        img=input.squeeze()              
        f= random.choice(range(5, 50, 1))
        par = {"ox": 0, "dx": 2000/(img.shape[1]), "nx": img.shape[1]/2, "ot": 0, "dt": 0.002, "nt": img.shape[0], "f0":f}            
        tsample=random.choice(range(1, 5, 1))*tsample
        t0 = list(np.arange(-10, 2.0, tsample)) 
        theta = [random.choice(range(slope, slope+2, 1))]*len(t0) 
        amp = [random.choice(range(-ampli, ampli, 1))]*len(t0)
        # create axis
        taxis, taxis2, xaxis, yaxis = pylops.utils.seismicevents.makeaxis(par)

        # create wavelet
        wav = ricker(taxis[:41], f0=par["f0"])[0]
#         wav = ricker(taxis, f0=par["f0"])[0]
        y = (
            pylops.utils.seismicevents.linear2d(xaxis, taxis, v, t0, theta, amp, wav)[1]
        )
        y = np.hstack([np.flip(y.T, axis=1)[:,:], y.T])
        input +=np.expand_dims(y, axis=2)
       
        return {'input': input, 'target': target}

class add_hyperbolic_noise(object):
    def __init__(self, v=100,tsample=0.05,slope=10,ampli=20): 
        self.velocity=v
        self.tsample=tsample
        self.slope=slope
        self.ampli=ampli
        
    def dist_calc(self,r, s):
        ''' euclidean distance
        '''
        dx = r[0] - s[0]
        dy = r[1] - s[1]
        dz = r[2] - s[2]
        return np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

    def comp_tt(self,dist, vel):
        ''' assume constant vel model and compute traveltimes
        '''
        return dist / vel

    def compute_tts(self,source, recs, vel):
        ''' compute travel times between a single source and array of receivers
        '''

        dist = [(lambda r: self.dist_calc(r, source))(r) for r in recs]
        tts = [(lambda d: self.comp_tt(d, vel))(d) for d in dist]

        return np.array(tts)    
        
        
        
    
    def __call__(self, sample):        
        v=self.velocity
        tsample=self.tsample
        slope=self.slope
        ampli=self.ampli
        
        
        input, target = sample['input'], sample['target']
        img=input.squeeze()  
    
        dt = 0.002 # time sampling
    
        
        dd= np.zeros_like(img)
#         print(dd.shape)

        num_events=random.choice(range(5, 20, 1))
    #     print(num_events)   
        z=np.sort(np.random.randint(low=50, high=6000, size=(num_events)))
        v=np.sort(np.random.randint(low=100, high=500, size=(num_events)))
    #     print(z,v)
        for j in range(num_events):   
            f = random.choice(range(30, 50, 1))
            #print(f)
            length = 0.4        
            x_axis = np.arange(start=0,
                           stop=1000,
                           step=1000/(img.shape[1]))
            y_axis = np.arange(start=0,
                               stop=1,
                               step=50)

            rx, ry = np.meshgrid(x_axis, y_axis)
            rz=0
            recs = np.vstack([rx.flatten(),ry.flatten(), np.ones_like(rx.flatten())*rz]).T

            # Compute the traveltimes between a source and all the receivers

            tts = self.compute_tts([500,0,z[j]], recs, v[j])
            t_wav = np.arange(-length / 2, (length - dt) / 2, dt)
            wav = (1.0 - 2.0 * (np.pi ** 2) * (f ** 2) * (t_wav ** 2)) * np.exp(-(np.pi ** 2) * (f ** 2) * (t_wav ** 2))

            y= pylops.waveeqprocessing.marchenko.directwave(wav, tts, img.shape[0], dt, nfft=None, dist=None, kind='2d', derivative=True)
#             print(x.shape,dd.shape)

            input +=np.expand_dims(y, axis=2)
       
        return {'input': input, 'target': target}
#             dd=dd+x        
#         image_RGB.append(dd)   
#     image_RGB=np.array(np.array(image_RGB)).squeeze()    

#     return image_RGB   
        
    
  
def odd(l,u):
    return([a for a in range(l,u) if a%2 != 0])
    
complex_noise_transforms \
    = [add_color_noise(), add_bandpassed_noise(), add_blurnoise(), add_rainnoise(), add_gaussnoise(), add_spnoise(), add_specklenoise(), add_linearnoise(), add_noise_FFT()]

strong_noise_transforms \
    = [add_color_noise(),add_bandpassed_noise(),add_blurnoise(),add_rainnoise(),add_gaussnoise(),add_spnoise(),add_specklenoise(),add_linearnoise(tsample=0.05,slope=10,ampli=20),add_hyperbolic_noise(),add_noise_FFT(masktype="crossfilter"),add_noise_FFT(masktype="addrandomnoise"),add_noise_FFT(masktype="adds&pnoise"),add_noise_FFT(masktype="lowpassfilter"),add_noise_FFT(masktype="highpassfilter")]

# complex_noise_transforms \
#     = [add_hpyer_noise()]
