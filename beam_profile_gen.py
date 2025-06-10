# -*- coding: utf-8 -*-

import os

import numpy as np
# import scipy.integrate
# import scipy.signal
# import scipy.special
import skimage.io
import random
# from skimage import filters
# from skimage.transform import rescale, resize
# from scipy.io import savemat
# import scipy.io as sio

from PIL import Image
# import matplotlib.pyplot as plt
# from scipy import interpolate
import sys
N = 500 # grid size
px = 1e-6 # pixel size (um)
focal_length = 2e-3
wavelength = 0.561e-6
refractive_index = 1.0
psf_width_pixels = 101
pixel_size_meters = 1e-6
psf_width_meters = psf_width_pixels * pixel_size_meters
numerical_aperture = 0.6

def normalize_to_uint16(img):
    #img = img.astype(np.float32)
    img = img - img.min()
    if img.max() > 0:
        img = img / img.max()
    img_uint16 = (img * 65535).astype(np.uint16)
    return img_uint16

# generate phase mask which is the input on SLM (spatial light modulator) 
def phase_mask_gen(fwhm=0.0001): 
    # default value for c is 0.0001
    x = list(range(-N//2,N//2))
    y = list(range(-N//2,N//2))
    [X, Y] = np.meshgrid(x,y)
    X = X*px
    Y = Y*px

    mask = 1*np.exp(-(np.square(X) + np.square(Y))/(2*fwhm**2))
    # choose a mask
    # bessel mask
    #set_1 = np.logical_or(mask >= 0.76, mask <= 0.75)
    #mask[set_1] = 0.0
    # gaussian mask
    #set_2 = mask <= 0.75
    #mask[set_2] = 0.75
    return mask

# the beam propagate through SLM and lens and focus the beam at focal distance 
def beam_profile_focus(mask): 
    x = list(range(-N//2,N//2))
    y = list(range(-N//2,N//2))
    [X, Y] = np.meshgrid(x,y)
    X = X*px
    Y = Y*px
    C1 = (np.pi/(wavelength*focal_length) * (np.square(X) + np.square(Y)))%(2*np.pi) # lens function lens as a phase transformer
    B1 = np.exp(-1j*C1) # convert lens function to phase
    B1=B1*mask # propgation multiply by lens, fourier proerties of lens
    
    xx = list(range(-N + 1, N + 1))
    yy = list(range(-N + 1, N + 1))
    [XX, YY] = np.meshgrid(xx,yy)
    XX = XX*px
    YY = YY*px
    E1 = np.pad(B1,N//2)
    Q1 = np.exp(1j * (np.pi/(wavelength*focal_length)) * (np.square(XX) + np.square(YY))) # Fresnel diffraction equation at distance = focal length
    E2 = np.fft.ifftshift(np.fft.ifft2(np.fft.fft2(E1) * np.fft.fft2(Q1)))
    ans = E2[N//2:3*N//2,N//2:3*N//2]
    #return (np.fft.fftshift(np.fft.fft2(mask)))
    return ans

def beam_profile_lensless(mask): 
    x = list(range(-N//2,N//2))
    y = list(range(-N//2,N//2))
    [X, Y] = np.meshgrid(x,y)
    X = X*px
    Y = Y*px
    C1 = (np.pi/(wavelength*focal_length) * (np.square(X) + np.square(Y)))%(2*np.pi) # lens function lens as a phase transformer
    B1 = np.exp(-1j*C1) # convert lens function to phase
    B1=B1*mask # propgation multiply by lens, fourier proerties of lens
    
    xx = list(range(-N + 1, N + 1))
    yy = list(range(-N + 1, N + 1))
    [XX, YY] = np.meshgrid(xx,yy)
    XX = XX*px
    YY = YY*px
    E1 = np.pad(B1,N//2)
    Q1 = np.exp(1j * (np.pi/(wavelength*focal_length)) * (np.square(XX) + np.square(YY))) # Fresnel diffraction equation at distance = focal length
    E2 = np.fft.ifftshift(np.fft.ifft2(np.fft.fft2(E1) * np.fft.fft2(Q1)))
    ans = E2[N//2:3*N//2,N//2:3*N//2]
    #return (np.fft.fftshift(np.fft.fft2(mask)))
    return ans


# generate profile along illumination axis using angular specturm, z is the propagate distance. 
def angularSpec(layer,z):
    k = 2*np.pi/wavelength
    phy_x = N*px # physical width (meters) 
    phy_y = N*px # physical length (meters) 
    obj_size = layer.shape
    # generate meshgrid
    dx = np.linspace(-phy_x/2,phy_x/2,obj_size[1])
    dy = np.linspace(-phy_y/2,phy_y/2,obj_size[0])
    Fs_x = obj_size[1]/phy_x
    Fs_y = obj_size[0]/phy_y
    dFx = Fs_x/obj_size[1]
    dFy = Fs_y/obj_size[0]
    Fx = np.arange(-Fs_x/2,Fs_x/2,dFx)
    Fy = np.arange(-Fs_y/2,Fs_y/2,dFy)
    # alpha and beta (wavenumber components) 
    alpha = wavelength*Fx
    beta = wavelength*Fy
    [ALPHA,BETA] = np.meshgrid(alpha,beta)
    #gamm_cust = np.zeros((len(alpha),len(beta)))
    gamma_cust = np.sqrt(1 - np.square(ALPHA) - np.square(BETA))
    U1 = np.fft.ifft2(np.fft.ifftshift(np.fft.fftshift(np.fft.fft2(layer))*np.exp(1j*k*gamma_cust*z)))
    
    return np.real(U1 * np.conj(U1))

# generate a tilted pahse 
def phase_gen():
    line = np.linspace(0,0,500)
    
    ans = np.tile(line,(500,1))
    # another gradient
    # ans = ans.transpose()
    #return 4*(np.random.random((500,500)) - 0.5)
    return np.tile(line,(500,1)) 

# used to generate beam profile in 3D space
# confused about which one is x and y
def beam_section(layer,output_folder,x_min,x_max,y_min,y_max):
    profile = np.zeros((x_max-x_min,101))
    x_dist = range(x_min,x_max,1)
    for index in range(len(x_dist)):
        print(index)
        pro_x = angularSpec(layer,x_dist[index]*px) # beam in [z,x] plane (camera coordinates)
        #return pro_x.shape
        profile[index] = pro_x[pro_x.shape[0]//2,pro_x.shape[1]//2+y_min-1:pro_x.shape[1]//2+y_max] # [z,y]
        skimage.io.imsave(os.path.join(output_folder,str(index) + '.tiff'), normalize_to_uint16(pro_x))
        #skimage.io.imsave(os.path.join(output_folder,str(index) + '.png'), (pro_x/1e7).astype('uint16'))
    return normalize_to_uint16(profile)


if __name__ == '__main__':
    # generate psf for defocus, only for once
    # generate_psf()
    mask_real = phase_mask_gen()
    #mask_real = skimage.io.imread('mask_real_epoch_178_1600.tiff')
    #skimage.io.imsave('mask.tiff',mask_real.astype('float32'))
    

    #sys.exit()
    #mask_phase = skimage.io.imread('results/phase_model__20221103-214157/mask_phase_epoch_0_0.tiff')
    #mask_phase = skimage.io.imread('phase_learned/mask_phase_epoch_0_0.tiff')
    #mask_phase = 0*2*np.pi*(np.random.rand(500,500))
    mask_phase = phase_gen()
    #mask_phase = skimage.io.imread('./results/model_fig4/mask_phase_epoch_100_0.tiff')
    mask_param = mask_real*np.exp(1j*mask_phase)
    layer = beam_profile_focus(mask_param)
    #ans = angularSpec(layer,0*1e-6)
    profile = beam_section(layer)
    skimage.io.imsave('aaa.tiff',(profile/1e6).astype('uint16'))
    #ans = img_gen(layer)
    
    # 253 
    

