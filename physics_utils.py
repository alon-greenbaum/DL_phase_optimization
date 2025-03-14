# -*- coding: utf-8 -*-
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from math import pi
import torch.fft
import skimage.io
import matplotlib.pyplot as plt
from datetime import datetime
import os
from data_utils import save_output_layer

# nohup python mask_learning.py &> ./logs/01-31-25-09-38.txt &


# This function creates a 2D gaussian filter with std=1, without normalization.
# during training this filter is scaled with a random std to simulate different blur per emitter
def gaussian2D_unnormalized(shape=(7, 7), sigma=1.0):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    hV = torch.from_numpy(h).type(torch.FloatTensor)
    return hV


def NextPowerOfTwo(number):
    # Returns next power of two following 'number'
    return 2 ** math.ceil(math.log(number, 2))

class BlurLayer(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.gauss = gaussian2D_unnormalized(shape=(7, 7)).to(device)
        self.std_min = 0.8
        self.std_max = 1.2

    def forward(self, img_4d, device):
        # number of the input PSF images
        Nbatch = img_4d.size(0)
        Nemitters = img_4d.size(1)
        # generate random gaussian blur for each emitter
        RepeatedGaussian = self.gauss.expand(1, Nemitters, 7, 7)
        stds = (self.std_min + (self.std_max - self.std_min) * torch.rand((Nemitters, 1))).to(device)
        MultipleGaussians = torch.zeros_like(RepeatedGaussian)
        for i in range(Nemitters):
            MultipleGaussians[:, i, :, :] = 1 / (2 * pi * stds[i] ** 2) * torch.pow(RepeatedGaussian[:, i, :, :],
                                                                                    1 / (stds[i] ** 2))
        # blur each emitter with slightly different gaussian
        images4D_blur = F.conv2d(img_4d, MultipleGaussians, padding=(2, 2))
        return images4D_blur


# ================================
# Cropping layer: keeps only the center part of the FOV to prevent unnecessary processing
# ==============================
class Croplayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, images4D):
        H = images4D.size(2)
        mid = int((H - 1) / 2)
        images4D_crop = images4D[:, :, mid - 20:mid + 21, mid - 20:mid + 21]
        return images4D_crop


class imgs4dto3d(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device

    def forward(self, images4D, xyz):
        Nbatch, Nemitters, H, W = images4D.shape[0], images4D.shape[1], images4D.shape[2], images4D.shape[3]
        img = torch.zeros((Nbatch, 1, 200, 200)).type(torch.FloatTensor).to(self.device)
        #img.requires_grad_()
        for i in range(Nbatch):
            for j in range(Nemitters):
                x = int(xyz[i, j, 0])
                y = int(xyz[i, j, 1])
                img[i, 0, x - 15:x + 16, y - 15: y + 16] += images4D[i, j]
        return img


class poisson_noise_approx(nn.Module):
    def __init__(self, device, Nimgs, conv3d):
        super().__init__()
        self.H, self.W = 200, 200
        self.device = device
        self.mean = 3e8
        self.std = 2e8
        self.Nimgs = Nimgs
        self.conv3d = conv3d

    def forward(self, input):
        # number of images
        Nbatch = input.size(0)
        # approximate the poisson noise using CLT and reparameterization
        input = input + 1e5 + (self.std * torch.randn(input.size()) + self.mean).type(torch.FloatTensor).to(self.device)
        input[input <= 0] = 0
        if self.conv3d == True:
            input_poiss = input + torch.tensor(100) * torch.sqrt(input) * torch.randn(Nbatch, 1, self.Nimgs, self.H, self.W).type(
                torch.FloatTensor).to(self.device)
        else:
            input_poiss = input + torch.tensor(100) * torch.sqrt(input) * torch.randn(Nbatch, self.Nimgs, self.H, self.W).type(
                torch.FloatTensor).to(self.device)
        
        # if torch.isnan(input_poiss).any():
        #     print('yes')

        # result
        return input_poiss


# Overall noise layer
class NoiseLayer(nn.Module):
    def __init__(self, device, Nimgs, conv3d):
        super().__init__()
        self.poiss = poisson_noise_approx(device, Nimgs, conv3d)
        self.unif_bg = 100

    def forward(self, input):
        inputb = input + self.unif_bg
        inputb_poiss = self.poiss(inputb)
        # if torch.isnan(inputb).any():
        #     print('yes')
        return inputb_poiss


class Normalize01(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, result_noisy):
        Nbatch = result_noisy.size(0)
        result_noisy_01 = torch.zeros_like(result_noisy)
        #min_val = (result_noisy[0, 0, :, :]).min()
        min_val = 0
        #max_val = (result_noisy[:, :, :, :]).max()
        #print(max_val)
        max_val = 4e9
        # if torch.isnan(result_noisy).any():
        #     print('yes')
        result_noisy[result_noisy <= 10] = 1
        result_noisy[result_noisy >= max_val] = max_val
        # for i in range(Nbatch):

        #     result_noisy_01[i, :, :, :] = (result_noisy[i, :, :, :] - min_val) / (max_val - min_val)
        result_noisy_01 = (result_noisy) / (max_val)
        return result_noisy_01


# ==================================================
# Physical encoding layer, from 3D to 2D:
# this layer takes in the learnable parameter "mask"
# and output the resulting 2D image corresponding to the emitters location.
# ===================================================

class PhysicalLayer(nn.Module):
    def __init__(self, config):
        super(PhysicalLayer, self).__init__()
        #unpack the config
        self.config = config
        self.bfp_dir = config["bfp_dir"]
        N =  config['N']
        self.px = config['px']  #the pixel size used
        self.wavelength = config['wavelength']
        self.focal_length = config['focal_length']
        psf_width_pixels = config['psf_width_pixels']
        psf_edge_remove = config['psf_edge_remove']
        laser_beam_FWHC = config['laser_beam_FWHC']
        refractive_index = config['refractive_index']
        max_defocus = config['max_defocus']
        image_volume = config['image_volume']
        max_intensity = config['max_intensity']
        psf_keep_radius = config['psf_keep_radius']
        device = config['device']
        self.lens_approach = config['lens_approach']
        self.device = device
        self.psf_keep_radius = psf_keep_radius
        self.N = N # the size of the FOV in pixels
        self.max_intensity = torch.tensor(max_intensity)
        self.counter = 0
        #self.power_2 = config['power_2']
        #self.pad_to_power_2 = self.power_2-N
        self.pad = 500
        self.datetime = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.Nimgs = config.get('Nimgs', 1)
        self.conv3d = config.get('conv3d', False)
        if self.Nimgs % 2 == 0:
            raise ValueError('Nimgs must be odd')
        

        # to transer the physical size of the imaging volume
        self.image_volume_um = image_volume

        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        x = list(range(-N // 2, N // 2))
        y = list(range(-N // 2, N // 2))
        [X, Y] = np.meshgrid(x, y)
        X = X * self.px
        Y = Y * self.px
        
        xx = list(range(-N + 1, N + 1))
        yy = list(range(-N + 1, N + 1))
        [XX, YY] = np.meshgrid(xx, yy)
        XX = XX * self.px
        YY = YY * self.px

        
        # initialize phase mask
        self.incident_gaussian = 1 * np.exp(-(np.square(X) + np.square(Y)) / (2 * laser_beam_FWHC ** 2))
        self.incident_gaussian = torch.from_numpy(self.incident_gaussian).type(torch.FloatTensor).to(device)
        

        C1 = (np.pi / (self.wavelength * self.focal_length) * (np.square(X) + np.square(Y))) % (
            2 * np.pi)  # lens function lens as a phase transformer
        self.B1 = np.exp(-1j * C1)
        self.B1 = torch.from_numpy(self.B1).type(torch.cfloat).to(device)
    
        # need to check if it relates to air or not added refractive index
        Q1 = np.exp(1j * (np.pi * refractive_index / (self.wavelength * self.focal_length)) * (
                    np.square(XX) + np.square(YY)))  # Fresnel diffraction equation at distance = focal length
        self.Q1 = torch.from_numpy(Q1).type(torch.cfloat).to(device)

        # angular specturm
        k = 2 * refractive_index * np.pi / self.wavelength
        self.k = k
        phy_x = N * self.px  # physical width (meters)
        phy_y = N * self.px  # physical length (meters)
        obj_size = [N, N]
        # generate meshgrid
        Fs_x = obj_size[1] / phy_x
        Fs_y = obj_size[0] / phy_y
        dFx = Fs_x / obj_size[1]
        dFy = Fs_y / obj_size[0]
        Fx = np.arange(-Fs_x / 2, Fs_x / 2, dFx)
        Fy = np.arange(-Fs_y / 2, Fs_y / 2, dFy)
        # alpha and beta (wavenumber components) 
        alpha = refractive_index * self.wavelength * Fx
        beta = refractive_index * self.wavelength * Fy
        [ALPHA, BETA] = np.meshgrid(alpha, beta)
        # go over and make sure that it is not complex
        gamma_cust = np.zeros_like(ALPHA)
        for i in range(len(ALPHA)):
            for j in range(len(ALPHA[0])):
                if 1 - np.square(ALPHA[i][j]) - np.square(BETA[i][j]) > 0:
                    gamma_cust[i, j] = np.sqrt(1 - np.square(ALPHA[i][j]) - np.square(BETA[i][j]))
        self.gamma_cust = torch.from_numpy(gamma_cust).type(torch.FloatTensor).to(device)

        # read defocus images
        self.imgs = []
        #Cut the PSF images at different planes
        for z in range(0, max_defocus):
            img = skimage.io.imread('beads_img_defocus/z' + str(z).zfill(2) + '.tiff')
            center_img = len(img)//2

            self.imgs.append(img[center_img + -psf_keep_radius:center_img+psf_keep_radius+1,\
                                       center_img + -psf_keep_radius:center_img+psf_keep_radius+1])
            #Debug
            #plt.imshow(img[center_img + -psf_keep_radius:center_img+psf_keep_radius+1,\
            #                           center_img + -psf_keep_radius:center_img+psf_keep_radius+1])
            #plt.axis('off')  # Turn off axis labels
            #plt.show()



        self.blur = BlurLayer(device)
        self.crop = Croplayer()
        self.img4dto3d = imgs4dto3d(device)
        self.noise = NoiseLayer(device, self.Nimgs, self.conv3d)
        self.norm01 = Normalize01()

    def forward(self, mask_param, xyz, Nphotons):

        Nbatch, Nemitters = xyz.shape[0], xyz.shape[1]
        
        if self.lens_approach == 'fresnel':
            mask_param = self.incident_gaussian * torch.exp(1j * mask_param)
            mask_param = mask_param[None, None, :]
            #multiply the mask with the phase function of the lens (B1)
            B1 = self.B1 * mask_param

            # AG - Need to check if the FFT will be faster if padded to 1024
            # pad_to_power_2 = NextPowerOfTwo(B1.shape[0])-B1.shape[0]
            
            E1 = F.pad(B1, (self.N//2, self.N//2, self.N//2, self.N//2), 'constant', 0)
            # Goodman book equation 4-14, convolution method - lens kernel (Q1) and the image after mask and lens function (E2)
            E2 = torch.fft.ifftshift(torch.fft.ifft2(torch.fft.fft2(E1) * torch.fft.fft2(self.Q1)))
            output_layer = E2[:, :, self.N // 2:3 * self.N // 2, self.N // 2:3 * self.N // 2]
        
        elif self.lens_approach == 'fourier':
            Ta = torch.exp(1j * mask_param) # amplitude transmittance (in our case the slm reflectance)
            Ta = Ta[None, None, :] 
            Uo = self.incident_gaussian * Ta # light directly behiund the SLM (or in our case reflected from the SLM)
            #Uo = Uo[None, None, :] # not sure why mani did this?
            Uo_pad = F.pad(Uo, (self.N//2, self.N//2, self.N//2, self.N//2), 'constant', 0) # padded to interpolate with fft
            Fo = torch.fft.fftshift(torch.fft.fft2(Uo_pad)) # fourier spectrum of the light directly after the SLM
            # can ignore a constant phase factor from goodman 1/(1j * self.wavelength * self.focal_length)
            Uf = Fo # light at the back focal plane of the lens   
            output_layer = Uf[:, :, self.N // 2:3 * self.N // 2, self.N // 2:3 * self.N // 2]
        
        elif self.lens_approach == 'convolution':
            Ta = torch.exp(1j * mask_param) # amplitude transmittance (in our case the slm reflectance)
            Ta = Ta[None, None, :]
            Uo = self.incident_gaussian * Ta # light directly behiund the SLM (or in our case reflected from the SLM)
            Uo_pad = F.pad(Uo, (0, self.N, 0, self.N), 'constant', 0) # padded to interpolate with fft
            #Fo = torch.fft.fftshift(torch.fft.fft2(Uo_pad)) # fourier spectrum of the light directly after the SLM
            #Fl = Fo * self.H # propogated through free space from the SLM to a plane directly incident on the lens
            Ul = torch.fft.ifft2(torch.fft.fft2(Uo_pad) * torch.fft.fft2(self.Q1)) # light directly infront of the lens
            Ul_cropped = Ul[:, :, -self.N:, -self.N:]
            #Ul_cropped = Ul[-self.N:, -self.N:]
            Ul_prime = Ul_cropped * self.B1 # light after the lens
            Ul_prime_pad = F.pad(Ul_prime, (0, self.N, 0, self.N), 'constant', 0) # padded to interpolate with fft
            Uf = torch.fft.ifft2(torch.fft.fft2(Ul_prime_pad) * torch.fft.fft2(self.Q1)) # light at the back focal plane of the lens   
            output_layer = Uf[:, :, -self.N:, -self.N:]
            #output_layer = Uf[-self.N:, -self.N:]
        
        else:
            raise ValueError('lens approach not supported')
            
        if self.counter == 0 and not self.training:    
            save_output_layer(output_layer, self.bfp_dir, self.lens_approach, self.counter, self.datetime, self.config)
        self.counter += 1
        
        if self.conv3d == False:
            # make a 4D tensor to store the 2D images
            imgs3D = torch.zeros(Nbatch, self.Nimgs, self.image_volume_um[0], self.image_volume_um[0]).type(torch.FloatTensor).to(self.device)
            for l in range(self.Nimgs):
                for i in range(Nbatch):
                    for j in range(Nemitters):
                        # change x value to fit different field of view
                        x = xyz[i, j, 0].type(torch.LongTensor) - self.image_volume_um[0]//2
                        y = xyz[i, j, 1].type(torch.LongTensor)
                        z = xyz[i, j, 2].type(torch.LongTensor)

                        x_ori = xyz[i, j, 0].type(torch.LongTensor)
                        U1 = torch.fft.ifft2(
                            torch.fft.ifftshift(
                                torch.fft.fftshift(torch.fft.fft2(output_layer)) 
                                * torch.exp(1j * self.k * self.gamma_cust * x * self.px)
                                )
                            ) # should this 1e-6 be self.px?
                        U1 = torch.real(U1 * torch.conj(U1))

                        # Here we assume that the beam is being dithered up and down
                        intensity = torch.sum(U1[0, 0, :, int((self.N//2-1) + z)]) # if px is 1e-6 why multiply by 1e6?
                        imgs3D[i, l, x_ori - self.psf_keep_radius:x_ori + self.psf_keep_radius+1, y - self.psf_keep_radius: y + self.psf_keep_radius + 1] += torch.from_numpy(
                            self.imgs[
                                abs(
                                    z.item()-( l - (self.Nimgs//2))
                                    )].astype('float32')).type(torch.FloatTensor).to(self.device) * intensity

        elif self.conv3d == True and self.Nimgs > 1:
            #make a 5D tensor to store the 3D images
            imgs3D = torch.zeros(Nbatch, 1, self.Nimgs, self.image_volume_um[0], self.image_volume_um[0]).type(torch.FloatTensor).to(self.device)
            for l in range(self.Nimgs):
                for i in range(Nbatch):
                    for j in range(Nemitters):
                        # change x value to fit different field of view
                        x = xyz[i, j, 0].type(torch.LongTensor) - self.image_volume_um[0]//2
                        y = xyz[i, j, 1].type(torch.LongTensor)
                        z = xyz[i, j, 2].type(torch.LongTensor)

                        x_ori = xyz[i, j, 0].type(torch.LongTensor)
                        U1 = torch.fft.ifft2(
                            torch.fft.ifftshift(
                                torch.fft.fftshift(torch.fft.fft2(output_layer)) 
                                * torch.exp(1j * self.k * self.gamma_cust * x * self.px)
                                )
                            ) # should this 1e-6 be self.px?
                        U1 = torch.real(U1 * torch.conj(U1))

                        # Here we assume that the beam is being dithered up and down
                        intensity = torch.sum(U1[0, 0, :, int((self.N//2-1) + z)]) # if px is 1e-6 why multiply by 1e6?
                        imgs3D[i, 0, l, x_ori - self.psf_keep_radius:x_ori + self.psf_keep_radius+1, y - self.psf_keep_radius: y + self.psf_keep_radius + 1] += torch.from_numpy(
                            self.imgs[
                                abs(
                                    z.item()-( l - (self.Nimgs//2) )
                                    )].astype('float32')).type(torch.FloatTensor).to(self.device) * intensity

        else:
            raise ValueError('Nimgs must be > 1 to use conv3d')
        
        # #### AG seems not necessary, since you multiply the gamma_cust with 0
        #U1 = torch.fft.ifft2(torch.fft.ifftshift(
        #   torch.fft.fftshift(torch.fft.fft2(output_layer)) * torch.exp(1j * self.k * self.gamma_cust * 0)))
        #U1 = torch.real(U1 * torch.conj(U1))
        # #### AG

        # Go over the position of each bead and create the appropriate image
        # l = 0 the center of the bead volume is focused by the detection obj
        # l = 1 the focal plane is 1 unit further away from the detection obj
        # l = -1 the focal plane is 1 unit closer to the detection obj
        
        # need to check the normalization here
        imgs3D = imgs3D / self.max_intensity

        # Conditionally bypass noise addition during inference if skip_noise flag is set
        if self.config.get('skip_noise', False) and not self.training:
            result = self.norm01(imgs3D)
            return result
        else:
            result_noisy = self.noise(imgs3D)
            result_noisy01 = self.norm01(result_noisy)
            return result_noisy01
