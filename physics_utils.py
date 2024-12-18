# -*- coding: utf-8 -*-
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from math import pi
import torch.fft
import skimage.io



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
    def __init__(self):
        super().__init__()

    def forward(self, images4D, xyz):
        Nbatch, Nemitters, H, W = images4D.shape[0], images4D.shape[1], images4D.shape[2], images4D.shape[3]
        img = torch.zeros((Nbatch, 1, 200, 200)).type(torch.FloatTensor).to(device)
        #img.requires_grad_()
        for i in range(Nbatch):
            for j in range(Nemitters):
                x = int(xyz[i, j, 0])
                y = int(xyz[i, j, 1])
                img[i, 0, x - 15:x + 16, y - 15: y + 16] += images4D[i, j]
        return img


class poisson_noise_approx(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.H, self.W = 200, 200
        self.device = device
        self.mean = 3e8
        self.std = 2e8

    def forward(self, input):
        # number of images
        Nbatch = input.size(0)
        # approximate the poisson noise using CLT and reparameterization
        input = input + 1e5 + (self.std * torch.randn(input.size()) + self.mean).type(torch.FloatTensor).to(self.device)
        input[input <= 0] = 0
        input_poiss = input + torch.tensor(100) * torch.sqrt(input) * torch.randn(Nbatch, 1, self.H, self.W).type(
            torch.FloatTensor).to(self.device)
        # if torch.isnan(input_poiss).any():
        #     print('yes')

        # result
        return input_poiss


# Overall noise layer
class NoiseLayer(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.poiss = poisson_noise_approx(device)
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
        N =  config['N']
        px = config['px']
        wavelength = config['wavelength']
        focal_length = config['focal_length']
        psf_width_pixels = config['psf_width_pixels']
        psf_edge_remove = config['psf_edge_remove']
        laser_beam_FWHC = config['laser_beam_FWHC']
        refractive_index = config['refractive_index']
        max_defocus = config['max_defocus']
        image_volume = config['image_volume']
        max_intensity = config['max_intensity']
        psf_keep_radius = config['psf_keep_radius']
        device = config['device']

        self.device = device
        self.psf_keep_radius = psf_keep_radius
        self.px = px #the pixel size used
        self.N = N #the size of the FOV in pixels
        self.max_intensity = torch.tensor(max_intensity)

        # to transer the physical size of the imaging volume
        self.image_volume_um = image_volume


        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        x = list(range(-N // 2, N // 2))
        y = list(range(-N // 2, N // 2))
        [X, Y] = np.meshgrid(x, y)
        X = X * px
        Y = Y * px
        C1 = (np.pi / (wavelength * focal_length) * (np.square(X) + np.square(Y))) % (
                    2 * np.pi)  # lens function lens as a phase transformer
        self.B1 = np.exp(-1j * C1)
        self.B1 = torch.from_numpy(self.B1).type(torch.cfloat).to(device)

        # initialize phase mask
        mask_init = 1 * np.exp(-(np.square(X) + np.square(Y)) / (2 * laser_beam_FWHC ** 2))
        self.mask_real = torch.from_numpy(mask_init).type(torch.FloatTensor).to(device)

        xx = list(range(-N + 1, N + 1))
        yy = list(range(-N + 1, N + 1))
        [XX, YY] = np.meshgrid(xx, yy)
        XX = XX * px
        YY = YY * px
        #need to check if it relates to air or not added refractive index
        Q1 = np.exp(1j * (np.pi * refractive_index / (wavelength * focal_length)) * (
                    np.square(XX) + np.square(YY)))  # Fresnel diffraction equation at distance = focal length
        self.Q1 = torch.from_numpy(Q1).type(torch.cfloat).to(device)

        # angular specturm
        k = 2 * refractive_index * np.pi / wavelength
        self.k = k
        phy_x = N * px  # physical width (meters)
        phy_y = N * px  # physical length (meters)
        obj_size = [N, N]
        # generate meshgrid
        Fs_x = obj_size[1] / phy_x
        Fs_y = obj_size[0] / phy_y
        dFx = Fs_x / obj_size[1]
        dFy = Fs_y / obj_size[0]
        Fx = np.arange(-Fs_x / 2, Fs_x / 2, dFx)
        Fy = np.arange(-Fs_y / 2, Fs_y / 2, dFy)
        # alpha and beta (wavenumber components) 
        alpha = refractive_index * wavelength * Fx
        beta = refractive_index * wavelength * Fy
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
            range_crop_psf = range(-(psf_width_pixels - 1)//2 - psf_edge_remove,(psf_width_pixels - 1)//2 + psf_edge_remove+1)
            self.imgs.append(img[range_crop_psf, range_crop_psf])


        self.blur = BlurLayer(device)
        self.crop = Croplayer()
        self.img4dto3d = imgs4dto3d()
        self.noise = NoiseLayer(device)
        self.norm01 = Normalize01()

    def forward(self, mask_param, xyz):

        Nbatch, Nemitters = xyz.shape[0], xyz.shape[1]
        mask_param = self.mask_real * torch.exp(1j * mask_param)
        mask_param = mask_param[None, None, :]
        #multiply the mask with the phase function of the lens (B1)
        B1 = self.B1 * mask_param

        # AG - Need to check if the FFT will be faster if padded to 1024
        # pad_to_power_2 = NextPowerOfTwo(B1.shape[0])-B1.shape[0]
        pad_to_power_2 = 500
        E1 = F.pad(B1, (pad_to_power_2/2, pad_to_power_2/2, pad_to_power_2/2, pad_to_power_2/2), 'constant', 0)
        # Goodman book equation 4-14, convolution method - lens kernel (Q1) and the image after mask and lens function (E2)
        E2 = torch.fft.ifftshift(torch.fft.ifft2(torch.fft.fft2(E1) * torch.fft.fft2(self.Q1)))

        output_layer = E2[:, :, N // 2:3 * N // 2, N // 2:3 * N // 2]

        imgs3D = torch.zeros(Nbatch, 1, self.image_volume_um[0], self.image_volume_um[0]).type(torch.FloatTensor).to(self.device)

        # #### AG seems not necessary, since you multiply the gamma_cust with 0
        #U1 = torch.fft.ifft2(torch.fft.ifftshift(
        #   torch.fft.fftshift(torch.fft.fft2(output_layer)) * torch.exp(1j * self.k * self.gamma_cust * 0)))
        #U1 = torch.real(U1 * torch.conj(U1))
        # #### AG

        # Go over the position of each bead and create the appropriate image
        for i in range(Nbatch):
            for j in range(Nemitters):
                # change x value to fit different field of view
                x = xyz[i, j, 0].type(torch.LongTensor) - self.image_volume_um[0]//2
                y = xyz[i, j, 1].type(torch.LongTensor)
                z = xyz[i, j, 2].type(torch.LongTensor)

                x_ori = xyz[i, j, 0].type(torch.LongTensor)
                U1 = torch.fft.ifft2(torch.fft.ifftshift(torch.fft.fftshift(torch.fft.fft2(output_layer)) * torch.exp(
                    1j * self.k * self.gamma_cust * x * 1e-6)))
                U1 = torch.real(U1 * torch.conj(U1))

                #Here we assume that the beam is being dithered up and down
                intensity = torch.sum(U1[0, 0, :, ((self.N*self.px*1e6)//2-1) + z])
                imgs3D[i, 0, x_ori - self.psf_keep_radius:x_ori + self.psf_keep_radius+1,\
                y - self.psf_keep_radius: y + self.psf_keep_radius + 1] += torch.from_numpy(
                    self.imgs[abs(z.item())].astype('float32')).type(torch.FloatTensor).to(self.device) * intensity

        imgs3D = imgs3D / self.max_intensity

        result_noisy = self.noise(imgs3D, self.device)
        result_noisy01 = self.norm01(result_noisy)
        return result_noisy01
