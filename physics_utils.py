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
            #input_poiss = input + torch.tensor(100) * torch.sqrt(input) * torch.randn(Nbatch, self.Nimgs, self.H, self.W).type(
            #    torch.FloatTensor).to(self.device)
            input_poiss = input + torch.tensor(100) * torch.sqrt(input) * torch.randn(Nbatch, 1, self.H, self.W).type(
                torch.FloatTensor).to(self.device) # same noise applied to each z depth
        
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
        self.refractive_index = config['refractive_index']
        max_defocus = config['max_defocus']
        image_volume = config['image_volume']
        psf_keep_radius = config['psf_keep_radius']
        device = config['device']
        self.lens_approach = config['lens_approach']
        if self.lens_approach == 'fresnel':
            max_intensity = config.get('max_intensity_fresnel', 5.0e+4)
        elif self.lens_approach == 'convolution':
            max_intensity = config.get('max_intensity_conv', 8.0e+10)
        else:
            max_intensity = config.get('max_intensity_conv', 8.0e+10)    
        self.device = device
        self.psf_keep_radius = psf_keep_radius
        self.N = N # the size of the FOV in pixels
        self.max_intensity = torch.tensor(max_intensity)
        self.counter = 0
        self.focal_length_2 = config['focal_length_2']  # for 4f approach
       
            
        #self.power_2 = config['power_2']
        #self.pad_to_power_2 = self.power_2-N
        self.pad = 500
        self.datetime = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.conv3d = config.get('conv3d', False)
        self.aperature = config.get('aperature', False)
        #self.z_spacing = config.get('z_spacing', 0)
        #self.z_img_mode = config.get('z_img_mode', 'edgecenter')
        #if self.z_img_mode == 'everyother':
            #self.z_depth_list = list(range(0,self.Nimgs,2))
        #if self.z_img_mode == 'edgecenter' and self.z_spacing > 0:
        #    self.z_depth_list = [-self.z_spacing, 0, self.z_spacing]
        #else:
        #    self.z_depth_list = list(range(-self.z_spacing,self.z_spacing+1))
        
        self.Nimgs = config['Nimgs']
        self.z_depth_list = config['z_depth_list']

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
        self.XX = XX * self.px
        self.YY = YY * self.px

        
        # initialize phase mask
        self.incident_gaussian = 1 * np.exp(-(np.square(X) + np.square(Y)) / (2 * laser_beam_FWHC ** 2))
        self.incident_gaussian = torch.from_numpy(self.incident_gaussian).type(torch.FloatTensor).to(device)
        

        C1 = (np.pi / (self.wavelength * self.focal_length) * (np.square(X) + np.square(Y))) % (
            2 * np.pi)  # lens function lens as a phase transformer
        self.B1 = np.exp(-1j * C1)
        self.B1 = torch.from_numpy(self.B1).type(torch.cfloat).to(device)
    
        # need to check if it relates to air or not added refractive index
        Q1 = np.exp(1j * (np.pi * self.refractive_index / (self.wavelength * self.focal_length)) * (
                    np.square(XX) + np.square(YY)))  # Fresnel diffraction equation at distance = focal length
        self.Q1 = torch.from_numpy(Q1).type(torch.cfloat).to(device)

        # --- Lens 2 (B2) and Propagation Kernel 2 (Q2) ---
        C2 = (np.pi / (self.wavelength * self.focal_length_2) * (np.square(X) + np.square(Y))) % (2 * np.pi)
        self.B2 = np.exp(-1j * C2)
        self.B2 = torch.from_numpy(self.B2).type(torch.cfloat).to(device)

        # Q2 for propagation over self.focal_length_2
        Q2_val = np.exp(1j * (np.pi * self.refractive_index / (self.wavelength * self.focal_length_2)) * (np.square(self.XX) + np.square(self.YY)))
        self.Q2 = torch.from_numpy(Q2_val).type(torch.cfloat).to(device)
        
        # angular specturm
        k = 2 * self.refractive_index * np.pi / self.wavelength
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
        alpha = self.refractive_index * self.wavelength * Fx
        beta = self.refractive_index * self.wavelength * Fy
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

    def circular_aperature(self, arr):
        """
        Sets elements outside a centered circle to zero for a PyTorch tensor.

        Args:
        arr: A 2D square PyTorch tensor.

        Returns:
        A new PyTorch tensor with elements outside the circle set to zero.
        """
        h = arr.shape[0]
        if arr.shape[1] != h:
            raise ValueError("Input tensor must be square.")

        center_x = (h - 1) / 2
        center_y = (h - 1) / 2
        radius = h / 2

        # Create a meshgrid of coordinates
        x = torch.arange(h)
        y = torch.arange(h)
        xx, yy = torch.meshgrid(x, y, indexing='ij')  # Use indexing='ij' for correct meshgrid

        # Calculate the distance from each point to the center
        distances = torch.sqrt((xx - center_x)**2 + (yy - center_y)**2)

        # Create a mask where True indicates points inside the circle
        mask = distances <= radius

        # Apply the mask to the array
        result = arr * mask.to(self.device)

        return result

    def against_lens(self, mask_param):
        Ta = torch.exp(1j * mask_param) # amplitude transmittance (in our case the slm reflectance)
        Ta = Ta[None, None, :] 
        Uo = self.incident_gaussian * Ta # light directly behind the SLM (or in our case reflected from the SLM)
        #Uo = Uo[None, None, :] # not sure why mani did this?
        Uo_pad = F.pad(Uo, (self.N//2, self.N//2, self.N//2, self.N//2), 'constant', 0) # padded to interpolate with fft
        Fo = torch.fft.fftshift(torch.fft.fft2(Uo_pad)) # fourier spectrum of the light directly after the SLM
        # can ignore a constant phase factor from goodman 1/(1j * self.wavelength * self.focal_length)
        Uf = Fo # light at the back focal plane of the lens   
        output_layer = Uf[:, :, self.N // 2:3 * self.N // 2, self.N // 2:3 * self.N // 2]
        return output_layer

    def fourier_lens(self, mask_param):
        Ta = torch.exp(1j * mask_param) # amplitude transmittance (in our case the slm reflectance)
        Ta = Ta[None, None, :]
        Uo = self.incident_gaussian * Ta # light directly behind the SLM (or in our case reflected from the SLM)
        Uo_pad = F.pad(Uo, (0, self.N, 0, self.N), 'constant', 0) # padded to interpolate with fft
        Ul = torch.fft.ifft2(torch.fft.fft2(Uo_pad) * torch.fft.fft2(self.Q1)) # light directly infront of the lens
        Ul_cropped = Ul[:, :, -self.N:, -self.N:]
        if self.aperature == True:
            Ul_cropped = self.circular_aperature(Ul_cropped,self.device)
        Ul_prime = Ul_cropped * self.B1 # light after the lens
        Ul_prime_pad = F.pad(Ul_prime, (0, self.N, 0, self.N), 'constant', 0) # padded to interpolate with fft
        Uf = torch.fft.ifft2(torch.fft.fft2(Ul_prime_pad) * torch.fft.fft2(self.Q1)) # light at the back focal plane of the lens   
        output_layer = Uf[:, :, -self.N:, -self.N:]
        return output_layer

    def lensless(self, mask_param):
        Ta = torch.exp(1j * mask_param) # amplitude transmittance (in our case the slm reflectance)
        Ta = Ta[None, None, :]
        Uo = self.incident_gaussian * Ta # light directly behind the SLM (or in our case reflected from the SLM)
        output_layer = Uo
        return output_layer

    def fourf(self, mask_param):
        Ta = torch.exp(1j * mask_param) # amplitude transmittance (in our case the slm reflectance)
        Ta = Ta[None, None, :]
        Uo = self.incident_gaussian * Ta # light directly behind the SLM (or in our case reflected from the SLM)
        Uo_pad = F.pad(Uo, (0, self.N, 0, self.N), 'constant', 0) # padded to interpolate with fft
        #Fo = torch.fft.fftshift(torch.fft.fft2(Uo_pad)) # fourier spectrum of the light directly after the SLM
        #Fl = Fo * self.H # propogated through free space from the SLM to a plane directly incident on the lens
        Ul = torch.fft.ifft2(torch.fft.fft2(Uo_pad) * torch.fft.fft2(self.Q1)) # light directly incident of the lens
        Ul_cropped = Ul[:, :, -self.N:, -self.N:]
        if self.aperature == True:
            Ul_cropped = self.circular_aperature(Ul_cropped,self.device)
        #Ul_cropped = Ul[-self.N:, -self.N:]
        Ul_prime = Ul_cropped * self.B1 # light after the lens
        Ul_prime_pad = F.pad(Ul_prime, (0, self.N, 0, self.N), 'constant', 0) # padded to interpolate with fft
        Uf = torch.fft.ifft2(torch.fft.fft2(Ul_prime_pad) * torch.fft.fft2(self.Q1)) # light at the back focal plane of the lens   
        Uf_cropped = Uf[:, :, -self.N:, -self.N:]
        # continue to progoate the light to and through the 2nd lens ot the bfp of the 2nd lens
        
        # Extract the cropped Fourier plane field (Uf)
        # Assuming Uf has 2*N x 2*N after padding, so we take the central N x N for processing
        #Uf_cropped = Uf[:, :, self.N:2*self.N, self.N:2*self.N]

        # 1. Propagate from BFP of L1 (Fourier plane) to FFP of L2 (distance f1)
        U_before_L2_pad = F.pad(Uf_cropped, (0, self.N, 0, self.N), 'constant', 0) # Pad for FFT
        U_before_L2 = torch.fft.ifft2(torch.fft.fft2(U_before_L2_pad) * torch.fft.fft2(self.Q1)) # Uses Q1 (for f1)

        # 2. Apply the phase transformation of the second lens (self.B2)
        U_after_L2_cropped = U_before_L2[:, :, -self.N:, -self.N:] # Crop after propagation
        U_after_L2_prime = U_after_L2_cropped * self.B2 # Light after the 2nd lens

        # 3. Propagate from the 2nd lens to its BFP (the output plane of the 4f system) (distance f2)
        U_after_L2_prime_pad = F.pad(U_after_L2_prime, (0, self.N, 0, self.N), 'constant', 0) # Pad for FFT
        U_output_4f = torch.fft.ifft2(torch.fft.fft2(U_after_L2_prime_pad) * torch.fft.fft2(self.Q2)) # Uses Q2 (for f2)

        # Final output of the 4f system (cropped BFP of the 2nd lens)
        output_layer = U_output_4f[:, :, -self.N:, -self.N:]
        return output_layer

    def angular_spectrum_propagation(self, output_layer, z):
        """
        Angular spectrum propagation for a given output_layer and z pixel distance.

        Args:
            output_layer (torch.Tensor): The complex field to propagate.
            z (float or int): The z pixel distance for propagation.

        Returns:
            torch.Tensor: The propagated intensity (real-valued).
        """
        U1 = torch.fft.ifft2(
            torch.fft.ifftshift(
                torch.fft.fftshift(torch.fft.fft2(output_layer)) 
                * torch.exp(1j * self.k * self.gamma_cust * z * self.px)
            )
        )
        U1 = torch.real(U1 * torch.conj(U1))
        return U1
    
    def fresnel_propagation(self, input_img, z):
        """
        Fresnel propagation for a given output_layer and z pixel distance.

        Args:
            output_layer (torch.Tensor): The complex field to propagate.
            z (float or int): The z pixel distance for propagation.

        Returns:
            torch.Tensor: The propagated intensity (real-valued).
        """
        Q = np.exp(1j * (np.pi * self.refractive_index / (self.wavelength * z)) * (
                    np.square(self.XX) + np.square(self.YY)))  # Fresnel diffraction equation
        Q = torch.from_numpy(Q).type(torch.cfloat).to(self.device)

        #Q1 = np.exp(1j * (np.pi * refractive_index / (self.wavelength * self.focal_length)) * (
        #            np.square(XX) + np.square(YY)))  # Fresnel diffraction equation at distance = focal length
        #self.Q1 = torch.from_numpy(Q1).type(torch.cfloat).to(device)
        Uo_pad = F.pad(input_img, (0, self.N, 0, self.N), 'constant', 0) # padded to interpolate with fft
        U1 = torch.fft.ifft2(torch.fft.fft2(Uo_pad) * torch.fft.fft2(Q)) # light directly incident of the lens
        U1_cropped = U1[-self.N:, -self.N:]
        U1_intensity = torch.real(U1_cropped * torch.conj(U1_cropped))
        return U1_intensity
    
    @staticmethod
    def normalize_to_uint16(img: np.ndarray) -> np.ndarray:
        """
        Normalizes a numpy array to the uint16 range [0, 65535].
        
        Args:
            img (np.ndarray): The input image array.
            
        Returns:
            np.ndarray: The normalized image as a uint16 array.
        """
        # if tesnor convert to numpy
        if isinstance(img, torch.Tensor):
            img = img.cpu().numpy()
        # Ensure the image is in float format for normalization
        img_float = img.astype(np.float32)
        img_float -= img_float.min()
        max_val = img_float.max()
        if max_val > 0:
            img_float /= max_val
        return (img_float * 65535).astype(np.uint16)
    
    def generate_beam_cross_section(self, initial_field: np.ndarray, output_folder: str, z_range_px: tuple, y_slice_px: tuple, asm: bool = True) -> np.ndarray:
        """
        Generates a 2D cross-section of the beam by stacking 1D slices along the z-axis.
        
        Args:
            initial_field (np.ndarray): The complex field at z=0.
            output_folder (str): Folder to save 2D intensity profiles at each z-step.
            z_range_px (tuple): (min, max) propagation distance in pixels.
            y_slice_px (tuple): (min, max) slice of the y-axis in pixels.
            
        Returns:
            np.ndarray: A 2D array representing the beam's cross-section (ZY plane).
        """
        os.makedirs(output_folder, exist_ok=True)
        z_min_px, z_max_px, z_step = z_range_px
        y_min_px, y_max_px = y_slice_px
        
        num_z_steps = (z_max_px - z_min_px)// z_step
        # Initialize the cross-section profile array accounting for step size

        cross_section_profile = np.zeros((num_z_steps, y_max_px - y_min_px))

        print(f"Generating beam cross-section for {num_z_steps} slices...")
        for i, z_px in enumerate(range(z_min_px, z_max_px, z_step)):
            if z_px == 0:
                intensity_at_z = initial_field
            else:
                # Propagate field and get intensity
                if asm:
                    intensity_at_z = self.angular_spectrum_propagation(initial_field, z_px)
                else:
                    intensity_at_z = self.fresnel_propagation(initial_field, z_px)

            # Extract the central 1D slice along the y-axis
            center_row_idx = intensity_at_z.shape[0] // 2
            center_col_idx = intensity_at_z.shape[1] // 2
            cross_section_profile[i,:] = intensity_at_z[center_row_idx, center_col_idx + y_min_px : center_col_idx + y_max_px]

            # Save the full 2D intensity profile at the current z-step
            save_path = os.path.join(output_folder, f'intensity_z_{i:04d}.tiff')
            skimage.io.imsave(save_path, self.normalize_to_uint16(intensity_at_z))
            
        print("Cross-section generation complete.")
        return self.normalize_to_uint16(cross_section_profile)


    def forward(self, mask_param, xyz):

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
        
        elif self.lens_approach == 'fourier' or self.lens_approach == 'against_lens':
            output_layer = self.against_lens(mask_param)
            
        elif self.lens_approach == 'fourier_lens' or self.lens_approach == 'convolution':
            output_layer = self.fourier_lens(mask_param)
            """ Ta = torch.exp(1j * mask_param) # amplitude transmittance (in our case the slm reflectance)
            Ta = Ta[None, None, :]
            Uo = self.incident_gaussian * Ta # light directly behind the SLM (or in our case reflected from the SLM)
            Uo_pad = F.pad(Uo, (0, self.N, 0, self.N), 'constant', 0) # padded to interpolate with fft
            #Fo = torch.fft.fftshift(torch.fft.fft2(Uo_pad)) # fourier spectrum of the light directly after the SLM
            #Fl = Fo * self.H # propogated through free space from the SLM to a plane directly incident on the lens
            Ul = torch.fft.ifft2(torch.fft.fft2(Uo_pad) * torch.fft.fft2(self.Q1)) # light directly infront of the lens
            Ul_cropped = Ul[:, :, -self.N:, -self.N:]
            if self.aperature == True:
                Ul_cropped = circular_aperature(Ul_cropped,self.device)
            #Ul_cropped = Ul[-self.N:, -self.N:]
            Ul_prime = Ul_cropped * self.B1 # light after the lens
            Ul_prime_pad = F.pad(Ul_prime, (0, self.N, 0, self.N), 'constant', 0) # padded to interpolate with fft
            Uf = torch.fft.ifft2(torch.fft.fft2(Ul_prime_pad) * torch.fft.fft2(self.Q1)) # light at the back focal plane of the lens   
            output_layer = Uf[:, :, -self.N:, -self.N:]
            #output_layer = Uf[-self.N:, -self.N:] """
            
        elif self.lens_approach == 'lensless':
            output_layer = self.lensless(self, mask_param)
            #Ta = torch.exp(1j * mask_param) # amplitude transmittance (in our case the slm reflectance)
            #Ta = Ta[None, None, :]
            #Uo = self.incident_gaussian * Ta # light directly behind the SLM (or in our case reflected from the SLM)
            """ Uo_pad = F.pad(Uo, (0, self.N, 0, self.N), 'constant', 0) # padded to interpolate with fft
            #Fo = torch.fft.fftshift(torch.fft.fft2(Uo_pad)) # fourier spectrum of the light directly after the SLM
            #Fl = Fo * self.H # propogated through free space from the SLM to a plane directly incident on the lens
            Ul = torch.fft.ifft2(torch.fft.fft2(Uo_pad) * torch.fft.fft2(self.Q1)) # light directly infront of the lens
            Ul_cropped = Ul[:, :, -self.N:, -self.N:]
            if self.aperature == True:
                Ul_cropped = circular_aperature(Ul_cropped,self.device)
            #Ul_cropped = Ul[-self.N:, -self.N:]
            Ul_prime = Ul_cropped * self.B1 # light after the lens
            Ul_prime_pad = F.pad(Ul_prime, (0, self.N, 0, self.N), 'constant', 0) # padded to interpolate with fft
            Uf = torch.fft.ifft2(torch.fft.fft2(Ul_prime_pad) * torch.fft.fft2(self.Q1)) # light at the back focal plane of the lens   
            output_layer = Uf[:, :, -self.N:, -self.N:] """

            
        elif self.lens_approach == '4f':
            if self.focal_length_2 <= 0:
                raise ValueError('focal_length_2 must be greater than 0 for 4f approach')
            
            output_layer = self.fourf(mask_param)
            
            """ Ta = torch.exp(1j * mask_param) # amplitude transmittance (in our case the slm reflectance)
            Ta = Ta[None, None, :]
            Uo = self.incident_gaussian * Ta # light directly behind the SLM (or in our case reflected from the SLM)
            Uo_pad = F.pad(Uo, (0, self.N, 0, self.N), 'constant', 0) # padded to interpolate with fft
            #Fo = torch.fft.fftshift(torch.fft.fft2(Uo_pad)) # fourier spectrum of the light directly after the SLM
            #Fl = Fo * self.H # propogated through free space from the SLM to a plane directly incident on the lens
            Ul = torch.fft.ifft2(torch.fft.fft2(Uo_pad) * torch.fft.fft2(self.Q1)) # light directly incident of the lens
            Ul_cropped = Ul[:, :, -self.N:, -self.N:]
            if self.aperature == True:
                Ul_cropped = circular_aperature(Ul_cropped,self.device)
            #Ul_cropped = Ul[-self.N:, -self.N:]
            Ul_prime = Ul_cropped * self.B1 # light after the lens
            Ul_prime_pad = F.pad(Ul_prime, (0, self.N, 0, self.N), 'constant', 0) # padded to interpolate with fft
            Uf = torch.fft.ifft2(torch.fft.fft2(Ul_prime_pad) * torch.fft.fft2(self.Q1)) # light at the back focal plane of the lens   
            Uf_cropped = Uf[:, :, -self.N:, -self.N:]
            # continue to progoate the light to and through the 2nd lens ot the bfp of the 2nd lens
            
            # Extract the cropped Fourier plane field (Uf)
            # Assuming Uf has 2*N x 2*N after padding, so we take the central N x N for processing
            #Uf_cropped = Uf[:, :, self.N:2*self.N, self.N:2*self.N]

            # 1. Propagate from BFP of L1 (Fourier plane) to FFP of L2 (distance f1)
            U_before_L2_pad = F.pad(Uf_cropped, (0, self.N, 0, self.N), 'constant', 0) # Pad for FFT
            U_before_L2 = torch.fft.ifft2(torch.fft.fft2(U_before_L2_pad) * torch.fft.fft2(self.Q1)) # Uses Q1 (for f1)

            # 2. Apply the phase transformation of the second lens (self.B2)
            U_after_L2_cropped = U_before_L2[:, :, -self.N, -self.N] # Crop after propagation
            U_after_L2_prime = U_after_L2_cropped * self.B2 # Light after the 2nd lens

            # 3. Propagate from the 2nd lens to its BFP (the output plane of the 4f system) (distance f2)
            U_after_L2_prime_pad = F.pad(U_after_L2_prime, (0, self.N, 0, self.N), 'constant', 0) # Pad for FFT
            U_output_4f = torch.fft.ifft2(torch.fft.fft2(U_after_L2_prime_pad) * torch.fft.fft2(self.Q2)) # Uses Q2 (for f2)

            # Final output of the 4f system (cropped BFP of the 2nd lens)
            output_layer = U_output_4f[:, :, -self.N, -self.N] """
        
        else:
            raise ValueError('lens approach not supported')
            
        if self.counter == 0 and not self.training:    
            save_output_layer(output_layer, self.bfp_dir, self.lens_approach, self.counter, self.datetime, self.config)

        self.counter += 1
        
        if self.conv3d == False:
            # make a 4D tensor to store the 2D images
            imgs3D = torch.zeros(Nbatch, self.Nimgs, self.image_volume_um[0], self.image_volume_um[0]).type(torch.FloatTensor).to(self.device)
        elif self.conv3d == True and self.Nimgs > 1:
            #make a 5D tensor to store the 3D images
            imgs3D = torch.zeros(Nbatch, 1, self.Nimgs, self.image_volume_um[0], self.image_volume_um[0]).type(torch.FloatTensor).to(self.device)
        else:
            raise ValueError('Nimgs must be > 1 to use conv3d')
        
        for l in range(self.Nimgs):
            for i in range(Nbatch):
                for j in range(Nemitters):
                    # change x value to fit different field of view
                    x = xyz[i, j, 0].type(torch.LongTensor) - self.image_volume_um[0]//2
                    y = xyz[i, j, 1].type(torch.LongTensor)
                    z = xyz[i, j, 2].type(torch.LongTensor)

                    x_ori = xyz[i, j, 0].type(torch.LongTensor)
                    U1 = self.angular_spectrum_propagation(output_layer, x) # angular spectrum propagation

                    # Here we assume that the beam is being dithered up and down
                    intensity = torch.sum(U1[0, 0, :, int((self.N//2-1) + z)]) # if px is 1e-6 why multiply by 1e6?
                    if self.conv3d == False:
                        imgs3D[i, l, x_ori - self.psf_keep_radius:x_ori + self.psf_keep_radius+1, y - self.psf_keep_radius: y + self.psf_keep_radius + 1] += torch.from_numpy(
                            self.imgs[abs(z.item()-self.z_depth_list[l])].astype('float32')).type(torch.FloatTensor).to(self.device) * intensity
                    
                    elif self.conv3d == True and self.Nimgs > 1:
                        imgs3D[i, 0, l, x_ori - self.psf_keep_radius:x_ori + self.psf_keep_radius+1, y - self.psf_keep_radius: y + self.psf_keep_radius + 1] += torch.from_numpy(
                            self.imgs[abs(z.item()-self.z_depth_list[l])].astype('float32')).type(torch.FloatTensor).to(self.device) * intensity

        
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
            return result, self.Nimgs
        else:
            result_noisy = self.noise(imgs3D)
            result_noisy01 = self.norm01(result_noisy)
            return result_noisy01
