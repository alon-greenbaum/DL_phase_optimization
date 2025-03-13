# -*- coding: utf-8 -*-

import torch
from torch.utils.data import Dataset
from torch.autograd import Variable
import numpy as np
import random
from skimage.io import imread
import skimage
import yaml
import os
import datetime
import glob


# ======================================================================================================================
# numpy array conversion to variable and numpy complex conversion to 2 channel torch tensor
# ======================================================================================================================
def load_config(config_file):
    with open(config_file, 'r') as stream:
        return yaml.load(stream, Loader=yaml.Loader)

def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)
        
def img_save_tiff(img, out_dir, name, key=None):
    """
    Save a 3D image array as separate 2D tiffs or a 2D array as a 2D tiff.
    Args:
        img (numpy.ndarray): The image array to save.
        out_dir (str): The directory to save the image(s) to.
        name (str): The base name for the saved image file(s).
        key (str, optional): An optional identifier to include in the filename. Defaults to None.
    """
    if key is None:
        base_filename = f"{name}.tiff"
        base_path = os.path.join(out_dir, base_filename)
    else:
        base_filename = f"{name}_{key}.tiff"
        base_path = os.path.join(out_dir, base_filename)

    if img.ndim == 3:
        for i in range(img.shape[0]):
            if key is None:
                path = os.path.join(out_dir, f"{name}_z_{i-(img.shape[0]//2)}.tiff")
            else:
                path = os.path.join(out_dir, f"{name}_{key}_z_{i-(img.shape[0]//2)}.tiff")
            skimage.io.imsave(path, img[i])
            print(f"Saved {name} at depth {i} for key {key} to {path}")
    elif img.ndim == 2:
        skimage.io.imsave(base_path, img)
        print(f"Saved {name} to {base_path}")
    else:
        print(f"Unexpected dimensions for phys_img: {img.shape}")
         
def find_image_with_wildcard(directory, filename_prefix, file_type):
    """
    Loads an image from a directory, where the filename matches a prefix
    followed by a wildcard (e.g., a number) and a file extension.

    Args:
    directory: The directory containing the image.
    filename_prefix: The prefix of the filename (e.g., "mask_phase_epoch_1_").

    Returns:
    A matching image path or None if no matching image is found.
    """
    search_pattern = os.path.join(directory, filename_prefix + "*." + file_type)  # Adjust extension if needed
    matching_files = glob.glob(search_pattern)

    if not matching_files:
        print(f"No files found matching pattern: {search_pattern}")
        return None

    # Load the first matching file (you might want to add logic to select
    # a specific file if multiple matches are possible).
    image_path = matching_files[0]
    return image_path



# function converts numpy array on CPU to torch Variable on GPU
def to_var(x):
    """
    Input is a numpy array and output is a torch variable with the data tensor
    on cuda.
    """
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


# function converts numpy array on CPU to torch Variable on GPU
def complex_to_tensor(phases_np):
    Nbatch, Nemitters, Hmask, Wmask = phases_np.shape
    phases_torch = torch.zeros((Nbatch, Nemitters, Hmask, Wmask, 2)).type(torch.FloatTensor)
    phases_torch[:, :, :, :, 0] = torch.from_numpy(np.real(phases_np)).type(torch.FloatTensor)
    phases_torch[:, :, :, :, 1] = torch.from_numpy(np.imag(phases_np)).type(torch.FloatTensor)
    return phases_torch


# Define a batch data generator for training and testing
def generate_batch(batch_size, num_particles_range, particle_spatial_range_xy, particle_spatial_range_z, seed=None):
    # if we're testing then seed the random generator
    #if seed is not None:
    #    np.random.seed(seed)

    # upper and lower limits for the number fo emitters
    num_particles = np.random.randint(num_particles_range[0], num_particles_range[1], 1).item()

    # range of signal counts assuming a uniform distribution
    # not sure what Chen wanted to achieve but it is uniform
    Nsig_range = [10000, 10001]  # in [counts]
    Nphotons = np.random.randint(Nsig_range[0], Nsig_range[1], (batch_size, num_particles))
    Nphotons = Nphotons.astype('float32')

    xyz_grid = np.zeros((batch_size, num_particles, 3)).astype('int')
    for k in range(batch_size):
        xyz_grid[k, :, 0] = random.choices(particle_spatial_range_xy, k=num_particles)  # in pixel
        xyz_grid[k, :, 1] = random.choices(particle_spatial_range_xy, k=num_particles)  # in pixel
        xyz_grid[k, :, 2] = random.choices(particle_spatial_range_z, k=num_particles)  # in pixel

    return xyz_grid, Nphotons


# ==================================
# projection of the continuous positions on the recovery grid in order to generate the training label
# =====================================
# converts continuous xyz locations to a boolean grid
def batch_xyz_to_boolean_grid(xyz_np, config):


    image_volume = config["image_volume"]
    ratio_input_output_image_size = config["ratio_input_output_image_size"]
    z_range_cost_function = config["z_range_cost_function"]

    # number of particles
    batch_size, num_particles = xyz_np[:, :, 2].shape
    # set dimension
    H = image_volume[0]
    W = image_volume[1]
    D = 1

    boolean_grid = np.zeros((batch_size, 1, H // int(ratio_input_output_image_size), \
                             W // int(ratio_input_output_image_size)))
    for i in range(batch_size):
        for j in range(num_particles):
            z = xyz_np[i, j, 2]
            if -z_range_cost_function <= z <= z_range_cost_function:
                x = xyz_np[i, j, 0]
                y = xyz_np[i, j, 1]
                boolean_grid[i, 0, int(x // ratio_input_output_image_size), int(y // ratio_input_output_image_size)] = 1
    boolean_grid = torch.from_numpy(boolean_grid).type(torch.FloatTensor)
    return boolean_grid


# ==============
# continuous emitter positions sampling using two steps: first sampling disjoint indices on a coarse 3D grid,
# and afterwards refining each index using a local perturbation.
# ================

class PhasesOnlineDataset(Dataset):

    # initialization of the dataset
    def __init__(self, list_IDs, labels, config):
        self.list_IDs = list_IDs
        self.labels = labels
        self.config = config

    # total number of samples in the dataset
    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        ID = self.list_IDs[index]

        # associated number of photons
        dict = self.labels[ID]
        Nphotons_np = dict['N']
        Nphotons = torch.from_numpy(Nphotons_np)

        # corresponding xyz labels turned to a boolean tensor
        xyz_np = dict['xyz']
        bool_grid = batch_xyz_to_boolean_grid(xyz_np, self.config)
        return xyz_np, Nphotons, bool_grid


def savePhaseMask(mask_param, ind, epoch, res_dir):
    mask_numpy = mask_param.data.cpu().clone().numpy()
    #mask_real = np.abs(mask_numpy)
    #mask_phase = np.angle(mask_numpy)
    #skimage.io.imsave('phase_learned/mask_real_epoch_' + str(epoch) + '_' + str(ind) + '.tiff' , mask_real)
    skimage.io.imsave(res_dir + '/mask_phase_epoch_' + str(epoch) + '_' + str(ind) + '.tiff', mask_numpy)
    return 0

def save_output_layer(output_layer, base_dir, lens_approach, counter, datetime, config):
    # Convert the tensor to a NumPy array
    output_array = torch.square(torch.abs(output_layer)).cpu().detach().numpy()[0,0,:,:]

    # Create the filename
    filename = f"{counter}.tiff"
    dir = os.path.join(base_dir, lens_approach, datetime)
    
    # create directory if it does not exist
    makedirs(dir)
    
    filepath = os.path.join(dir, filename)
    
    # Save the array as a TIFF file
    skimage.io.imsave(filepath, output_array)
    
    if counter == 0:
        # print config
        with open(os.path.join(dir, 'config.yaml'), 'w') as file:
            for key, value in config.items():
                file.write(f'{key}: {value}\n')

if __name__ == '__main__':
    generate_batch(8)
