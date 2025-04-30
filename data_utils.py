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
def generate_batch(batch_size, num_particles_range, particle_spatial_range_xy, particle_spatial_range_z, seed=None, z_coupled_ratio=0.0, z_coupled_spacing_range=None):
    """
    Generate a batch of bead locations, with an option to include a mixture of random beads and z-coupled bead pairs.
    Args:
        batch_size: Number of images in the batch.
        num_particles_range: [min, max] number of beads per image.
        particle_spatial_range_xy: allowed x/y positions.
        particle_spatial_range_z: allowed z positions.
        seed: random seed.
        z_coupled_ratio: fraction of beads that are z-coupled pairs (0.0 = all random, 1.0 = all z-coupled).
        z_coupled_spacing_range: [min, max] allowed z spacing for z-coupled pairs (inclusive).
    Returns:
        xyz_grid: (batch_size, num_particles, 3) array of bead positions.
        Nphotons: (batch_size, num_particles) array of photon counts.
    """
    #if seed is not None:
    #    np.random.seed(seed)
    #    random.seed(seed)

    num_particles = np.random.randint(num_particles_range[0], num_particles_range[1], 1).item()
    Nsig_range = [10000, 10001]  # in [counts]
    Nphotons = np.random.randint(Nsig_range[0], Nsig_range[1], (batch_size, num_particles)).astype('float32')

    xyz_grid = np.zeros((batch_size, num_particles, 3)).astype('int')
    for k in range(batch_size):
        n_z_coupled = int(num_particles * z_coupled_ratio // 2) * 2  # must be even, each pair is 2 beads
        n_random = num_particles - n_z_coupled
        beads = []
        # Generate z-coupled pairs
        for _ in range(n_z_coupled // 2):
            x = random.choice(particle_spatial_range_xy)
            y = random.choice(particle_spatial_range_xy)
            z1 = random.choice(particle_spatial_range_z)
            if z_coupled_spacing_range is not None:
                min_spacing, max_spacing = z_coupled_spacing_range
                possible_spacings = [s for s in range(min_spacing, max_spacing+1) if (z1 + s) in particle_spatial_range_z]
                if not possible_spacings:
                    # fallback: just pick another z
                    z2 = random.choice(particle_spatial_range_z)
                else:
                    spacing = random.choice(possible_spacings)
                    z2 = z1 + spacing
            else:
                z2 = random.choice(particle_spatial_range_z)
            beads.append([x, y, z1])
            beads.append([x, y, z2])
        # Generate remaining random beads
        for _ in range(n_random):
            x = random.choice(particle_spatial_range_xy)
            y = random.choice(particle_spatial_range_xy)
            z = random.choice(particle_spatial_range_z)
            beads.append([x, y, z])
        # Shuffle to avoid ordering bias
        random.shuffle(beads)
        xyz_grid[k, :, :] = np.array(beads)
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

def other_planes_gt(xyz_np, config, plane):


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
            if z == plane:
                x = xyz_np[i, j, 0]
                y = xyz_np[i, j, 1]
                boolean_grid[i, 0, int(x // ratio_input_output_image_size), int(y // ratio_input_output_image_size)] = 1
    boolean_grid = torch.from_numpy(boolean_grid).type(torch.FloatTensor)
    return boolean_grid


def batch_xyz_to_3d_volume(xyz_np, config):
    """
    Converts batch xyz bead locations to a 3D binary volume for each batch.
    Args:
        xyz_np: (batch_size, num_particles, 3) array of bead positions.
        config: dict with keys 'image_volume' (list/tuple of [H, W, D])
    Returns:
        volumes: (batch_size, D, H, W) numpy array, 1 where bead exists, 0 elsewhere
    """
    image_volume = config["image_volume"]  # [H, W, D]
    H, W, D = image_volume
    batch_size, num_particles, _ = xyz_np.shape
    volumes = np.zeros((batch_size, D, H, W), dtype=np.uint8)
    for i in range(batch_size):
        for j in range(num_particles):
            x = xyz_np[i, j, 0]
            y = xyz_np[i, j, 1]
            z = xyz_np[i, j, 2]
            # Check bounds
            if 0 <= x < H and 0 <= y < W and -D//2 <= z < D//2:
                volumes[i, z+D//2, x, y] = 1
    return volumes


def save_3d_volume_as_tiffs(volume, out_dir, base_name):
    """
    Save a 3D numpy array (D, H, W) as a series of 2D tiff images, one per z-slice, in a unique subfolder.
    Args:
        volume: 3D numpy array (D, H, W)
        out_dir: directory to save images
        base_name: base filename for each slice and subfolder name
    """
    import os
    import skimage.io
    subfolder = os.path.join(out_dir, base_name)
    os.makedirs(subfolder, exist_ok=True)
    D = volume.shape[0]
    for z in range(D):
        fname = os.path.join(subfolder, f"{base_name}_z{z:02d}.tiff")
        # Save as 8-bit, 255 for bead, 0 for background
        img8 = (volume[z] * 255).astype(np.uint8)
        skimage.io.imsave(fname, img8)


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
    import datetime
    # Example config (edit as needed)
    config = {
        "image_volume": [200, 200, 30],  # [H, W, D]
        "num_particles_range": [50, 51],
        "particle_spatial_range_xy": range(15, 185),
        "particle_spatial_range_z": range(-10, 11),
    }
    batch_size = 1
    dt_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = os.path.join("visualization_examples", dt_str)
    os.makedirs(out_dir, exist_ok=True)

    # 1. Visualize random beads (no z-coupling)
    xyz_rand, _ = generate_batch(
        batch_size,
        config["num_particles_range"],
        config["particle_spatial_range_xy"],
        config["particle_spatial_range_z"],
        seed=42,
        z_coupled_ratio=0.0
    )
    vol_rand = batch_xyz_to_3d_volume(xyz_rand, config)[0]
    save_3d_volume_as_tiffs(vol_rand, out_dir, "random_beads")
    print("Saved random bead 3D volume slices to:", os.path.join(out_dir, "random_beads"))

    # 2. Visualize z-coupled beads (50% z-coupled)
    xyz_zc, _ = generate_batch(
        batch_size,
        config["num_particles_range"],
        config["particle_spatial_range_xy"],
        config["particle_spatial_range_z"],
        seed=43,
        z_coupled_ratio=0.5,
        z_coupled_spacing_range=(2, 5)
    )
    vol_zc = batch_xyz_to_3d_volume(xyz_zc, config)[0]
    save_3d_volume_as_tiffs(vol_zc, out_dir, "z_coupled_beads")
    print("Saved z-coupled bead 3D volume slices to:", os.path.join(out_dir, "z_coupled_beads"))
