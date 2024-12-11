# -*- coding: utf-8 -*-

import torch
from torch.utils.data import Dataset
from torch.autograd import Variable
import numpy as np
import random
from skimage.io import imread
import skimage
# ======================================================================================================================
# numpy array conversion to variable and numpy complex conversion to 2 channel torch tensor
# ======================================================================================================================


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
    if seed is not None:
        np.random.seed(seed)
        
    # upper and lower limits for the number fo emitters
    num_particles = np.random.randint(num_particles_range[0], num_particles_range[1], 1).item()

    # range of signal counts assuming a uniform distribution
    # not sure what Chen wanted to achieve but it is uniform
    Nsig_range = [10000, 10001]  # in [counts]
    Nphotons = np.random.randint(Nsig_range[0], Nsig_range[1], (batch_size, num_particles))
    Nphotons = Nphotons.astype('float32')
    
    xyz_grid = np.zeros((batch_size,num_particles,3)).astype('int')
    for k in range(batch_size):
        
        xyz_grid[k,:,0] = random.choices(particle_spatial_range_xy,k = num_particles) # in pixel
        xyz_grid[k,:,1] = random.choices(particle_spatial_range_xy,k = num_particles) # in pixel
        xyz_grid[k,:,2] = random.choices(particle_spatial_range_z,k = num_particles) # in pixel

    return xyz_grid, Nphotons
    


# ==================================
# projection of the continuous positions on the recovery grid in order to generate the training label
# =====================================
# converts continuous xyz locations to a boolean grid
def batch_xyz_to_boolean_grid(xyz_np):

    # number of particles
    batch_size, num_particles = xyz_np[:,:,2].shape
    # set dimension
    width_detect = 1
    H, W, D = 200,200,1
    # project xyz locations on the grid and shift xy to the upper left corner
    # xg = np.floor(xyz_np[:,:,0])
    # yg = np.floor(xyz_np[:,:,1])
    # zg = np.floor(xyz_np[:,:,2])
    
    # indX = xg.flatten('F')
    # indY = yg.flatten('F')
    # indZ = zg.flatten('F')
    # indZ1 = zg.flatten('F')
    # z_loc = np.logical_and(indZ >= -width_detect, indZ <= +width_detect)
    # indX = indX[z_loc].tolist()
    # indY = indY[z_loc].tolist()
    # indZ = indZ[z_loc].tolist()
    # # indices for sparse tensor
    # #indX, indY, indZ = (xg.flatten('F')).tolist(), (yg.flatten('F')).tolist(), (zg.flatten('F')).tolist()
    
    # if batch_size > 1:
    #     #indS = (np.kron(np.ones(num_particles), np.arange(0, batch_size, 1)).astype('int')).tolist()
    #     indS = [i% batch_size for i in range(len(indZ1)) if indZ1[i] >= -width_detect and indZ1[i] <= +width_detect]
    #     ibool = torch.LongTensor([indS, [i+width_detect for i in indZ], indY, indX])
    # else:
    #     ibool = torch.LongTensor([indZ, indY, indX])
        
    # # spikes for sparse tensor
    # #vals = torch.ones(batch_size*num_particles)
    # vals = torch.ones(len(indZ))
    # # resulting 3D boolean tensor
    # if batch_size > 1:
    #     boolean_grid = torch.sparse.FloatTensor(ibool, vals, torch.Size([batch_size, D, H, W])).to_dense()
    # else:
    #     boolean_grid = torch.sparse.FloatTensor(ibool, vals, torch.Size([D, H, W])).to_dense()
    
    boolean_grid = np.zeros((batch_size,1,H//4,W//4))
    for i in range(batch_size):
        for j in range(num_particles):
            z = xyz_np[i,j,2]
            if z >=-1 and z<= 1:
                x = xyz_np[i,j,0]
                y = xyz_np[i,j,1]
                boolean_grid[i,0,int(x//4),int(y//4)] = 1
                #print(int(x//4),int(y//4))
    boolean_grid = torch.from_numpy(boolean_grid).type(torch.FloatTensor)
    return boolean_grid
    
        
        
        
        
# ==============
# continuous emitter positions sampling using two steps: first sampling disjoint indices on a coarse 3D grid,
# and afterwards refining each index using a local perturbation.
# ================

class PhasesOnlineDataset(Dataset):
    
    # initialization of the dataset
    def __init__(self, list_IDs, labels):
        self.list_IDs = list_IDs
        self.labels = labels
    
    # total number of samples in the dataset
    def __len__(self):
        return len(self.list_IDs)
    
    def __getitem__(self,index):
        ID = self.list_IDs[index]
        
        # associated number of photons
        dict = self.labels[ID]
        Nphotons_np = dict['N']
        Nphotons = torch.from_numpy(Nphotons_np)
        
        # corresponding xyz labels turned to a boolean tensor
        xyz_np = dict['xyz']
        bool_grid = batch_xyz_to_boolean_grid(xyz_np)
        return xyz_np, Nphotons, bool_grid
        

def savePhaseMask(mask_param,ind,epoch,res_dir):
    mask_numpy = mask_param.data.cpu().clone().numpy()
    #mask_real = np.abs(mask_numpy)
    #mask_phase = np.angle(mask_numpy)
    #skimage.io.imsave('phase_learned/mask_real_epoch_' + str(epoch) + '_' + str(ind) + '.tiff' , mask_real)
    skimage.io.imsave(res_dir + '/mask_phase_epoch_' + str(epoch) + '_' + str(ind) + '.tiff' , mask_numpy)
    return 0
    
    
    
if __name__ == '__main__':
    generate_batch(8)
    
    
    
    
    
    
