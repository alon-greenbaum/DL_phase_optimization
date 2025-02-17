# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 12:11:51 2022

@author: ische
"""
import time
import numpy as np
# import scipy.integrate
# import scipy.signal
# import scipy.special
import skimage.io
import random
import os


import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pickle
from psf_gen import apply_blur_kernel
from data_utils import PhasesOnlineDataset, savePhaseMask
from cnn_utils import OpticsDesignCNN
from loss_utils import KDE_loss3D, jaccard_coeff
from beam_profile_gen import phase_gen, phase_mask_gen
import scipy.io as sio


N = 500 # grid size
px = 1e-6 # pixel size (um)
focal_length = 6e-3
wavelength = 0.561e-6
refractive_index = 1.0
psf_width_pixels = 101
pixel_size_meters = 1e-6
psf_width_meters = psf_width_pixels * pixel_size_meters
numerical_aperture = 0.6

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")





mask_phase = phase_gen()
mask_phase = torch.from_numpy(mask_phase).type(torch.FloatTensor).to(device)
#mask_param = mask_real + 1j*mask_phase

mask_param = nn.Parameter(mask_phase)

if __name__ == '__main__':
    
    path_save = 'data_mask_learning/'
    path_train = 'traininglocations/'
    batch_size = 1
    max_epochs = 10
    ntrain = 10000
    nvalid = 1000
    initial_learning_rate = 0.0002
    if not (os.path.isdir(path_save)):
        os.mkdir(path_save)
    
    # load all locations pickle file
    path_pickle = path_train + 'labels.pickle'
    with open(path_pickle, 'rb') as handle:
        labels = pickle.load(handle)
    
    # parameters for data loaders batch size is 1 because examples are generated 16 at a time
    params_train = {'batch_size': 1, 'shuffle': False}
    params_valid = {'batch_size': 1, 'shuffle': False}
    batch_size_gen = 2
    ntrain_batches = int(ntrain/batch_size_gen)
    nvalid_batches = int(nvalid/batch_size_gen)
    steps_per_epoch = ntrain_batches
    # partition built in simulation
    ind_all = np.arange(0, ntrain_batches + nvalid_batches, 1)
    list_all = ind_all.tolist()
    list_IDs = [str(i) for i in list_all]
    train_IDs = list_IDs[:ntrain_batches]
    valid_IDs = list_IDs[ntrain_batches:]
    partition = {'train': train_IDs, 'valid': valid_IDs}
    
    training_set = PhasesOnlineDataset(partition['train'],labels)
    training_generator = DataLoader(training_set, **params_train)
    
    cnn = OpticsDesignCNN()
    cnn.to(device)
    #model_path = './results/phase_model__20221107-161136_beads_120_to_130/net_195.pt'
    model_path = './results/model_fig3/net_100.pt'
    
    cnn.load_state_dict(torch.load(model_path))
    mask_real = phase_mask_gen()
    mask_phase = skimage.io.imread('./results/model_fig3/mask_phase_epoch_100_0.tiff')
    #mask_phase = phase_gen()
    mask_phase = torch.from_numpy(mask_phase).type(torch.FloatTensor).to(device)
    mask_param = nn.Parameter(mask_phase)
    mask_param.requires_grad_()
    with torch.no_grad():
        cnn.eval()
        for batch_ind, (xyz_np, Nphotons, targets) in enumerate(training_generator):
            xyz_np = xyz_np.to(device)
            xyz_np = xyz_np.squeeze()
            targets = targets.to(device)
            targets = targets.squeeze(0)
            outputs = cnn(mask_param,xyz_np,Nphotons)
            print(outputs.shape)
            print(targets.shape)
            
            img = outputs[0,0,:,:]
            img_np = img.detach().cpu().numpy()
            skimage.io.imsave('output.tiff',img_np)
            tar = targets[0,0,:,:]
            tar = tar.detach().cpu().numpy()
            skimage.io.imsave('label.tiff',tar)
            break
    #os._exit()
    
    
    
    
    