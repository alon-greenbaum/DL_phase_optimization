# -*- coding: utf-8 -*-
import os
import time
import numpy as np
import skimage.io
import random
from datetime import datetime

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
from beam_profile_gen import phase_gen
import scipy.io as sio

# Define the parameters 
N = 500 # grid size
px = 1e-6 # pixel size [m]
focal_length = 2e-3 #[m] equivalent to 2 mm away from the lens
wavelength = 0.561e-6 # [m]
refractive_index = 1.0 # We assume propagation is in air
psf_width_pixels = 101 # How big will be the psf image
psf_width_meters = psf_width_pixels * px
numerical_aperture = 0.6 #Relates to the resolution of the detection objective 
bead_radius = 1 #pixels

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# AG this part is generating the images of the defocused images at different planes 
# the code will save these images as template for future use
def beads_img(psf_width_pixels, bead_radius ):
    data_path = "beads_img_defocus/" #Should we define it ?
    bead_ori_img = np.zeros((psf_width_pixels,psf_width_pixels))  
    setup_defocus_psf = sio.loadmat('psf_z.mat')['psf'] #this is a matrix that Chen has generated before
    ori_intensity = 20000 #seems like a random number 
    for x in range(math.floor(psf_width_pixels/2)-bead_radius, math.floor(psf_width_pixels/2)+bead_radius+1):
        for y in range(math.floor(psf_width_pixels/2)-bead_radius, math.floor(psf_width_pixels/2)+bead_radius+1):
            if (x - math.floor(psf_width_pixels/2))**2 + (y - math.floor(psf_width_pixels/2))**2 <= bead_radius**2:
                bead_ori_img[x,y] = ori_intensity
    #AG I assume that it smooth every bead 
    bead_ori_img = skimage.filters.gaussian(bead_ori_img, sigma=1)

    #Create the images that have the PSF, based on distance from the detection lens focal point
    #41 refers to 41 um maximum defocus in the simulation; the volume was 200x200x30um^3, 30 um in Z so max 15 um defocus
    for i in range(41):
        blurred_img = apply_blur_kernel(bead_ori_img, setup_defocus_psf[i])
        skimage.io.imsave(data_path + 'z' + str(i).zfill(2) + '.tiff', blurred_img.astype('uint16'))
    return None

def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)
        
def learn_mask():
    # set random number generators for repeatability
    torch.manual_seed(99)
    np.random.seed(99)
    
    # train on GPU if available
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    
    # initial learning rate
    initial_learning_rate = 0.01
    # 1 for mask learning, examples are generated 16 at a time)
    batch_size = 1
    max_epochs = 200
    ntrain = 10000
    nvalid = 1000
    mask_phase_pixels = 500

    #mask_real = torch.from_numpy(mask_init).type(torch.FloatTensor).to(device)
    mask_phase = np.zeros((mask_phase_pixels,mask_phase_pixels))
    #in its current version it will return zeros 
    #mask_phase = phase_gen()
    mask_phase = torch.from_numpy(mask_phase).type(torch.FloatTensor).to(device)
    #mask_param = mask_real + 1j*mask_phase
    
    mask_param = nn.Parameter(mask_phase)
    mask_param.requires_grad_()
    
    path_save = 'data_mask_learning/'
    path_train = 'traininglocations/'
    if not (os.path.isdir(path_save)):
        os.mkdir(path_save)
    
    # set results folder
    model_name = '{}_{}'.format('phase_model_', datetime.now().strftime("%Y%m%d-%H%M%S"))
    res_dir = os.path.join('./results', model_name)
    makedirs(res_dir)
    
    # load all locations pickle file
    path_pickle = path_train + 'labels.pickle'
    with open(path_pickle, 'rb') as handle:
        labels = pickle.load(handle)
    
    # parameters for data loaders batch size is 1 because examples are generated 16 at a time
    params_train = {'batch_size': 1, 'shuffle': True}
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
    
    validation_set = PhasesOnlineDataset(partition['valid'], labels)
    validation_generator = DataLoader(validation_set, **params_valid)
    
    # build model and convert all the weight tensors to cuda()
    print('=' * 20)
    print('CNN architecture')
    print('=' * 20)
    
    cnn = OpticsDesignCNN()
    cnn.to(device)
    
    # gap between validation and training loss
    gap_thresh = 1e-4
    
    # adam optimizer
    optimizer = Adam(list(cnn.parameters()) + [mask_param], lr=initial_learning_rate)
    # learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True, min_lr=1e-6)
    
    # loss function 
    # wait to design
    #criterion = KDE_loss3D(100.0)
    criterion = nn.BCEWithLogitsLoss().to(device)
    # Model layers and number of parameters
    #print(cnn)
    print("number of parameters: ", sum(param.numel() for param in cnn.parameters()))
    # start from scratch
    start_epoch, end_epoch, num_epochs = 0, max_epochs, max_epochs
    # initialize the learning results dictionary
    learning_results = {'train_loss': [], 'train_jacc': [], 'valid_loss': [], 'valid_jacc': [],
                            'max_valid': [], 'sum_valid': [], 'steps_per_epoch': steps_per_epoch}
    # initialize validation set loss to be infinity and jaccard to be 0
    valid_loss_prev, valid_JI_prev = float('Inf'), 0.0
    
    
    # starting time of training
    train_start = time.time()
    
    # loop over epochs
    not_improve = 0
    train_losses = []
    for epoch in np.arange(start_epoch,end_epoch):
        epoch_start_time = time.time()
        # print current epoch number
        print('='*20)
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('='*20)
        
        # training phase
        cnn.train()
        train_loss = 0.0
        train_jacc = 0.0
        with torch.set_grad_enabled(True):
            for batch_ind, (xyz_np, Nphotons, targets) in enumerate(training_generator):
                #return xyz_np
                # transfer data to variable on GPU
                targets = targets.to(device)
                Nphotons = Nphotons.to(device)
                xyz_np = xyz_np.to(device)
                
                # squeeze batch dimension
                targets = targets.squeeze(dim=0)
                Nphotons = Nphotons.squeeze()
                xyz_np = xyz_np.squeeze()
                
                # print(batch_ind)
                # print(xyz_np.shape)
                # print(targets.shape)
                # forward + backward + optimize
                #img = torch.zeros((batch_size_gen,1,500,500)).type(torch.FloatTensor).to(device)
                optimizer.zero_grad()
                outputs = cnn(mask_param,xyz_np,Nphotons)
                #return targets
                loss = criterion(outputs,targets)
                
                loss.backward(retain_graph=True)
                optimizer.step()
                #return loss
                
                # running statistics
                train_loss += loss.item()
                jacc_ind = jaccard_coeff(outputs,targets)
                train_jacc += jacc_ind.item()
                
                # print training loss
                #print('Epoch [%d/%d], Iter [%d/%d], Loss: %.4f\n' % (epoch+1,
                #      num_epochs, batch_ind+1, steps_per_epoch, loss.item()))
                
                if batch_ind % 1000 == 0:
                    savePhaseMask(mask_param,batch_ind,epoch,res_dir)
                
        train_losses.append(train_loss)
        np.savetxt('train_losses.txt',train_losses,delimiter=',')
        if epoch % 10 == 0:
            torch.save(cnn.state_dict(),res_dir + '/net_{}.pt'.format(epoch))
        
    return labels

if __name__ == '__main__':
    # pre generate defocus beads
    # beads_img()
    
    a = learn_mask()
    
    
    
    
    
    
    
    
    
