# -*- coding: utf-8 -*-
import os
import time
import numpy as np
import skimage.io
import random
from datetime import datetime
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pickle
from psf_gen import apply_blur_kernel
from data_utils import PhasesOnlineDataset, savePhaseMask, generate_batch, load_config, makedirs
from cnn_utils import OpticsDesignCNN
from cnn_utils_unet import OpticsDesignUnet
from loss_utils import KDE_loss3D, jaccard_coeff
#from beam_profile_gen import phase_gen
import scipy.io as sio
#import hdf5storage  # new import for saving MATLAB 7.3 files

# systemctl restart --user xdg-desktop-portal-gnome
# nohup python mask_learning.py &> ./logs/$(date +'%Y%m%d-%H%M%S').txt &
def list_to_range(lst):
    if len(lst) == 1:
        return range(lst[0])
    if len(lst) == 2:
        return range(lst[0], lst[1])
    elif len(lst) == 3:
        return range(lst[0], lst[1], lst[2])

# This part generates the beads location for the training and validation sets
def gen_data(config, res_dir):
    ntrain = config['ntrain']
    nvalid = config['nvalid']
    batch_size_gen = config['batch_size_gen']
    particle_spatial_range_xy = list_to_range(config['particle_spatial_range_xy'])
    particle_spatial_range_z = list_to_range(config['particle_spatial_range_z'])
    num_particles_range = config['num_particles_range']
    z_coupled_ratio = config.get('z_coupled_ratio',0)
    z_coupled_spacing_range = config.get('z_coupled_spacing_range',(0,0))
    #device = config['device']
    #path_train = config['path_train']
    

    torch.backends.cudnn.benchmark = True

    # set random seed for repeatability 
    # seed already set in main function - ryan
    #torch.manual_seed(random_seed)
    #np.random.seed(random_seed)

    #if not (os.path.isdir(path_train)):
    #    os.mkdir(path_train)

    # print status
    print('=' * 50)
    print('Sampling examples for training')
    print('=' * 50)

    # locations for phase mask learning are saved in batches of 16 for convenience

    # calculate the number of training batches to sample
    ntrain_batches = int(ntrain / batch_size_gen)

    # generate examples for training
    labels_dict = {}
    for i in range(ntrain_batches):
        # sample a training example
        xyz, xyz_between_beads = generate_batch(batch_size_gen, num_particles_range, particle_spatial_range_xy,
                                       particle_spatial_range_z, z_coupled_ratio, 
                                       z_coupled_spacing_range)
        labels_dict[str(i)] = {'xyz': xyz, 'xyz_between_beads': xyz_between_beads}
        # print number of example
        print('Training Example [%d / %d]' % (i + 1, ntrain_batches))

    nvalid_batches = int(nvalid / batch_size_gen)
    for i in range(nvalid_batches):
        xyz, xyz_between_beads = generate_batch(batch_size_gen, num_particles_range, particle_spatial_range_xy,
                                       particle_spatial_range_z, z_coupled_ratio, 
                                       z_coupled_spacing_range)
        labels_dict[str(i + ntrain_batches)] = {'xyz': xyz, 'xyz_between_beads': xyz_between_beads}
        print('Validation Example [%d / %d]' % (i + 1, nvalid_batches))

    path_labels = os.path.join(res_dir,'labels.pickle')
    with open(path_labels, 'wb') as handle:
        pickle.dump(labels_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # Save labels_dict in .mat format as MATLAB 7.3
    #path_labels_mat = os.path.join(res_dir, 'labels.mat')
    #hdf5storage.savemat(path_labels_mat, labels_dict, matlab_ver='7.3')

# AG this part is generating the images of the defocused images at different planes 
# the code will save these images as template for future use.
# The code takes into aacount large beads, as well as small
def beads_img(config):
    #Unpack the parameters
    psf_width_pixels = config['psf_width_pixels']
    data_path = config['data_path']
    bead_radius = config['bead_radius']
    bead_ori_img = np.zeros((psf_width_pixels,psf_width_pixels))  
    setup_defocus_psf = sio.loadmat('psf_z.mat')['psf'] #this is a matrix that Chen has generated before
    ori_intensity = config['ori_intensity']
    max_defocus = config['max_defocus']
    
    makedirs(data_path)

    #generate the bead in the center based on the bead size in pixels
    for x in range(int(math.floor(psf_width_pixels/2)-bead_radius), int(math.floor(psf_width_pixels/2)+bead_radius+1)):
        for y in range(int(math.floor(psf_width_pixels/2)-bead_radius), int(math.floor(psf_width_pixels/2)+bead_radius+1)):
            if (x - math.floor(psf_width_pixels/2))**2 + (y - math.floor(psf_width_pixels/2))**2 <= bead_radius**2:
                bead_ori_img[x,y] = ori_intensity
    #Smooth every bead
    bead_ori_img = skimage.filters.gaussian(bead_ori_img, sigma=1)

    #Create the images that have the PSF, based on distance from the detection lens focal point
    #41 refers to 41 um maximum defocus in the simulation; the volume was 200x200x30um^3, 30 um in Z so max 15 um defocus
    for i in range(max_defocus):
        blurred_img = apply_blur_kernel(bead_ori_img, setup_defocus_psf[i])
        skimage.io.imsave(os.path.join(data_path, 'z' + str(i).zfill(2) + '.tiff'), blurred_img.astype('uint16'))
    return None


        
def learn_mask(config,res_dir):

    #Unpack parameters
    ntrain = config['ntrain']
    nvalid = config['nvalid']
    batch_size_gen = config['batch_size_gen']
    initial_learning_rate = config['initial_learning_rate']
    batch_size = config['batch_size']
    max_epochs = config['max_epochs']
    mask_phase_pixels = config['mask_phase_pixels']
    #psf_width_meters = config['psf_width_pixels'] * config['px']
    #path_save = config['path_save']
    #path_train = config['path_train']
    learning_rate_scheduler_factor = config['learning_rate_scheduler_factor']
    learning_rate_scheduler_patience = config['learning_rate_scheduler_patience']
    learning_rate_scheduler_patience_min_lr = config['learning_rate_scheduler_patience_min_lr']
    device = config['device']
    use_unet = config['use_unet']
    z_spacing = config.get('z_spacing', 0)
    z_img_mode = config.get('z_img_mode', 'edgecenter')
    if z_img_mode == 'edgecenter' and z_spacing > 0:
        z_depth_list = [-z_spacing, 0, z_spacing]
    else:
        z_depth_list = list(range(-z_spacing,z_spacing+1))
    Nimgs = len(z_depth_list)
    config['Nimgs'] = Nimgs
    
    num_classes = config['num_classes']
    

    
    # train on GPU if available
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    #Set the mask_phase as the parameter to derive
    mask_phase = np.zeros((mask_phase_pixels,mask_phase_pixels))
    mask_phase = torch.from_numpy(mask_phase).type(torch.FloatTensor).to(device)
    mask_param = nn.Parameter(mask_phase)
    mask_param.requires_grad_()

    # I don't know what goes in here so I commented it out - ryan
    #if not (os.path.isdir(path_save)):
    #    os.mkdir(path_save)
    
    

    # Save dictionary to a text file ensuring key 'px' is saved as a float
    with open(os.path.join(res_dir, 'config.yaml'), 'w') as file:
        for key, value in config.items():
            if key == "px":
                file.write(f"{key}: {float(value)}\n")
            else:
                file.write(f"{key}: {value}\n")


    # load all locations pickle file, to generate the labels go to Generate data folder
    path_pickle = os.path.join(res_dir, 'labels.pickle')
    with open(path_pickle, 'rb') as handle:
        labels = pickle.load(handle)
    
    # parameters for data loaders batch size is 1 because examples are generated 16 at a time
    params_train = {'batch_size': batch_size, 'shuffle': True}
    #params_valid = {'batch_size': batch_size, 'shuffle': False}
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
    
    training_set = PhasesOnlineDataset(partition['train'],labels, config)
    training_generator = DataLoader(training_set, **params_train)
    
    #validation_set = PhasesOnlineDataset(partition['valid'], labels, config)
    #validation_generator = DataLoader(validation_set, **params_valid)
    
    # build model and convert all the weight tensors to cuda()
    print('=' * 20)

    if use_unet:
        cnn = OpticsDesignUnet(config)
        print('UNET architecture')
    else:
        cnn = OpticsDesignCNN(config)
        print('CNN architecture')

    print('=' * 20)

    cnn.to(device)

    # adam optimizer
    optimizer = Adam(list(cnn.parameters()) + [mask_param], lr=initial_learning_rate)
    # learning rate scheduler for now
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=learning_rate_scheduler_factor,\
                                  patience=learning_rate_scheduler_patience, verbose=True, 
                                  min_lr=learning_rate_scheduler_patience_min_lr)
    
    # loss function
    #criterion = KDE_loss3D(100.0)
    if num_classes > 1:
        criterion = nn.CrossEntropyLoss().to(device)
    else:
        criterion = nn.BCEWithLogitsLoss().to(device)
    # Model layers and number of parameters
    print("number of parameters: ", sum(param.numel() for param in cnn.parameters()))
    # start from scratch
    start_epoch, end_epoch, num_epochs = 0, max_epochs, max_epochs
    # initialize the learning results dictionary
    #learning_results = {'train_loss': [], 'train_jacc': [], 'valid_loss': [], 'valid_jacc': [],
    #                        'max_valid': [], 'sum_valid': [], 'steps_per_epoch': steps_per_epoch}
    # initialize validation set loss to be infinity and jaccard to be 0
    #valid_loss_prev, valid_JI_prev = float('Inf'), 0.0
    
    
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
            for batch_ind, (xyz, targets) in enumerate(training_generator):
                #return xyz_np
                # transfer data to variable on GPU
                targets = targets.to(device)
                #Nphotons = Nphotons.to(device)
                xyz = xyz.to(device)
                
                # squeeze batch dimension
                targets = targets.squeeze(0)
                if num_classes > 1:
                    targets = targets.squeeze(1)
                    targets = targets.long()
                #Nphotons = Nphotons.squeeze()
                xyz = xyz.squeeze(0)
                
                # print(batch_ind)
                # print(xyz_np.shape)
                # print(targets.shape)
                # forward + backward + optimize
                #img = torch.zeros((batch_size_gen,1,500,500)).type(torch.FloatTensor).to(device)
                optimizer.zero_grad()
                outputs = cnn(mask_param,xyz)
                #return targets
                loss = criterion(outputs, targets)
                
                loss.backward(retain_graph=True)
                optimizer.step() # modified to use the learning rate scheduler
                
                #return loss
                
                # running statistics
                train_loss += loss.item()
                #jacc_ind = jaccard_coeff(outputs,targets)
                #train_jacc += jacc_ind.item()
                
                # print training loss
                print('Epoch [%d/%d], Iter [%d/%d], Loss: %.4f\n' % (epoch+1,
                      num_epochs, batch_ind+1, steps_per_epoch, loss.item()))
                
                #if batch_ind % 1000 == 0:
                #    savePhaseMask(mask_param,batch_ind,epoch,res_dir)
        scheduler.step(train_loss)        
        train_losses.append(train_loss)
        np.savetxt(os.path.join(res_dir,'train_losses.txt'),train_losses,delimiter=',')
        if epoch % 10 == 0:
            torch.save(cnn.state_dict(),os.path.join(res_dir, 'net_{}.pt'.format(epoch)))
        
        savePhaseMask(mask_param,batch_ind,epoch,res_dir)
    return labels

if __name__ == '__main__':

    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    config = load_config("config.yaml")
    z_range_cost_function = config['z_range_cost_function']
    z_coupled_ratio = config.get('z_coupled_ratio',0)
    z_coupled_spacing_range = config.get('z_coupled_spacing_range',(0,0))
    num_classes = config['num_classes']
   
    assert (z_range_cost_function[1]-z_range_cost_function[0]+1) == 1, "we haven't set up multi img output prediction yet" 
    if num_classes > 1:
        assert z_coupled_ratio > 0 and z_coupled_spacing_range[0] > 0, "z_coupled_ratio and z_coupled_spacing_range should be set for multi class imgs"
    
    config['device'] = device
    
    # Set the random seed
    random_seed = config['random_seed']
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed) # ryan added for gen data
    
    """
    config = {
        #How many bead cases per epoch
        "device": device, #Same GPU for all
        "ntrain": 1000, #default 10000 how many images per epoch
        "nvalid": 100, #default 1000
        "batch_size_gen": 2, #default 2
        # Number of emitters per image
        "num_particles_range": [20, 30], #with a strong gpu [450, 550]
        "image_volume": [200, 200, 30], #the volume of imaging each dimension in um
        "particle_spatial_range_xy": range(15, 185), #dependece on the volume size default 200 um
        # um, to avoid edges of the image so 15 um away
        "particle_spatial_range_z": range(-10, 11, 1),  # um defualt range(-10,11,1)
         # Define the parameters
        "N": 500,  # grid size,
        "px": 1e-6,  # pixel size [m]
        "focal_length": 2e-3,  # [m] equivalent to 2 mm away from the lens
        "wavelength": 0.561e-6,  # [m]
        "refractive_index":  1.0,  # We assume propagation is in air
        "psf_width_pixels": 101,  # How big will be the psf image
        "psf_edge_remove": 15,
        "psf_keep_radius":15,
        "numerical_aperture": 0.6,  # Relates to the resolution of the detection objective
        "bead_radius": 1.0,  # pixels 1.0 default, can change to make larger beads
        "random_seed": 99,
        "initial_learning_rate": 0.01,
        "batch_size": 1,
        "max_epochs": 200,
        "mask_phase_pixels": 500,
        "data_path": "beads_img_defocus/",
        "path_save":"data_mask_learning/",
        "path_train":"traininglocations/",
        "ori_intensity": 20000, #arbitrary units
        "laser_beam_FWHC": 0.0001, #in um, the size of the FWHM of the Gaussian beam
        "max_defocus": 41, #um for generating the PSFs
        "learning_rate_scheduler_factor": 0.1,
        "learning_rate_scheduler_patience": 5,
        "learning_rate_scheduler_patience_min_lr": 1e-6,
        "max_intensity": 5e4,
        #In case that the network needs to downsample the image, GPU memory issues
        "ratio_input_output_image_size": 4, # defualt with Unet 1, with ResNet 4
        #How many planes to consider as right classification, besides the plane
        # in focus, if 1 the -1, in focus, and +1 plane will be considered correct
        #if 0, only the infocus plane is considered, default value 1
        "z_range_cost_function": 1,
        "use_unet": False,
        "num_classes": 1
    }
    """
    

    
    # set results folder
    model_name = f"phase_model_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    res_dir = os.path.join('training_results', model_name)
    makedirs(res_dir)
    
    # Generate the data for the training
    gen_data(config,res_dir)
    # pre generate defocus beads - can only run once
    if not os.path.isdir(config['data_path']):
        beads_img(config)
    # learn the mask
    learn_mask(config,res_dir)










