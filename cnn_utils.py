# -*- coding: utf-8 -*-

# Import modules and libraries
import torch
import torch.nn as nn
from torch.nn.functional import interpolate
from physics_utils import PhysicalLayer
from torch.autograd import Function
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import os
from data_utils import PhasesOnlineDataset
import pickle
import time
import skimage.io
from skimage.transform import rescale, resize
from loss_utils import KDE_loss3D, jaccard_coeff


N = 500 # grid size
px = 1e-6 # pixel size (um)
focal_length = 2e-3
wavelength = 0.561e-6
refractive_index = 1.0
psf_width_pixels = 101
pixel_size_meters = 1e-6
psf_width_meters = psf_width_pixels * pixel_size_meters
numerical_aperture = 0.6

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Define the basic Conv-LeakyReLU-BN
class Conv2DLeakyReLUBN(nn.Module):
    def __init__(self, input_channels, layer_width, kernel_size, padding, dilation, negative_slope):
        super(Conv2DLeakyReLUBN, self).__init__()
        self.conv = nn.Conv2d(input_channels, layer_width, kernel_size, 1, padding, dilation)
        self.lrelu = nn.LeakyReLU(negative_slope, inplace=True)
        self.bn = nn.BatchNorm2d(layer_width)
        
    def forward(self, x):
        out = self.conv(x)
        out = self.lrelu(out)
        out = self.bn(out)
        
        return out


# Phase mask learning architecture
class OpticsDesignCNN(nn.Module):
    def __init__(self, config):
        super(OpticsDesignCNN, self).__init__()
        self.physicalLayer = PhysicalLayer(config)
        self.norm = nn.BatchNorm2d(num_features=1, affine=True)
        self.layer1 = Conv2DLeakyReLUBN(1, 32, 3, 1, 1, 0.1)
        self.layer2 = Conv2DLeakyReLUBN(32 + 1, 32, 3, 1, 1, 0.1)
        self.layer3 = Conv2DLeakyReLUBN(32 + 1, 32, 3, 1, 1, 0.1)
        self.layer4 = Conv2DLeakyReLUBN(32 + 1, 32, 3, 1, 1, 0.1)
        self.layer5 = Conv2DLeakyReLUBN(32 + 1, 32, 3, 1, 1, 0.1)
        self.layer6 = Conv2DLeakyReLUBN(32 + 1, 32, 3, 1, 1, 0.2)
        self.layer7 = Conv2DLeakyReLUBN(32 + 1, 32, 3, 1, 1, 0.2)
        self.layer8 = Conv2DLeakyReLUBN(32 + 1, 32, 3, 1, 1, 0.2)
        self.layer9 = Conv2DLeakyReLUBN(32 + 1, 32, 3, 1, 1, 0.2)
        self.layer10 = nn.Conv2d(32, 1, kernel_size=1, dilation=1)
        scale_factor = 100
        #self.pred = nn.Hardtanh(min_val=0.0, max_val=scale_factor)
        
        self.maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

    def forward(self,mask, xyz, Nphotons):
        im = self.physicalLayer(mask,xyz,Nphotons)
        im = self.norm(im)
        # extract depth features
        out = self.layer1(im)
        #print(out)
        features = torch.cat((out, im), 1)
        
        out = self.layer2(features) + out
        features = torch.cat((out, im), 1)
        
        out = self.layer3(features) + out
        features = torch.cat((out, im), 1)
        
        out = self.layer4(features) + out
        features = torch.cat((out, im), 1)
        
        out = self.layer5(features)
        out = self.maxpool(out)
        #features = torch.cat((out, im), 1)
        
        #out = self.layer6(features) + out
        #features = torch.cat((out, im), 1)
        
        #out = self.layer7(features) + out
        # features = torch.cat((out, im), 1)
        
        # out = self.layer8(features) + out
        # features = torch.cat((out, im), 1)
        
        # out = self.layer9(features) + out
        # 1x1 conv and hardtanh for final result
        out = self.layer10(out)
        out = self.maxpool(out)
        #out = self.pred(out)
        
        return out
if __name__ == '__main__':
    a = 1
    path_save = 'data_mask_learning/'
    path_train = 'traininglocations/'
    batch_size = 1
    max_epochs = 10
    ntrain = 16000
    nvalid = 3200
    
    if not (os.path.isdir(path_save)):
        os.mkdir(path_save)
    
    # load all locations pickle file
    path_pickle = path_train + 'labels.pickle'
    with open(path_pickle, 'rb') as handle:
        labels = pickle.load(handle)
    
    # parameters for data loaders batch size is 1 because examples are generated 16 at a time
    params_train = {'batch_size': 1, 'shuffle': True}
    params_valid = {'batch_size': 1, 'shuffle': False}
    batch_size_gen = 4
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
    
    start_epoch, end_epoch, num_epochs = 0, max_epochs, max_epochs

    #phylayer = PhysicalLayer()
    cnn = OpticsDesignCNN().to(device)
    criterion = KDE_loss3D(100.0)
    
    
    # generate initial mask
    x = list(range(-N//2,N//2))
    y = list(range(-N//2,N//2))
    [X, Y] = np.meshgrid(x,y)
    X = X*px
    Y = Y*px
    c = 0.02
    mask_init = np.exp(-np.sqrt(np.square(X) + np.square(Y))/(2*c**2))
    set_2 = mask_init <= 0.75
    mask_init[set_2] = 0

    #mask_init = np.zeros((500,500))
    mask_real = nn.Parameter(torch.from_numpy(mask_init).type(torch.FloatTensor).to(device))
    mask_phase = np.zeros((500,500))
    mask_phase = nn.Parameter(torch.from_numpy(mask_phase).type(torch.FloatTensor).to(device))
    mask_param = mask_real + 1j*mask_phase
    initial_learning_rate = 0.0002
    # adam optimizer
    optimizer = Adam(list(cnn.parameters()) + [mask_phase], lr=initial_learning_rate)
    # learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True, min_lr=1e-6)
    
    for epoch in np.arange(start_epoch,end_epoch):
        epoch_start_time = time.time()
        # print current epoch number
        print('='*20)
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('='*20)
        with torch.set_grad_enabled(True):
            for batch_ind, (xyz_np, Nphotons, targets) in enumerate(training_generator):
                print(batch_ind)
                print(xyz_np.shape)
                print(targets.shape)
                #img = torch.zeros((Nbatch,1,500,500)).type(torch.FloatTensor).to(device)
                targets = targets.to(device)
                Nphotons = Nphotons.to(device)
                xyz_np = xyz_np.to(device)
                xyz_np = xyz_np.squeeze()
                targets = targets.squeeze(0)
                Nphotons = Nphotons.squeeze(0)
                a = cnn(mask_param,xyz_np,Nphotons)
                
                break
                
        break
    
    
    
    
    