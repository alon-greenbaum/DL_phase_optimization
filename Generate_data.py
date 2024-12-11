# -*- coding: utf-8 -*-

import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import pickle
from PIL import Image
import argparse
from data_utils import generate_batch, complex_to_tensor

def gen_data(config):

    ntrain = config['ntrain']
    nvalid = config['nvalid']
    batch_size_gen = config['batch_size_gen']
    particle_spatial_range_xy = config['particle_spatial_range_xy']
    particle_spatial_range_z = config['particle_spatial_range_z']
    num_particles_range =config['num_particles_range']




    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu")
    torch.backends.cudnn.benchmark = True


    # set random seed for repeatability
    torch.manual_seed(99)
    np.random.seed(50)

    path_train = 'traininglocations/'
    if not (os.path.isdir(path_train)):
        os.mkdir(path_train)
        
    # print status
    print('=' * 50)
    print('Sampling examples for training')
    print('=' * 50)

    # locations for phase mask learning are saved in batches of 16 for convenience

    # calculate the number of training batches to sample
    ntrain_batches = int(ntrain/batch_size_gen)
    
    # generate examples for training
    labels_dict = {}
    for i in range(ntrain_batches):
        # sample a training example
        xyz, Nphotons = generate_batch(batch_size_gen, num_particles_range, particle_spatial_range_xy, particle_spatial_range_z)
        labels_dict[str(i)] = {'xyz':xyz, 'N': Nphotons}
        # print number of example
        print('Training Example [%d / %d]' % (i + 1, ntrain_batches))
        
    
    nvalid_batches = int(nvalid/ batch_size_gen)
    for i in range(nvalid_batches):
        xyz, Nphotons = generate_batch(batch_size_gen, num_particles_range, particle_spatial_range_xy, particle_spatial_range_z)
        labels_dict[str(i+ntrain_batches)] = {'xyz':xyz, 'N': Nphotons}
        print('Validation Example [%d / %d]' % (i + 1, nvalid_batches))
    
    path_labels = path_train + 'labels.pickle'
    with open(path_labels, 'wb') as handle:
        pickle.dump(labels_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
if __name__ == '__main__':

    config = {
        "ntrain" : 10000,
        # number of emitters for the validation set
        "nvalid" : 1000,
        "batch_size_gen" : 2,
        # Number of emitters per image
        "num_particles_range" : [450, 550],
        "particle_spatial_range_xy" : range(15, 185),
        # um, to avoid edges of the image so 15 um away
        "particle_spatial_range_z" : range(-10, 11, 1)  # um
    }

    gen_data(config)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    