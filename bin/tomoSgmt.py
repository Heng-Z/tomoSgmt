#!/usr/bin/env python3
#tomoSgmt
#training a model to map from original tomo to vesicle binary mask
import mrcfile
from mwr.models import unet2
from mwr.util.image import *
import numpy as np

def gene_train_data(settings):
    with mrcfile.open(settings.orig_tomo) as o:
        orig_tomo=o.data 
    
    with mrcfile.open(settings.mask_tomo) as m:
        mask_tomo=m.data
    #create random center seeds and crop subtomos
    #10% ncube will be saved as test_set
    seeds1=create_cube_seeds(orig_tomo,settings.ncube,settings.cropsize)
    seeds2=create_cube_seeds(orig_tomo,int(settings.ncube*0.1),settings.cropsize)

    orig_subtomos=crop_cubes(orig_tomo,seeds1,settings.cropsize)
    mask_subtomos=crop_cubes(mask_tomo,seeds1,settings.cropsize)

    orig_test_subtomos=crop_cubes(orig_tomo,seeds2,settings.cropsize)
    mask_test_subtomos=crop_cubes(mask_tomo,seeds2,settings.cropsize)

    for j,s in enumerate(orig_subtomos):
        with mrcfile.new('{}/train_x/{}_{:0>6d}.mrc'.format(settings.data_folder, 'subtomo_x',j), overwrite=True) as output_mrc:
            output_mrc.set_data(s.astype(np.float32))
    
    for j,s in enumerate(mask_subtomos):
        with mrcfile.new('{}/train_y/{}_{:0>6d}.mrc'.format(settings.data_folder, 'subtomo_y',j), overwrite=True) as output_mrc:
            output_mrc.set_data(s) 

    for j,s in enumerate(orig_test_subtomos):
        with mrcfile.new('{}/test_x/{}_{:0>6d}.mrc'.format(settings.data_folder, 'test_x',j), overwrite=True) as output_mrc:
            output_mrc.set_data(s.astype(np.float32)) 

    for j,s in enumerate(mask_test_subtomos):
        with mrcfile.new('{}/test_y/{}_{:0>6d}.mrc'.format(settings.data_folder, 'test_y',j), overwrite=True) as output_mrc:
            output_mrc.set_data(s) 

if __name__=='__main__':
    import os
    import sys
    sys.path.insert(0,os.getcwd()) 
    import settings
    settings.ngpus = len(settings.gpuID.split(','))
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=settings.gpuID  # specify which GPU(s) to be used
    if not settings.cropped:
        if not os.path.isdir(settings.data_folder):
            print('data_folder does not exits, mkdir')
            os.makedirs(settings.data_folder)
            dirs_tomake = ['train_x','train_y', 'test_x', 'test_y']
            for d in dirs_tomake:
                os.makedirs('{}/{}'.format(settings.data_folder, d))
        gene_train_data(settings)

    history = unet2.train3D_seq('results/model_sgmt.h5', data_folder = settings.data_folder, epochs = settings.epochs, steps_per_epoch = settings.steps_per_epoch,  batch_size = settings.batch_size, n_gpus = settings.ngpus, loss='binary_crossentropy')
