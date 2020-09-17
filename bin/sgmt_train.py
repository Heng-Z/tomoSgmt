#!/usr/bin/env python3
#tomoSgmt
#training a model to map from original tomo to vesicle binary mask
import mrcfile
from mwr.training.train import train3D_seq
import numpy as np
from utils import gene_train_data

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

    history = train3D_seq('results/model_sgmt.h5', data_dir = settings.data_folder, epochs = settings.epochs, steps_per_epoch = settings.steps_per_epoch,  batch_size = settings.batch_size, n_gpus = settings.ngpus, loss='binary_crossentropy',last_activation = 'sigmoid')
