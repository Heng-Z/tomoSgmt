#!/usr/bin/env python3
#tomoSgmt
#training a model to map from original tomo to vesicle binary mask
import mrcfile
# from mwr.training.train import train3D_seq
# from mwr.models.unet2 import train3D_seq
import numpy as np
import os
from tomoSgmt.bin.utils import gene_train_data, gene_2d_training_data
from tomoSgmt.unet.unet2D import build_compiled_model
import tensorflow as tf
if __name__=='__main__':
    import os
    import sys
    sys.path.insert(0,os.getcwd()) 
    import settings
    settings.ngpus = len(settings.gpuID.split(','))
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=settings.gpuID  # specify which GPU(s) to be used
    # if not settings.cropped:
    # if not os.path.isdir(settings.data_folder):
    #     print('data_folder does not exits, mkdir')
    #     os.makedirs(settings.data_folder)
    #     dirs_tomake = ['train_x','train_y', 'test_x', 'test_y']
    #     for d in dirs_tomake:
    #         os.makedirs('{}/{}'.format(settings.data_folder, d))
    #     # gene_train_data(settings)
    train_data, test_data = gene_2d_training_data(settings.orig_tomo,settings.mask_tomo,sample_mask=settings.sample_mask,num=400,sidelen=128,neighbor=5)

    model = build_compiled_model()
    model.summary()
    strategy  = tf.distribute.MirroredStrategy()
    with strategy.scope():
        model = build_compiled_model()
    model.summary()
    model.fit(train_data[0],train_data[1], validation_data=test_data,
                                  epochs=settings.epochs, steps_per_epoch=settings.steps_per_epoch, verbose=1)
    # history = train3D_seq(settings.model_name, data_dir = settings.data_folder, 
    # epochs = settings.epochs, steps_per_epoch = settings.steps_per_epoch,  
    # batch_size = settings.batch_size, n_gpus = settings.ngpus, 
    # loss='binary_crossentropy',last_activation = 'sigmoid',residual = False)

    # history = train3D_seq(settings.model_name, data_folder = settings.data_folder, 
    # epochs = settings.epochs, steps_per_epoch = settings.steps_per_epoch,  
    # batch_size = settings.batch_size, n_gpus = settings.ngpus, 
    # loss='binary_crossentropy')

    model.save(settings.model_name)
