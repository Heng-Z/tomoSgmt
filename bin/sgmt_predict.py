#!/usr/bin/env python3

import numpy as np
from mwr.util.norm import normalize
from mwr.util.toTile import reform3D,DataWrapper
import mrcfile
from mwr.util.image import *
import tensorflow as tf
def predict(model,mrc,output,cubesize=64, cropsize=96, batchsize=8, gpuID='0', if_percentile=True):
    import os
    # import tensorflow.keras
    import logging
    from tensorflow.keras.models import load_model

    logging.basicConfig(filename='myapp.log', level=logging.INFO)
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpuID
    ngpus = len(gpuID.split(','))
    # model = load_model(args.model)
    logging.info('gpuID:{}'.format(args.gpuID))
    if ngpus >1:
        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            model = load_model(args.model)
    else:
        model = load_model(args.model)

    N = batchsize * ngpus

    if True:
        if True:
            root_name = mrc.split('/')[-1].split('.')[0]
            print('predicting:{}'.format(root_name))
            with mrcfile.open(mrc) as mrcData:
                real_data = mrcData.data.astype(np.float32)
            real_data = normalize(-real_data,percentile=if_percentile)
            data=np.expand_dims(real_data,axis=-1)
            reform_ins = reform3D(data)
            data = reform_ins.pad_and_crop_new(cubesize,cropsize)
            #to_predict_data_shape:(n,cropsize,cropsize,cropsize,1)
            #imposing wedge to every cubes
            #data=wedge_imposing(data)
            num_batches = data.shape[0]
            if num_batches%N == 0:
                append_number = 0
            else:
                append_number = N - num_batches%N
            data = np.append(data, data[0:append_number], axis = 0)

            outData=model.predict(data, batch_size= batchsize,verbose=1)

            outData = outData[0:num_batches]
            outData=reform_ins.restore_from_cubes_new(outData.reshape(outData.shape[0:-1]), cubesize, cropsize)
            outData=np.around(outData).astype(np.uint8)
            with mrcfile.new(output, overwrite=True) as output_mrc:
                output_mrc.set_data(outData)
    return 0

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('mrc_file', type=str, default=None, help='Your mrc file')
    parser.add_argument('output_file', type=str, default=None, help='output mrc file')
    # parser.add_argument('--weight', type=str, default='results/modellast.h5' ,help='Weight file name to save')
    parser.add_argument('--model', type=str, default='model.json' ,help='Data file name to save')
    parser.add_argument('--gpuID', type=str, default='0,1,2,3', help='number of gpu for training')
    parser.add_argument('--cubesize', type=int, default=64, help='size of cube')
    parser.add_argument('--cropsize', type=int, default=96, help='crop size larger than cube for overlapping tile')
    parser.add_argument('--batchsize', type=int, default=8, help='batch size')

    args = parser.parse_args() 

    predict(args.model,args.mrc_file,args.output_file, cubesize=args.cubesize, cropsize=args.cropsize, batchsize=args.batchsize, gpuID=args.gpuID, )
