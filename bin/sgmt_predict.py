#!/usr/bin/env python3

import numpy as np
import mrcfile
import tensorflow as tf
from tomoSgmt.bin.utils import Patch,normalize
from tqdm import tqdm


def predict_new(mrc,output,model,sidelen=128,neighbor_in=5,neighbor_out=1, batch_size=8,gpuID='0'):
    import os
    import logging
    from tensorflow.keras.models import load_model

    logging.basicConfig(filename='myapp.log', level=logging.INFO)
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=gpuID
    ngpus = len(gpuID.split(','))
    # model = load_model(args.model)
    logging.info('gpuID:{}'.format(gpuID))
    if ngpus >1:
        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            kmodel = load_model(model)
    else:
        kmodel = load_model(model)

    with mrcfile.open(mrc) as mrcData:
        real_data = mrcData.data.astype(np.float32)
        real_data = normalize(real_data,percentile=False)    
    p = Patch(real_data)
    patch_list = p.to_patches(sidelen=sidelen,neighbor=neighbor_in)
    data =np.swapaxes(np.array(patch_list),1,-1)
    print(data.shape)
    N = batch_size * ngpus *4
    num_batches = data.shape[0]
    if num_batches%N == 0:
        append_number = 0
    else:
        append_number = N - num_batches%N
    data = np.append(data, data[0:append_number], axis = 0)
    num_big_batch = data.shape[0]//N
    outData = np.zeros((*data.shape[0:-1],neighbor_out))
    for i in tqdm(range(num_big_batch)):
        outData[i*N:(i+1)*N] = kmodel.predict(data[i*N:(i+1)*N], batch_size= batch_size,verbose=0)
    outData = outData[0:num_batches]
    # patch_predicted = kmodel.predict(data_to_predict,batch_size=batch_size,verbose=1)
    restored_tomo = p.restore_tomo(np.swapaxes(outData,1,-1),neighbor=neighbor_out)
    with mrcfile.new(output, overwrite=True) as output_mrc:
        output_mrc.set_data((np.round(restored_tomo).astype(np.uint8)>0).astype(np.uint8))
        


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
    parser.add_argument('--neighbor_in', type=int, default=5, help='number of neighbor channels')
    parser.add_argument('--neighbor_out', type=int, default=1, help='number of neighbor channels')
    parser.add_argument('--sidelen', type=int, default=128, help='side length during 2D training')
    args = parser.parse_args() 

    # predict(args.model,args.mrc_file,args.output_file, cubesize=args.cubesize, cropsize=args.cropsize, batchsize=args.batchsize, gpuID=args.gpuID, )

    predict_new(args.mrc_file,args.output_file,args.model,sidelen=args.sidelen,neighbor_in=args.neighbor_in,neighbor_out=args.neighbor_out)
