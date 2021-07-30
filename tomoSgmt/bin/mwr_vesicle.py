#!/usr/bin/env python3

import os
import mrcfile
import numpy as np
import sys
from scipy.sparse import csr_matrix
import json
from tomoSgmt.bin.sgmt_predict import predict_new
import math
#from tomoSgmt.bin.post_proc import morph_process, vesicle_measure, vesicle_rendering
from tomoSgmt.bin.morph import morph_process, vesicle_measure, vesicle_rendering
  
def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == "__main__":
    import argparse
    import json
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--tomo_file', type=str, default=None, help='Your original tomo')
    parser.add_argument('--output_file', type=str, default=None, help='output vesicles file name (xxx.json)')
    parser.add_argument('--output_file_in_area', type=str, default=None, help='output vesicles in presyn file name (xxx.json)')
    parser.add_argument('--dir', type=str, default='./', help='destination')
    parser.add_argument('--mwr_file', type=str, default=None, help='the output vesicle mask file name')
    parser.add_argument('--mask_file', type=str, default=None, help='the output vesicle mask file name')
    parser.add_argument('--mwrmodel', type=str, default=None ,help='model for mwr, skip mwr if None')
    parser.add_argument('--sgmtmodel', type=str, default=None ,help='model for vesicle sgmt, skip if None')
    parser.add_argument('--gpuID', type=str, default='0,1,2,3', help='number of gpu for training')
    parser.add_argument('--neighbor_in', type=int, default=5, help='number of neighbor channels')
    parser.add_argument('--neighbor_out', type=int, default=1, help='number of neighbor channels')
    parser.add_argument('--sidelen', type=int, default=128, help='side length during 2D training')
    parser.add_argument('--render', type=str, default=None, help='if draw fitted vesicles on a new tomo')
    parser.add_argument('--render_in', type=str, default=None, help='if draw fitted vesicles which in presyn on a new tomo')
    parser.add_argument('--batchsize', type=int, default=8, help='batch size')
    parser.add_argument('--min_radius', type=int, default=10, help='minimal radius of targeting vesicles')
    parser.add_argument('--area_file', type=str, default=None, help='.point file which defines interested area')
    args = parser.parse_args()

    # set some default file name
    
    root_name = args.mwr_file.split('/')[-1].split('.')[0]
    if args.area_file is not None:
        if args.output_file_in_area is None:
            args.output_file_in_area =  args.dir+'/'+root_name+'-vesicle-in-area.json'
    if args.output_file is None:
        args.output_file =  args.dir+'/'+root_name+'-vesicle-info.json'
    # predict vesicle mask if provided sgmtmodel
    if args.sgmtmodel is not None:
        if args.mask_file is None:
            args.mask_file = args.dir+'/'+root_name+'-mask.mrc'
        
        # predict_mask(args.mwr_file, args.mask_file, args.sgmtmodel,args.neighbor_in,args.neighbor_out,args.sidelen,args.gpuID,args.batchsize)
        predict_new(args.mwr_file,args.mask_file,args.sgmtmodel,sidelen=args.sidelen,neighbor_in=args.neighbor_in,neighbor_out=args.neighbor_out)
    
    min_radius = args.min_radius
    # save raw vesicle mask
    with mrcfile.open(args.mask_file) as m:
        bimask =  m.data
    shape = bimask.shape
    print('begin morph process')
    vesicle_list = morph_process(args.mask_file,radius=min_radius)
    print('done morph process')
    [vesicle_info, in_vesicle_info] = vesicle_measure(vesicle_list,min_radius,args.output_file,args.output_file_in_area,area_file=args.area_file)
    print('done vesicle measuring')
   
    # print('number of vesicle in given presyn:{}'.format(len()))
    #print(results_in)
    if args.render is not None:
        ves_tomo = vesicle_rendering(args.output_file,shape)
        with mrcfile.new(args.render,overwrite=True) as n:
            n.set_data(ves_tomo)
    
    if args.render_in is not None:
        ves_in_tomo = vesicle_rendering(args.output_file_in_area, shape)
        with mrcfile.new(args.render_in, overwrite=True) as n_in:
            n_in.set_data(ves_in_tomo)
