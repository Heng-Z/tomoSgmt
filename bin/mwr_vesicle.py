#!/usr/bin/env python3
import os
import mrcfile
import numpy as np
import sys
from scipy.sparse import csr_matrix
import json
from tomoSgmt.bin.sgmt_predict import predict_new
def run_mwr_predict(orig,outfile,model,weight,cubesize,cropsize,gpuID,batchsize):
    # cmd = 'mwr3D_predict {} {} --model {} --weight {} --cubesize {} --cropsize {} --gpuID {} --batchsize {}'.format(orig, outfile, model, weight, cubesize, cropsize, gpuID, batchsize)
    # os.system(cmd)
    pass
    

def predict_mask(mwr_tomo,outmask,model,neighbor_in,neighbor_out,sidelen,gpuID,batchsize):
    cmd = 'python3 /storage/heng/tomoSgmt/bin/sgmt_predict.py {} {} --model {}  --gpuID {} --batchsize {} --neighbor_in {} --neighbor_out {}--sidelen {}'.format(mwr_tomo, outmask, model, gpuID, batchsize,neighbor_in,neighbor_out,sidelen)

    os.system(cmd)
    print('predicting mask')

def morph_process(mask,elem_len=3,radius=15,save_labeled=None):
    # 1. closing and opening process of vesicle mask. 2. label the vesicles.
    # 3. exclude fasle vesicles by counting their volumes and thresholding, return only vesicle binary mask
    # 4. extract boundaries and labels them
    # 5. extract labeled individual vesicle boundary, convert into points vectors and output them.
    from skimage.morphology import opening, closing, erosion, cube
    from skimage.measure import label
    with mrcfile.open(mask) as f:
        tomo_mask = f.data 
    # transform mask into uint8
    bimask = np.round(tomo_mask).astype(np.uint8)
    closing_opening = closing(opening(bimask,cube(elem_len)),cube(elem_len))
    # label all connected regions
    labeled = label(closing_opening)
    idx = get_indices_sparse(labeled)
    num = np.max(labeled)
    for i in range(1,num+1):
        if idx[i][0].shape[0] <radius**3*4 :    
            labeled[idx[i][0],idx[i][1],idx[i][2]] = 0
    filtered  = (labeled >= 1).astype(np.uint8)
    print('complete filtering')
    boundaries = filtered - erosion(filtered,cube(3))
    # label the boundaries of vesicles 
    bd_labeled = label(boundaries)
    #the number of labeled vesicle
    num = np.max(bd_labeled)
    #vesicle list elements: np.where return point cloud positions whose shape is (3,N)
    idx = get_indices_sparse(bd_labeled)
    vesicle_list = [np.swapaxes(np.array(idx[i]),0,1) for i in range(1,num+1)]
    # for i in range(1,num+1):
    #     cloud = np.array(np.where(bd_labeled == i))
    #     cloud = np.swapaxes(cloud,0,1)
    #     vesicle_list.append(cloud)
    
    return vesicle_list

def vesicle_measure(vesicle_list,min_radius,outfile):
    from tomoSgmt.bin.ellipsoid import ellipsoid_fit as ef
    results = []
    def if_normal(radii,threshold=0.15):
        if np.std(radii)/np.mean(radii) >threshold:
            a = False
        elif np.mean(radii) < min_radius or np.mean(radii) > min_radius*3:
            a = False
        else: 
            a = True
        return a

    for i in range(len(vesicle_list)):
        print('fitting vesicle_',i)
        [center, evecs, radii]=ef.ellipsoid_fit(vesicle_list[i])            
        info={'name':'vesicle_'+str(i),'center':center.tolist(),'radii':radii.tolist(),'evecs':evecs.tolist()}
        if if_normal(radii):
            results.append(info)
        else:
            print('bad vescle {}'.format(i))
    #return vesicle information dict and save as json
    vesicle_info={'vesicles':results}
    if outfile is not None:
        with open(outfile,"w") as out:
            json.dump(vesicle_info,out)
    return vesicle_info

def compute_M(data):
    cols = np.arange(data.size)
    return csr_matrix((cols, (data.ravel(), cols)),
                      shape=(data.max()+1, data.size))

def get_indices_sparse(data):
    M = compute_M(data)
    return [np.unravel_index(row.data, data.shape) for row in M]

def vesicle2json(vesicle_info,filename):
    import json
    with open(filename,"w") as out:
        json.dump(vesicle_info,out)
        
    
def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def vesicle_rendering(vesicle_file,tomo_dims):
    # vesicle file can be json or a info list
    from tomoSgmt.utils import make_ellipsoid as mk 
    from skimage.morphology import closing, cube
    if type(vesicle_file) is str:
        with open(vesicle_file) as f:
            ves = json.load(f)
        vesicle_info = ves['vesicles']
    else:
        vesicle_info = vesicle_file
    vesicle_tomo = np.zeros(np.array(tomo_dims)+np.array([30,30,30]),dtype=np.uint8)
    for i,vesicle in enumerate(vesicle_info):
        print(i)
        ellip_i = mk.ellipsoid_point(
            np.array(vesicle['radii']),
            np.array(vesicle['center']),
            np.array(vesicle['evecs'])
        )
        #ellip_i is an array (N,3) of points of a filled ellipsoid 
        vesicle_tomo[ellip_i[:,0],ellip_i[:,1],ellip_i[:,2]] = 1
    vesicle_tomo = closing(vesicle_tomo,cube(3))
    return vesicle_tomo[0:tomo_dims[0],0:tomo_dims[1],0:tomo_dims[2]]


if __name__ == "__main__":
    import argparse
    import json
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--tomo_file', type=str, default=None, help='Your original tomo')
    parser.add_argument('--output_file', type=str, default=None, help='output vesicles file name (xxx.json)')
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
    parser.add_argument('--batchsize', type=int, default=8, help='batch size')
    parser.add_argument('--min_radius', type=int, default=10, help='minimal radius of targeting vesicles')
    args = parser.parse_args()

    # ****temperate model and weight****

    # args.mwrweight = '/storage/heng/mwrtest3D/multitomo/bin4/t1_dgx/model_iter35.h5'
    # args.mwrmodel = '/storage/heng/mwrtest3D/multitomo/bin4/t1_dgx/model.json'
    # args.sgmtmodel = '/storage/heng/tomoSgmt/example/model.json'
    # args.sgmtweight = '/storage/heng/tomoSgmt/example/modellast.h5'
    root_name = args.mwr_file.split('/')[-1].split('.')[0]
    # if args.mwrmodel is not None:# if input is original tomo, do missing wedge correction first
    #     mwr_out = args.dir+'/'+root_name+'-mwr.mrc'
    #     run_mwr_predict(args.tomo_file, args.mwr_file, args.mwrmodel, args.mwrweight, args.cubesize, args.cropsize, args.gpuID,args.batchsize)
    
    if args.sgmtmodel is not None:
        if args.mask_file is None:
            args.mask_file = args.dir+'/'+root_name+'-mask.mrc'
        
        # predict_mask(args.mwr_file, args.mask_file, args.sgmtmodel,args.neighbor_in,args.neighbor_out,args.sidelen,args.gpuID,args.batchsize)
        predict_new(args.mwr_file,args.mask_file,args.sgmtmodel,sidelen=args.sidelen,neighbor_in=args.neighbor_in,neighbor_out=args.neighbor_out)
    
    min_radius = args.min_radius

    with mrcfile.open(args.mask_file) as m:
        bimask =  m.data
    shape = bimask.shape
    print('begin morph process')
    vesicle_list = morph_process(args.mask_file,radius=min_radius)
    print('done morph process')
    vesicle_info = vesicle_measure(vesicle_list,min_radius,args.output_file)
    print('done vesicle measuring')
    if args.render is not None:
        ves_tomo = vesicle_rendering(args.output_file,shape)
        with mrcfile.new(args.render,overwrite=True) as n:
            n.set_data(ves_tomo)
