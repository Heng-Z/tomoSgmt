#!/usr/bin/env python3
import os
import mrcfile
import numpy as np
import sys
def run_mwr_predict(orig,outfile,model,weight,cubesize,cropsize,gpuID,batchsize):
    cmd = 'mwr3D_predict {} {} --model {} --weight {} --cubesize {} --cropsize {} --gpuID {} --batchsize {}'.format(orig, outfile, model, weight, cubesize, cropsize, gpuID, batchsize)

    os.system(cmd)

def predict_mask(mwr_tomo,outmask,model,weight,cubesize,cropsize,gpuID,batchsize):
    cmd = 'python3 /storage/heng/tomoSgmt/bin/sgmt_predict {} {} --model {} --weight {} --cubesize {} --cropsize {} --gpuID {} --batchsize {}'.format(mwr_tomo, outmask, model, weight, cubesize, cropsize, gpuID, batchsize)

    os.system(cmd)

def morph_process(mask,elem_len=9,save_labeled=None):
    # 1. closing and opening process of vesicle mask. 2. label the vesicles. 3. extract labeled individual vesicle vectors and output them.

    from skimage.morphology import opening, closing, erosion, cube
    from skimage.measure import label
    with mrcfile.open(mask) as f:
        tomo_mask = f.data 
    # transform mask into uint8
    bimask = np.round(tomo_mask).astype(np.uint8)
    closing_opening = closing(opening(bimask,cube(elem_len)),cube(elem_len))
    closing_opening = closing(bimask,cube(elem_len))
    # label the vesicles
    labeled = label(closing_opening)
    if save_labeled is not None:
        with mrcfile.new(save_labeled,overwrite=True) as f:
            f.set_data(labeled.astype(np.uint8))

    #for hollow labeled vesicles do not extract boundaries
    #labeled_boundary = labeled - erosion(label,cube(3))
    labeled_boundary = labeled
    #the number of labeled vesicle
    num = np.max(labeled)
    #vesicle list elements: np.where return point cloud positions whose shape is (3,N)
    vesicle_list = []
    for i in range(1,num+1):
        cloud = np.array(np.where(labeled_boundary == i))
        cloud = np.swapaxes(cloud,0,1)
        vesicle_list.append(cloud)
    
    return vesicle_list

def vesicle_measure(vesicle_list,outfile):
    sys.path.append('/storage/heng/tomoSgmt/bin/ellipsoid_fit_python/')
    import ellipsoid_fit as ef 
    import json
    results = []
    for i in range(len(vesicle_list)):
        print('fitting vesicle_',i)
        try:
            [center, evecs, radii]=ef.ellipsoid_fit(vesicle_list[i])            
            info={'name':'vesicle_'+str(i),'center':center.tolist(),'radii':radii.tolist(),'evecs':evecs.tolist()}
            results.append(info)
        except:
            print('bad vescle cloud points')

    vesicle_info={'vesicles':results}
    if outfile is not None:
        with open(outfile,"w") as out:
            json.dump(vesicle_info,out)
    return vesicle_info
    
def vesicle2json(vesicle_info,filename):
    import json
    with open(filename,"w") as out:
        json.dump(vesicle_info,out,)
        
    
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
    parser.add_argument('tomo_file', type=str, default=None, help='Your original tomo')
    parser.add_argument('output_file', type=str, default=None, help='output vesicles file name (xxx.json)')
    parser.add_argument('--dir', type=str, default='./', help='destination')
    parser.add_argument('--do_mwr',type=str2bool, nargs='?',const=True, default=True,help='if do mwr for input tomo')
    parser.add_argument('--mwr_file', type=str, default=None, help='if do mwr, the mwr file name to save as')
    parser.add_argument('--mask_file', type=str, default=None, help='the output vesicle mask file name')
    parser.add_argument('--mwrweight', type=str, default='results/modellast.h5' ,help='Weight file name to save')
    parser.add_argument('--sgmtweight', type=str, default='results/modellast.h5' ,help='Weight file name to save')
    parser.add_argument('--mwrmodel', type=str, default='model.json' ,help='Data file name to save')
    parser.add_argument('--sgmtmodel', type=str, default='model.json' ,help='Data file name to save')
    parser.add_argument('--gpuID', type=str, default='0,1,2,3', help='number of gpu for training')
    parser.add_argument('--cubesize', type=int, default=64, help='size of cube')
    parser.add_argument('--cropsize', type=int, default=96, help='crop size larger than cube for overlapping tile')
    parser.add_argument('--batchsize', type=int, default=8, help='batch size')
    args = parser.parse_args()

    # ****temperate model and weight****

    args.mwrweight = '/storage/heng/mwrtest3D/multitomo/bin4/t1_dgx/model_iter35.h5'
    args.mwrmodel = '/storage/heng/mwrtest3D/multitomo/bin4/t1_dgx/model.json'
    args.sgmtmodel = '/storage/heng/tomoSgmt/example/model.json'
    args.sgmtweight = '/storage/heng/tomoSgmt/example/modellast.h5'
    root_name = args.tomo_file.split('/')[-1].split('.')[0]
    if args.do_mwr is True:# if input is original tomo, do missing wedge correction first
        if args.mwr_file is None:
            args.mwr_file = args.dir+'/'+root_name+'-mwred.mrc'
        if args.mask_file is None:
            args.mask_file = args.dir+'/'+root_name+'-mask.mrc'
        run_mwr_predict(args.tomo_file, args.mwr_file, args.mwrmodel, args.mwrweight, args.cubesize, args.cropsize, args.gpuID,args.batchsize)

        predict_mask(args.mwr_file, args.mask_file, args.sgmtmodel, args.sgmtweight, args.cubesize, args.cropsize, args.gpuID,args.batchsize)
    else:
        if args.mask_file is None:
            args.mask_file = args.dir+'/'+root_name+'-mask.mrc'
        predict_mask(args.tomo_file, args.mask_file, args.sgmtmodel, args.sgmtweight, args.cubesize, args.cropsize, args.gpuID,args.batchsize)

    vesicle_list = morph_process(args.mask_file)
    vesicle_info = vesicle_measure(vesicle_list,args.output_file)
    with open(outfile, 'w') as fp:
        json.dump(vesicle_info, fp,indent=4)
