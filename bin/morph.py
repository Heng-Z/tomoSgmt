#! user/bin/env python3

import os
import mrcfile
import numpy as np
import sys
from scipy.sparse import csr_matrix
import json
#from tomoSgmt.bin.sgmt_predict import predict_new
import math

def morph_process(mask,elem_len=1,radius=10,save_labeled=None):
    # 1. closing and opening process of vesicle mask. 2. label the vesicles.
    # 3. exclude fasle vesicles by counting their volumes and thresholding, return only vesicle binary mask
    # 4. extract boundaries and labels them
    # 5. extract labeled individual vesicle boundary, convert into points vectors and output them.
    from skimage.morphology import opening, closing, erosion, cube, dilation
    from skimage.measure import label
    with mrcfile.open(mask) as f:
        tomo_mask = f.data 
    # transform mask into uint8
    bimask = np.round(tomo_mask).astype(np.uint8)
    shape = bimask.shape

    # extract labeled mask whose area more than a threshold 
    # (just after prediction, some vesicles will be predicted to be connected)
    area_thre = radius**3
    #bimask = dilation(bimask, cube(2))
    labeled_pre = label(bimask)
    sup_pro = np.zeros(labeled_pre.shape)
    pre_pro = np.zeros(labeled_pre.shape)
    idx_pre = get_indices_sparse(labeled_pre)
    num_pre = np.max(labeled_pre)

    for i in range(1, num_pre+1):
        if idx_pre[i][0].shape[0] > area_thre*15:
            pre_pro[idx_pre[i][0],idx_pre[i][1],idx_pre[i][2]] = 1
            labeled_pre[idx_pre[i][0],idx_pre[i][1],idx_pre[i][2]] = 0
    labeled_pre[labeled_pre > 1] = 1

    kernel_pre = cube(11)
    pre_pro = opening(pre_pro, kernel_pre)
    pre_pro = erosion(pre_pro, cube(2))
    labeled_pre_pro = label(pre_pro) #process linked vesicles just after prediction, Part 1

    # for other vesicles
    if True:
        kernel_xy = np.reshape([1 for i in range(9)], (3, 3, 1))
        closing_opening_xy = closing(labeled_pre, kernel_xy)
    else:
        closing_opening_xy = labeled_pre
    if True:
        kernel = np.reshape([1 for i in range(12)], (2, 2, 3))
        closing_opening = closing(closing_opening_xy, kernel)
    else:
        closing_opening = closing_opening_xy

    # label all connected regions
    labeled = label(closing_opening)
    post_pro = np.zeros(labeled.shape)
    
    idx = get_indices_sparse(labeled)
    num = np.max(labeled)

    for i in range(1,num+1):
        if idx[i][0].shape[0] < area_thre:    
            labeled[idx[i][0],idx[i][1],idx[i][2]] = 0
            if idx[i][0].shape[0] > 0.2*area_thre:
                sup_pro[idx[i][0],idx[i][1],idx[i][2]] = 1 #for very bad prediction, can only do ellipse fit
        elif idx[i][0].shape[0] <= 2*area_thre and idx[i][0].shape[0] > area_thre:
            sup_pro[idx[i][0],idx[i][1],idx[i][2]] = 1
        elif idx[i][0].shape[0] > area_thre*12:
            post_pro[idx[i][0],idx[i][1],idx[i][2]] = 1 #record positon of linked vesicles from closing, Part 2
            labeled[idx[i][0],idx[i][1],idx[i][2]] = 0 #main vesicles here, Part 3

    labeled = label(labeled) # update num of Part3
    num = np.max(labeled)

    # process for Part2
    kernel_p = cube(11)
    post_pro = opening(post_pro, kernel_p)
    #post_pro = erosion(post_pro, cube(3))
    labeled_post_pro = label(post_pro)
    num_post = np.max(labeled_post_pro)

    labeled_post_pro += num
    labeled_post_pro[labeled_post_pro == num] = 0 # update num of Part2

    num += num_post  # update total num of vesicles(except pre_pro)
    labeled_pre_pro += num
    labeled_pre_pro[labeled_pre_pro == num] = 0 # update num of label for part 1
    labeled = labeled + labeled_post_pro + labeled_pre_pro
    num = np.max(labeled)

    # for supplementary proccess
    labeled_sup_pro = label(sup_pro)
    sup_filtered = (labeled_sup_pro >= 1).astype(np.uint8)
    #sup_boundaries = sup_filtered - erosion(sup_filtered, cube(3))
    sup_bd_labeled = label(sup_filtered)
    sup_num = np.max(sup_bd_labeled)
    sup_idx = get_indices_sparse(sup_bd_labeled)
    vesicle_list_sup = [np.swapaxes(np.array(sup_idx[i]),0,1) for i in range(1, sup_num+1)]

    # for main vesicles
    filtered  = (labeled >= 1).astype(np.uint8)
    print('complete filtering')
    boundaries = filtered - erosion(filtered,cube(3))
    # label the boundaries of vesicles 
    bd_labeled = label(boundaries)
    # the number of labeled vesicle
    num = np.max(bd_labeled)
    # vesicle list elements: np.where return point cloud positions whose shape is (3,N)
    idx = get_indices_sparse(bd_labeled)
    vesicle_list = [np.swapaxes(np.array(idx[i]),0,1) for i in range(1,num+1)]
    # for i in range(1,num+1):
    #     cloud = np.array(np.where(bd_labeled == i))
    #     cloud = np.swapaxes(cloud,0,1)
    #     vesicle_list.append(cloud)
    
    return vesicle_list, vesicle_list_sup, shape


def vesicle_measure(vesicle_list, vesicle_list_sup, shape, min_radius, outfile, outfile_in_area, area_file=None):
    from tomoSgmt.bin.ellipsoid import ellipsoid_fit as ef
    from skimage.morphology import erosion
    from skimage.measure import label

    results = []
    results_in = []
    sup_results = []
    sup_results_in = []
    P = get_area_points(area_file)
    CH = Graham_scan(P)
    global in_count
    global sup_in_count
    in_count = 0
    sup_in_count = 0
    bd_shape = 0

    def if_normal(radii,threshold=0.22):
        if np.std(radii)/np.mean(radii) >threshold:
            a = False
        elif np.mean(radii) < 0.6*min_radius or np.mean(radii) > min_radius*4:
            a = False
        else:
            a = True
        return a
    
    def if_normal_opp(radii, threshold=0.30):
        if np.std(radii)/np.mean(radii) < threshold or np.std(radii)/np.mean(radii) > 3*threshold:
            a = False
        elif np.mean(radii) < 0.6*min_radius or np.mean(radii) > min_radius*4:
            a = False
        else:
            a = True
        return a

    for i in range(len(vesicle_list)):
        print('fitting vesicle_',i)
        [center, evecs, radii]=ef.ellipsoid_fit(vesicle_list[i])
        if if_normal(radii):
            info={'name':'vesicle_'+str(i),'center':center.tolist(),'radii':radii.tolist(),'evecs':evecs.tolist()}
            results.append(info)
            # check whether a vesicle in given presyn
            c = np.delete(center, 0)
            c[0], c[1] = c[1], c[0]
            if Check(CH, len(CH), c):
                results_in.append(info)
                print('in vesicle {}'.format(i))
                in_count = in_count+1
        else:
            print('bad vesicle {}'.format(i))
    '''
    for i in range(len(vesicle_list_sup)):
        print('fitting ellipse vesicle_',i)
        z = vesicle_list_sup[i][:, 0]
        Zc = np.mean(z)
        X_temp = np.delete(vesicle_list_sup[i], 0, axis=1)
        X = np.zeros(X_temp.shape)
        X[:, 0] = X_temp[:, 1]
        X[:, 1] = X_temp[:, 0]
        mask = np.zeros((shape[2], shape[1]))
        for k in range(len(X)):
            mask[int(X[k][0]), int(X[k][1])] = 1
        kernel = np.reshape([1, 1, 1, 1, 1, 1, 1, 1, 1], (3, 3))
        boundaries = mask - erosion(mask, kernel)
        bd_labeled = label(boundaries)
        idx = get_indices_sparse(bd_labeled)
        for id in range(1, len(idx)):
            bd_shape += idx[id][0].shape[0]
        bd_x = np.hstack((idx[j][0] for j in range(1, len(idx))))
        bd_y = np.hstack((idx[j][1] for j in range(1, len(idx))))
        #vesicle_sup = np.dstack((bd_x, bd_y))

        [center, evecs, radii] = ef.ellipse_fit(bd_x, bd_y, Zc)
        if if_normal(radii):
            info_sup = {'name':'vesicle_'+str(i+len(vesicle_list)), 'center':center.tolist(), 'radii':radii.tolist(), 'evecs':evecs.tolist()}
            #sup_results.append(info_sup)
            results.append(info_sup)
            c = np.delete(center, 0)
            c[0], c[1] = c[1], c[0]
            if Check(CH, len(CH), c):
                #sup_results_in.append(info_sup)
                results_in.append(info_sup)
                print('in vesicle {}'.format(i+len(vesicle_list)))
                in_count = in_count + 1
                sup_in_count = sup_in_count + 1
        else:
            print('bad vesicle {}'.format(i+len(vesicle_list)))
    '''    
    # return vesicle information dict and save as json
    vesicle_info={'vesicles':results}
    in_vesicle_info={'vesicles':results_in}

    if outfile is not None:    
        with open(outfile,"w") as out:
            json.dump(vesicle_info,out)

    if outfile_in_area is not None:
        with open(outfile_in_area,"w") as out_in:
            json.dump(in_vesicle_info, out_in)

    return [vesicle_info, in_vesicle_info]


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
        # ellip_i is an array (N,3) of points of a filled ellipsoid
        vesicle_tomo[ellip_i[:,0],ellip_i[:,1],ellip_i[:,2]] = 1
    vesicle_tomo = closing(vesicle_tomo,cube(3))
    return vesicle_tomo[0:tomo_dims[0],0:tomo_dims[1],0:tomo_dims[2]]


def get_bottom_point(points):
    # get the first point for Graham_scan
    # cuz the farthest point must be one of the vertex of convex hull
    min_index = 0
    n = len(points)
    for i in range(0, n):
        if points[i][1] < points[min_index][1] or (points[i][1] == points[min_index][1] and points[i][0] < points[min_index][0]):
            min_index = i
    return min_index


def sort_polar_angle_cos(points, center_point):
    # sort by polar angle with center point(with cos value)
    n = len(points)
    cos_value = []
    rank = []
    norm_list = []
    for i in range(0, n):
        point_ = points[i]
        point = [point_[0]-center_point[0], point_[1]-center_point[1]]
        rank.append(i)
        norm_value = math.sqrt(point[0]*point[0] + point[1]*point[1])
        norm_list.append(norm_value)
        if norm_value == 0:
            cos_value.append(1)
        else:
            cos_value.append(point[0] / norm_value)

    for i in range(0, n-1):
        index = i + 1
        while index > 0:
            if cos_value[index] > cos_value[index-1] or (cos_value[index] == cos_value[index-1] and norm_list[index] > norm_list[index-1]):
                temp = cos_value[index]
                temp_rank = rank[index]
                temp_norm = norm_list[index]
                cos_value[index] = cos_value[index-1]
                rank[index] = rank[index-1]
                norm_list[index] = norm_list[index-1]
                cos_value[index-1] = temp
                rank[index-1] = temp_rank
                norm_list[index-1] = temp_norm
                index = index-1
            else:
                break
    sorted_points = []
    for i in rank:
        sorted_points.append(points[i])

    return sorted_points


def vector_angle(vector):

    norm_ = math.sqrt(vector[0]*vector[0] + vector[1]*vector[1])
    if norm_ == 0:
        return 0

    angle = math.acos(vector[0]/norm_)
    if vector[1] >= 0:
        return angle
    else:
        return 2*math.pi - angle


def Cross(p1, p2, p0):

    return ((p1[0]-p0[0])*(p2[1]-p0[1])-(p2[0]-p0[0])*(p1[1]-p0[1]))


def Graham_scan(points):
    # output a vertex set by anticlockwise of convex hull

    bottom_index = get_bottom_point(points)
    bottom_point = points.pop(bottom_index)
    sorted_points = sort_polar_angle_cos(points, bottom_point)

    m = len(sorted_points)

    stack = []
    stack.append(bottom_point)
    stack.append(sorted_points[0])
    stack.append(sorted_points[1])

    for i in range(2, m):
        length = len(stack)
        top = stack[length-1]
        next_top = stack[length-2]
        v1 = [sorted_points[i][0]-next_top[0], sorted_points[i][1]-next_top[1]]
        v2 = [top[0]-next_top[0], top[1]-next_top[1]]
        v0 = [0, 0]

        while Cross(v1, v2, v0) >= 0:
            stack.pop()
            length = len(stack)
            top = stack[length-1]
            next_top = stack[length-2]
            v1 = [sorted_points[i][0] - next_top[0], sorted_points[i][1] - next_top[1]]
            v2 = [top[0] - next_top[0], top[1] - next_top[1]]

        stack.append(sorted_points[i])

    return stack


def Check(CH, n, p_che):
    # get vertex from Graham_scan, if checking point is in the hull, 
     #its multicross with vertex by sort must be positive
    for i in range(n-1):
        if (Cross(CH[i], CH[i+1], p_che)) < 0:
            return False
    if (Cross(CH[n-1], CH[0], p_che)) > 0:
        return True
    else:
        return False

def get_area_points(area_file):
    P = []
    # s = 'model2point area.mod area.point'
    # os.system(s)
    with open(area_file,'r') as f:
        line = f.read()
        line = line.split()
        point = np.reshape(line,(-1, 3))
        p = np.delete(point, 2, axis=1) # delete z value
        tmp = p.tolist()
        for poi in tmp:
            poi = list(map(int, poi))
            P.append(poi)
    return P


def compute_M(data):
    cols = np.arange(data.size)
    return csr_matrix((cols, (data.ravel(), cols)),
                      shape=(data.max()+1, data.size))

def get_indices_sparse(data):
    M = compute_M(data)
    return [np.unravel_index(row.data, data.shape) for row in M]


if __name__ == "__main__":
    import argparse
    import json
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--tomo', type=str, default=None, help='tomo file')
    parser.add_argument('--mask_file', type=str, default=None, help='the output vesicle mask file name')
    parser.add_argument('--render', type=str, default=None, help='if draw fitted vesicles on a new tomo')
    parser.add_argument('--render_in', type=str, default=None, help='if draw fitted vesicles which in presyn on a new tomo')
    parser.add_argument('--min_radius', type=int, default=10, help='minimal radius of targeting vesicles')
    parser.add_argument('--area_file', type=str, default='area.point', help='.point file which defines interested area')
    parser.add_argument('--output_file', type=str, default=None, help='output vesicles file name (xxx.json)')
    parser.add_argument('--output_file_in_area', type=str, default=None, help='output vesicles in presyn file name (xxx.json)')
    parser.add_argument('--sup',  type=str, default=True, help='whether need supplement info by ellipse fit(True or False)')
    args = parser.parse_args()

    # set some default files 
    if args.mask_file is None:
        args.mask_file = args.tomo + '-bin8-wbp_corrected-mask.mrc'
    if args.render is None:
        args.render = args.tomo + '-vesicles.mrc'
    if args.render_in is None:
        args.render_in = args.tomo + '-vesicles-in.mrc'
    if args.output_file is None:
        args.output_file = args.tomo + '_vesicle.json'
    if args.output_file_in_area is None:
        args.output_file_in_area = args.tomo + '_vesicle_in.json'

    min_radius = args.min_radius
    # save raw vesicle mask
    with mrcfile.open(args.mask_file) as m:
        bimask =  m.data
    shape = bimask.shape
    print('begin morph process')
    vesicle_list, vesicle_list_sup, shape = morph_process(args.mask_file,radius=min_radius)
    print('done morph process')
    [vesicle_info, in_vesicle_info] = vesicle_measure(vesicle_list, vesicle_list_sup, shape, min_radius, args.output_file, args.output_file_in_area, area_file=args.area_file)
    print('done vesicle measuring')

    if args.render is not None:
        ves_tomo = vesicle_rendering(args.output_file,shape)
        with mrcfile.new(args.render,overwrite=True) as n:
            n.set_data(ves_tomo)
    
    if args.render_in is not None:
        ves_in_tomo = vesicle_rendering(args.output_file_in_area, shape)
        with mrcfile.new(args.render_in, overwrite=True) as n_in:
            n_in.set_data(ves_in_tomo)
