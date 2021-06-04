import os
import mrcfile
import numpy as np
import sys
from scipy.sparse import csr_matrix
import json
from tomoSgmt.bin.sgmt_predict import predict_new
import math

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

def vesicle_measure(vesicle_list,min_radius,outfile, outfile_in_area,area_file=None):
    from tomoSgmt.bin.ellipsoid import ellipsoid_fit as ef
    results = []
    results_in = []
    # results_out = []
    P = get_area_points(area_file)
    CH = Graham_scan(P)
    global in_count
    in_count = 0
 
    def if_normal(radii,threshold=0.20):
        if np.std(radii)/np.mean(radii) >threshold:
            a = False
        elif np.mean(radii) < min_radius or np.mean(radii) > min_radius*4:
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
            #####################################
            #check whether a vesicle in given presyn
            c = np.delete(center, 0)
            c[0], c[1] = c[1], c[0]
            if Check(CH, len(CH), c):
                results_in.append(info)
                print('in vesicle {}'.format(i))
                in_count = in_count+1
            # else:
            #     results_out.append(info)
            #####################################
        else:
            print('bad vesicle {}'.format(i))
    #return vesicle information dict and save as json
    vesicle_info={'vesicles':results}
    #
    in_vesicle_info={'vesicles':results_in}

    if outfile is not None:
        with open(outfile,"w") as out:
            json.dump(vesicle_info,out)
    
    if outfile_in_area is not None:
        with open(outfile_in_area,"w") as out_in:
            json.dump(in_vesicle_info, out_in)

    return [vesicle_info, in_vesicle_info]

def compute_M(data):
    cols = np.arange(data.size)
    return csr_matrix((cols, (data.ravel(), cols)),
                      shape=(data.max()+1, data.size))

def get_indices_sparse(data):
    M = compute_M(data)
    return [np.unravel_index(row.data, data.shape) for row in M]

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

#############################################
def get_bottom_point(points):
    #get the first point for Graham_scan
    #cuz the farthest point must be one of the vertex of convex hull
    min_index = 0
    n = len(points)
    for i in range(0, n):
        if points[i][1] < points[min_index][1] or (points[i][1] == points[min_index][1] and points[i][0] < points[min_index][0]):
            min_index = i
    return min_index
 
 
def sort_polar_angle_cos(points, center_point):
    #sort by polar angle with center point(with cos value)
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
    #output a vertex set by anticlockwise of convex hull

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
    #get vertex from Graham_scan, if checking point is in 
     #the hull, its multicross with vertex by sort must be positive
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
        p = np.delete(point, 2, axis=1) #delete z value
        tmp = p.tolist()
        for poi in tmp:
            poi = list(map(int, poi))
            P.append(poi)
            
        # print(P)
    return P