import mrcfile
import sys
import numpy as np
import sys
from scipy.spatial.transform import Rotation as R
import json
from skimage.morphology import opening, closing, erosion, cube
def ellipsoid_point(radii,trans,rot_matrix,eps=0.03):
    y,z,x = np.meshgrid(np.arange(-50,50),np.arange(-50,50),np.arange(-50,50))
    eps = eps
    # ellips = np.logical_and((z/radii[0])**2 + (y/radii[1])**2 + (x/radii[2])**2 < 1+eps,
    # (z/radii[0])**2 + (y/radii[1])**2 + (x/radii[2])**2 > 1-eps).astype(np.int8)
    ellips = (z/radii[0])**2 + (y/radii[1])**2 + (x/radii[2])**2 < 1+eps
    ellips =  ellips.astype(np.uint8)
    cloud = np.array(np.where(ellips==1)).T - np.array([50,50,50])
    # r1 = R.from_matrix(rot_matrix)
    # cloud_r = r1.apply(cloud)
    cloud_r = np.dot(cloud,rot_matrix)
    cloud_trans = cloud_r + trans
    out = np.round(cloud_trans)
    return out.astype(np.int16)

def ifnormal(radii,threshold=0.4):
    if np.std(radii)/np.mean(radii) >threshold:
        a = False
    else: 
        a = True
    return a

def draw_vesicle(tomo,cloud):
    for points in range(cloud.shape[0]):
        idx = cloud[points,:]
        tomo[idx[0],idx[1],idx[2]] = 1
    # tomo = closing(tomo,cube(2))
    return tomo

if __name__ == '__main__':    
    tomo = np.zeros([100,200,200])
    from mwr_vesicle import morph_process,vesicle_measure,vesicle2json
    outfile = '/storage/heng/tomoSgmt/utils/pp676_vescles_hold_filled.mrc'
    save_labeled = '/storage/heng/tomoSgmt/utils/ellipsoid_10_13_14_cl2_labeled.mrc'
    infile = '/storage/heng/tomoSgmt/utils/ellipsoid_10_13_14_45D.mrc'
    # vlist = morph_process(infile,elem_len = 1,save_labeled = save_labeled)
    # ves = vesicle_measure(vlist,None)

    with open('/storage/heng/tomoSgmt/example/test_2021Jan/pp676_ves.json') as f:
        ves = json.load(f)
    vesicle_info = ves['vesicles']
    vesicle_tomo = np.zeros([214,928,928])
    # vesicle_tomo = tomo
    for i in range(len(vesicle_info)): 
        info_i = vesicle_info[i]
        print(info_i)
        ellip_i = ellipsoid_point(np.array(info_i['radii']),np.array(info_i['center']),np.array(info_i['evecs']))
        if ifnormal(info_i['radii'],threshold=0.3):
            print('draw_one')
            vesicle_tomo = draw_vesicle(vesicle_tomo,ellip_i)
        else:
            pass
    
    # ellip1 = ellipsoid_point([10,13,14],[45,47,48],np.array([[1,0,0],[0,0.7071,-0.7071],[0,0.7071,0.7071]]))
    # # ellip1 = ellipsoid_point([16,13,10],[45,47,48],np.array([[1,0,0],[0,1,0],[0,0,1]]))
    # vesicle_tomo = draw_vesicle(tomo,ellip1)
    vesicle_tomo = closing(vesicle_tomo,cube(3))
    with mrcfile.new(outfile,overwrite =True) as n:
        n.set_data(vesicle_tomo.astype(np.uint8))
    # ellip1 = ellipsoid_point([10,13,14],[45,47,48],np.array([[1,0,0],[0,0.7071,-0.7071],[0,0.7071,0.7071]]))
    # print(ellip1)
    # for points in range(ellip1.shape[0]):
    #     idx = ellip1[points,:]
    #     tomo[idx[0],idx[1],idx[2]] = 1
    # with mrcfile.new('ellipsoid_10_13_14.mrc',overwrite =True) as n:
    #     n.set_data(tomo.astype(np.uint8))


        
    # with mrcfile.new('ellisoid_15_13_12_000.mrc',overwrite =True) as n:
    #     n.set_data(ellips)
    # cloud = np.array(np.where(ellips==1))
    # cloud = np.swapaxes(cloud,0,1)
    # print(cloud.shape)
    # sys.path.append('/storage/heng/tomoSgmt/bin/ellipsoid_fit_python/')
    # import ellipsoid_fit as ef 
    # [center, evecs, radii]=ef.ellipsoid_fit(cloud)
    # info={'name':'vesicle_demo','center':center,'radii':radii,'evecs':evecs}
    # print(info)

    # cloud0 = cloud - np.mean(cloud,axis=0)
    # print(np.mean(cloud0,axis=0))
    # C = np.dot(cloud.T ,cloud)
    # eigval,eigvec = np.linalg.eig(C)

    # print('eigval',eigval)
    # print('eigvec',eigvec)


