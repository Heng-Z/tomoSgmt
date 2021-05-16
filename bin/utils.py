'''

'''
import mrcfile
import numpy as np
from skimage.morphology import opening,closing, disk
from tensorflow.keras.utils import Sequence


class DataWrapper(Sequence):

   def __init__(self, X,  batch_size):
       self.X = X
       self.batch_size = batch_size
   def __len__(self):
       return int(np.ceil(len(self.X) / float(self.batch_size)))
   def __getitem__(self, i):
       idx = slice(i*self.batch_size,(i+1)*self.batch_size)
       return self.X[idx]

def normalize(x, percentile = True, pmin=4.0, pmax=96.0, axis=None, clip=False, eps=1e-20):
    """Percentile-based image normalization."""

    if percentile:
        mi = np.percentile(x,pmin,axis=axis,keepdims=True)
        ma = np.percentile(x,pmax,axis=axis,keepdims=True)
        out = (x - mi) / ( ma - mi + eps )
        out = out.astype(np.float32)
        if clip:
            return np.clip(out,0,1)
        else:
            return out
    else:
        out = (x-np.mean(x))/np.std(x)
        out = out.astype(np.float32)
        return out

def gene_2d_training_data(tomo,mask,sample_mask=None,num=100,sidelen=128,neighbor_in=5, neighbor_out=1):
    with mrcfile.open(tomo) as o:
        orig_tomo=o.data 
    tomo = normalize(orig_tomo,percentile = False)
    with mrcfile.open(mask) as m:
        mask=m.data

    if  sample_mask != None:
        with mrcfile.open(sample_mask) as sm:
            sample_mask = sm.data
    else:
        sample_mask = np.ones(tomo.shape)
    sp=tomo.shape
    if sample_mask is None:
        sample_mask=np.ones(sp)
    else:
        sample_mask=sample_mask
    # if os.path.isdir('./data'):
    #     os.system('mv {} {}'.format('./data', './data'+'~'))
    # os.makedirs('./data')
    # dirs_tomake = ['train_x','train_y', 'test_x', 'test_y']
    # for d in dirs_tomake:
    #     os.makedirs('{}/{}'.format('./data', d))
    border_slices = tuple([slice(s // 2, d - s + s // 2 + 1) for s, d in zip((neighbor_in,sidelen,sidelen), sp)])
    valid_inds = np.where(sample_mask[border_slices])
    valid_inds = [v + s.start for s, v in zip(border_slices, valid_inds)]
    sample_inds1 = np.random.choice(len(valid_inds[0]), num, replace=len(valid_inds[0]) < num)
    sample_inds2 = np.random.choice(len(valid_inds[0]), int(num*0.1), replace=len(valid_inds[0]) < int(num*0.1))
    rand_inds1 = [v[sample_inds1] for v in valid_inds]
    rand_inds2 = [v[sample_inds2] for v in valid_inds]
    seeds1 = (rand_inds1[0],rand_inds1[1], rand_inds1[2])
    seeds2 = (rand_inds2[0],rand_inds2[1], rand_inds2[2])
    

    X_train = np.swapaxes(crop_patches(tomo,seeds1,sidelen=sidelen,neighbor=neighbor_in),1,-1)
    Y_train = np.swapaxes(crop_patches(mask,seeds1,sidelen=sidelen,neighbor=neighbor_out),1,-1)
    X_test = np.swapaxes(crop_patches(tomo,seeds2,sidelen=sidelen,neighbor=neighbor_in),1,-1)
    Y_test = np.swapaxes(crop_patches(mask,seeds2,sidelen=sidelen,neighbor=neighbor_out),1,-1)

    print(X_train.shape)
    # for j,s in enumerate(X_train):
    #     with mrcfile.new('{}/train_x/{}_{:0>6d}.mrc'.format('./data', 'subtomo_x',j), overwrite=True) as output_mrc:
    #         output_mrc.set_data(s.astype(np.float32))
    
    # for j,s in enumerate(Y_train):
    #     with mrcfile.new('{}/train_y/{}_{:0>6d}.mrc'.format('./data', 'subtomo_y',j), overwrite=True) as output_mrc:
    #         output_mrc.set_data(s) 

    # for j,s in enumerate(X_test):
    #     with mrcfile.new('{}/test_x/{}_{:0>6d}.mrc'.format('./data', 'test_x',j), overwrite=True) as output_mrc:
    #         output_mrc.set_data(s.astype(np.float32)) 

    # for j,s in enumerate(Y_test):
    #     with mrcfile.new('{}/test_y/{}_{:0>6d}.mrc'.format('./data', 'test_y',j), overwrite=True) as output_mrc:
    #         output_mrc.set_data(s) 
    return (X_train,Y_train), (X_test,Y_test)
def crop_patches(img3D,seeds,sidelen=128,neighbor=1):
    size=len(seeds[0])
    disk_size=(neighbor,sidelen,sidelen)
    cubes=[img3D[tuple(slice(_r-(_p//2),_r+_p-(_p//2)) for _r,_p in zip(r,disk_size))] for r in zip(*seeds)]
    cubes=np.array(cubes)
    return cubes


class Patch:
    def __init__(self, tomo):
        self.sp = tomo.shape
        self.tomo = tomo
    
    def to_patches(self,sidelen=128,overlap_rate = 0.25,neighbor=5):
        effect_len = int(sidelen * (1-overlap_rate))
        n1 = (self.sp[1] - sidelen)//effect_len + 2
        n2 = (self.sp[2] - sidelen)//effect_len + 2
        n0 = self.sp[0]
        pad_len1 = (n1-1)*effect_len + sidelen - self.sp[1]
        pad_len2 = (n2-1)*effect_len + sidelen - self.sp[2]
        tomo_padded = np.pad(self.tomo,((neighbor//2, neighbor-neighbor//2),
                                        (pad_len1//2,pad_len1 - pad_len1//2),
                                        (pad_len2//2,pad_len2 - pad_len2//2)),'symmetric')
        patch_list = []
        print('padded shape',tomo_padded.shape)
        for k in range(neighbor//2,neighbor//2+self.sp[0]):
            for i in range(n1):
                for j in range(n2):
                    one_patch = tomo_padded[k-neighbor//2:k+neighbor-neighbor//2,
                                            i*effect_len:i * effect_len + sidelen,
                                            j*effect_len:j * effect_len + sidelen]
                    
                    patch_list.append(one_patch)
        print('one patch shape',one_patch.shape)
        self.n12 = (n1,n2)
        self.sidelen = sidelen
        self.effect_len = effect_len
        self.neighbor = neighbor
        self.padded_dim = tomo_padded.shape

        return patch_list

    def restore_tomo(self,patch_list,neighbor=1):
        (n1,n2) = self.n12
        sidelen = self.sidelen
        effect_len = self.effect_len
        sp = self.sp
        # neighbor = self.neighbor
        # tomo_padded = np.zeros(self.padded_dim)
        tomo_padded = np.zeros((sp[0] + neighbor-neighbor%2,self.padded_dim[1],self.padded_dim[2]))
        for k in range(neighbor//2,neighbor//2+self.sp[0]):
            for i in range(n1):
                for j in range(n2):
                    one_patch = tomo_padded[ k-neighbor//2:k-neighbor//2+neighbor,
                                 i*effect_len:i * effect_len + sidelen,
                                j*effect_len:j * effect_len + sidelen]
                    # print('brop one_patch',one_patch.shape)
                    tomo_padded[ k-neighbor//2:k-neighbor//2+neighbor,
                                 i*effect_len:i * effect_len + sidelen,
                                j*effect_len:j * effect_len + sidelen] += patch_list[(k-neighbor//2)*n1*n2 + i*n2 + j]
            # print('k and index:',k,k*n1*n2 + i*n2 + j)
        # tomo_padded = (tomo_padded>0).astype(np.uint8)
        pad_len1 = (n1-1)*effect_len + sidelen - self.sp[1]
        pad_len2 = (n2-1)*effect_len + sidelen - self.sp[2]
        restored_tomo = tomo_padded[neighbor//2 : neighbor//2+self.sp[0],
                                pad_len1//2  : pad_len1//2 + self.sp[1],
                                pad_len2//2  : pad_len2//2 + self.sp[2],
                                ]

        return restored_tomo

def bottom_hat(tomo,disk_size=40,factor=1.0):
    sp = tomo.shape
    # transformed = np.zeros([1,sp[1],sp[2]])
    # sli = tomo[108,:,:]
    # transformed[0,:,:]  = sli - factor * closing(sli,disk(disk_size))
    transformed = np.zeros(sp)
    for i,sli in enumerate(tomo):
        transformed[i,:,:] = sli - factor * closing(sli,disk(disk_size))
        print(i)
    transformed[0,:,:] = closing(sli,disk(disk_size)) - sli

    return transformed.astype(type(tomo[0,0,0]))

def bottom_hat_2d(sli,disk_size=40,factor=1.0):
    transformed = sli - factor * closing(sli,disk(disk_size))
    return transformed

def bottom_hat_parallel(tomo,disk_size=40,factor=1.0,ncpu=8):
    from multiprocessing import Pool
    from functools import partial
    sp = tomo.shape
    slice_list  = [ tomo[i,:,:] for i in range(sp[0])]
    func = partial(bottom_hat_2d,disk_size=40,factor=1.0)
    with Pool(ncpu) as p:
        transformed_list  = list(p.map(func,slice_list))
    transformed = np.array(transformed_list)
    return transformed.astype(type(tomo[0,0,0]))
        