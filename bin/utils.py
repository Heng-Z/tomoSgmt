'''

'''
from mwr.preprocessing.cubes import create_cube_seeds, crop_cubes
def gene_train_data(settings):
    with mrcfile.open(settings.orig_tomo) as o:
        orig_tomo=o.data 
    
    with mrcfile.open(settings.mask_tomo) as m:
        mask_tomo=m.data
    #create random center seeds and crop subtomos
    #10% ncube will be saved as test_set
    seeds1=create_cube_seeds(orig_tomo,settings.ncube,settings.cropsize)
    seeds2=create_cube_seeds(orig_tomo,int(settings.ncube*0.1),settings.cropsize)

    orig_subtomos=crop_cubes(orig_tomo,seeds1,settings.cropsize)
    mask_subtomos=crop_cubes(mask_tomo,seeds1,settings.cropsize)

    orig_test_subtomos=crop_cubes(orig_tomo,seeds2,settings.cropsize)
    mask_test_subtomos=crop_cubes(mask_tomo,seeds2,settings.cropsize)

    for j,s in enumerate(orig_subtomos):
        with mrcfile.new('{}/train_x/{}_{:0>6d}.mrc'.format(settings.data_folder, 'subtomo_x',j), overwrite=True) as output_mrc:
            output_mrc.set_data(s.astype(np.float32))
    
    for j,s in enumerate(mask_subtomos):
        with mrcfile.new('{}/train_y/{}_{:0>6d}.mrc'.format(settings.data_folder, 'subtomo_y',j), overwrite=True) as output_mrc:
            output_mrc.set_data(s) 

    for j,s in enumerate(orig_test_subtomos):
        with mrcfile.new('{}/test_x/{}_{:0>6d}.mrc'.format(settings.data_folder, 'test_x',j), overwrite=True) as output_mrc:
            output_mrc.set_data(s.astype(np.float32)) 

    for j,s in enumerate(mask_test_subtomos):
        with mrcfile.new('{}/test_y/{}_{:0>6d}.mrc'.format(settings.data_folder, 'test_y',j), overwrite=True) as output_mrc:
            output_mrc.set_data(s) 