orig_tomo = ''
mask_tomo = ''
sample_mask = None

neighbor_in = 7
neighbor_out = 3
sidelen=128

cropped = False
ncube = 800
cropsize = 80
data_folder = './dataset'

gpuID = "0,1,2,3"
epochs = 30
batch_size = 8
steps_per_epoch = 100


# loss='mse'
# last_activation='linear'

testtomo = 'pp0312'
model = 'model_nb7-3'

vesicle_file = '/home/zhenhang/sgmt_test/tomoSgmt/example/test_2021Jan/t208/'+testtomo+'/demo/'+testtomo+'-bin8-wbp_corrected-vesicle-in-area.json'
#vesicle_file = '/home/zhenhang/sgmt_test/tomoSgmt/example/test_2021Jan/t208/'+testtomo+'/demo/'+testtomo+'_vesicle_in.json'

ves_xml = '/home/zhenhang/sgmt_test/tomoSgmt/example/test_2021Jan/t208/'+testtomo+'/demo/vesicle_area.xml'
binv = 2
