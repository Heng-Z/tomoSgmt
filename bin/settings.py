orig_tomo = ''
mask_tomo = ''

cropped = False
ncube = 800
cropsize = 80
data_folder = './dataset'

gpuID = "0,1,2,3"
epochs = 30
batch_size = 8
steps_per_epoch = 100

loss='mse'
last_activation='linear'
