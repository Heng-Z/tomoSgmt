#! usr/env/bin python3

import mrcfile
import numpy as np

with mrcfile.open("pp0294-bin8-wbp_corrected.mrc") as s:  # for cryo-ET data
    image = s.data # axis: z, y, x
    image.flags.writeable = True
    image = image.transpose(2, 1, 0)
mean = image.mean() # use mean value instead of random value
image_ran_era = np.zeros(image.shape)


def random_erase_np(img, M):
    height = img.shape[0]
    width = img.shape[1]
    
    for attempt in range(30):
        target_area = 15*15 #15*15 in bin8
        aspect_ratio = 1
        h = int(np.round(np.sqrt(target_area * aspect_ratio)))
        w = int(np.round(np.sqrt(target_area / aspect_ratio)))
        if w < width and h < height:
            x1 = np.random.randint(0, height - h)
            y1 = np.random.randint(0, width - w)
            img[x1:x1+h, y1:y1+w] = M
    return img


#sess = tf.Session()
#image_var = tf.Variable(image, validate_shape=False)
#init_op = tf.variables_initializer(var_list=[image_var])
#sess.run(init_op)



for i in range(image.shape[2]):
    image_ran_era[:, :, i] = random_erase_np(image[:, :, i], M = mean)

image_ran_era = image_ran_era.transpose(2, 1, 0)

with mrcfile.new("pp0294-bin8-wbp_corrected_ran_era.mrc", overwrite=True) as m:
    m.set_data(image_ran_era.astype(np.float32))
    
# img1 = tf.py_func(random_erase_np, [image], tf.uint8)
# imsave("random_erasing_np.png", sess.run(img1))
print("already erasing")
    
    
    
