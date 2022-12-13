import os
from os.path import join as join 
import PIL
from PIL import Image
from torchvision.transforms import *
import cv2
import numpy as np
import imageio
import utils
import skimage
import pdb
data_folder = r"./CT_Covid_19_part_png/test"

image_dir = join(data_folder)
                         
image_filenames=[]  
image_filenames.extend(join(image_dir, x) for x in sorted(os.listdir(image_dir)) if utils.is_imagefile(x))
        
img_num=0
scale_factor=2
kernel_size=3

save_dir = r"./CT_Covid_19_part_png/test_x2_guassian_k"+repr(kernel_size)+"_s0.5"
# save_dir = r"./CT_Covid_19_part_png/test_x2"
if not os.path.exists(save_dir):
	os.makedirs(save_dir)


for img_fn in image_filenames:
	img = Image.open(img_fn)

	hr_img_w,hr_img_h = img.size

	lr_img_w = hr_img_w // scale_factor
	lr_img_h = hr_img_h // scale_factor

	transform = Resize((lr_img_w, lr_img_h), interpolation=Image.BICUBIC)
	img = transform(img)  

	img_array = np.repeat(np.expand_dims(np.asarray(img),-1),3,-1)
	img_array = cv2.GaussianBlur(img_array, (kernel_size,kernel_size), 0.5)

	# img_array = skimage.util.random_noise(img_array, mode='gaussian')
	# img_array = (img_array*255.).astype(np.uint8)

	# pdb.set_trace()
	img_blur = Image.fromarray(img_array[:,:,0])
	# img_blur = img
	# pdb.set_trace()
	save_fn = join(save_dir, os.path.basename(img_fn))
	imageio.imwrite(save_fn, img_blur)
