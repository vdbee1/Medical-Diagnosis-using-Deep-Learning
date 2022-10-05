import Augmentor as ag
from PIL import ImageFile
import os

#Path = "Z:/majorProject/input/train_roi_0.15/Type_2"
multiThreaded = True
Paths = ["Z:/majorProject/input/train_roi_0.15/Type_1", "Z:/majorProject/input/train_roi_0.15/Type_2", "Z:/majorProject/input/train_roi_0.15/Type_3",
         "Z:/majorProject/input/test_roi_0.15/Type_1", "Z:/majorProject/input/test_roi_0.15/Type_2", "Z:/majorProject/input/test_roi_0.15/Type_3"]


for path in Paths:
	x, y, files = next(os.walk(path))
	file_count = len(files)
	print(file_count)
	ImageFile.LOAD_TRUNCATED_IMAGES = True
	p = ag.Pipeline(path)
	p.set_seed(6)
	p.rotate_random_90(probability=1)
	p.set_save_format(save_format="PNG")
	p.sample(4*file_count, multi_threaded=multiThreaded)


	'''
	insert at line 21
	p.flip_left_right(probability=0.9)
	p.random_brightness(probability=0.9, min_factor=0.8,max_factor=1.2)
	p.random_contrast(probability=0.9,min_factor=0.4,max_factor=1.5)
	'''

