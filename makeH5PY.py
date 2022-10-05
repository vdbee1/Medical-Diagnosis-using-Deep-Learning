import h5py
import numpy as np
import os
import glob
from keras.preprocessing.image import img_to_array, load_img
from random import shuffle
import  cv2



path1 = "Z:/majorProject/input/train_roi_0.15/"
path2 = "Z:/majorProject/input/test_roi_0.15/"
hdf5_path = 'Z:/majorProject/datasetFinal(with Aug).hdf5'
shuffle_data = True
train_addrs = []
test_addrs = []

addrs1 = glob.glob(path1+"Type_1/"+"*.png")
l1 = [0 for addr in addrs1]
addrs2 = glob.glob(path1+"Type_2/"+"*.png")
l2 = [1 for addr in addrs2]
addrs3 = glob.glob(path1+"Type_3/"+"*.png")
l3 = [2 for addr in addrs3]
train_addrs = addrs1 + addrs2 + addrs3
train_labels = l1 + l2 + l3

if shuffle_data:
    c = list(zip(train_addrs, train_labels))
    shuffle(c)
    train_addrs, train_labels = zip(*c)

addrs1 = glob.glob(path2+"Type_1/"+"*.png")
l1 = [0 for addr in addrs1]
addrs2 = glob.glob(path2+"Type_2/"+"*.png")
l2 = [1 for addr in addrs2]
addrs3 = glob.glob(path2+"Type_3/"+"*.png")
l3 = [2 for addr in addrs3]
test_addrs = addrs1 + addrs2 + addrs3
test_labels = l1 + l2 + l3

if shuffle_data:
    c = list(zip(test_addrs, test_labels))
    shuffle(c)
    test_addrs, test_labels = zip(*c)

data_order = 'tf'  # 'th' for Theano, 'tf' for Tensorflow
# check the order of data and chose proper data shape to save images
if data_order == 'th':
    train_shape = (len(train_addrs), 3, 224, 224)
    test_shape = (len(test_addrs), 3, 224, 224)
elif data_order == 'tf':
    train_shape = (len(train_addrs), 32, 32, 3)
    test_shape = (len(test_addrs), 32, 32, 3)
# open a hdf5 file and create earrays
hdf5_file = h5py.File(hdf5_path, mode='w')
hdf5_file.create_dataset("train_img", train_shape, np.float32)
hdf5_file.create_dataset("test_img", test_shape, np.float32)
hdf5_file.create_dataset("train_labels", (len(train_addrs),), np.int8)
hdf5_file["train_labels"][...] = train_labels
hdf5_file.create_dataset("test_labels", (len(test_addrs),), np.int8)
hdf5_file["test_labels"][...] = test_labels

for i in range(len(train_addrs)):
    # print how many images are saved every 1000 images
    if i % 10 == 0 and i > 1:
        print('Train data: {}/{}'.format(i, len(train_addrs)))
    # read an image and resize to (224, 224)
    # cv2 load images as BGR, convert it to RGB
    addr = train_addrs[i]
    img = cv2.imread(addr).astype('float32')/255
    img = cv2.resize(img, (32, 32), interpolation=cv2.INTER_CUBIC)
    # add any image pre-processing here
    # if the data order is Theano, axis orders should change)
    # save the image and calculate the mean so far
    hdf5_file["train_img"][i, ...] = img[None]
# loop over test addresses
for i in range(len(test_addrs)):
    # print how many images are saved every 1000 images
    if i % 10 == 0 and i > 1:
        print('Test data: {}/{}'.format(i, len(test_addrs)))
    addr = test_addrs[i]
    img = cv2.imread(addr).astype('float32')/255
    img = cv2.resize(img, (32, 32), interpolation=cv2.INTER_CUBIC)
    # save the image
    hdf5_file["test_img"][i, ...] = img[None]
# save the mean and close the hdf5 file
hdf5_file.close()


#hdf5_file = h5py.File(hdf5_path, mode='r')



