import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, GlobalAveragePooling2D, Activation, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras import backend as K
from PIL import Image, ImageFile
import numpy as np
import warnings
from PIL import Image, ImageFile
from sklearn.utils import class_weight
from keras.utils.vis_utils import plot_model

ImageFile.LOAD_TRUNCATED_IMAGES = True
warnings.filterwarnings("ignore")
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
rate = 0.01

train_data_directory = "Z:/minorProject/cervical data/train/"
test_data_dir = "Z:/minorProject/cervical data/test/"

'''
train_data_directory = "Z:/majorProject/input/train_roi_0.15"
test_data_dir = "Z:/majorProject/input/test_roi_0.15"
'''
opt = optimizers.Adagrad(lr=0.01)

train_datagen = ImageDataGenerator(rescale=1. / 255)
# shear_range=0.2,
# zoom_range=0.2,
# horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)
nClasses = 3
# img_width, img_height = (None, None)

if K.image_data_format() == 'channels_first':
	input_shape = (3, None, None)
else:
	input_shape = (None, None, 3)


def createModel():
	model = Sequential()
	model.add(Conv2D(32, (7, 7), padding='same', activation='relu', input_shape=input_shape))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.4))

	model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.4))

	model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.4))

	model.add(GlobalAveragePooling2D())
	# model.add(Flatten())
	model.add(Dense(64, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(nClasses, activation='softmax'))
	return model


model1 = createModel()
# batch_size = 32
epochs = 200
model1.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

train_generator = train_datagen.flow_from_directory(train_data_directory, target_size=(256, 256), batch_size=32, class_mode='categorical', shuffle=True)
test_generator = test_datagen.flow_from_directory(test_data_dir, target_size=(256, 256), batch_size=32, class_mode="categorical", shuffle=True)


model1.fit_generator(train_generator,
                     steps_per_epoch=186,
                     epochs=200,
                     validation_data=test_generator,
                     validation_steps=46)
                     #class_weight={0: 2.8701, 1: 1, 2: 1.73079})

model_json = model1.to_json()
with open("model_123.json", "w") as json_file:
	json_file.write(model_json)

model1.save_weights("model123.h5")
print("Model Saved")
