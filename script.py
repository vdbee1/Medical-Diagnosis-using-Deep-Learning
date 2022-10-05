import h5py
import numpy as np
from keras import backend as K
import tensorflow as tf
import keras
from keras.layers import Layer
from keras import activations, optimizers
from keras.models import Model
from keras.layers import *
from keras import layers
from keras.models import Sequential
from keras.models import load_model
from keras import models
from flask import Flask, request, jsonify, render_template,json
import cv2
import pyrebase 

#Tensorflow configs

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
config = tf.ConfigProto(
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
    # device_count = {'GPU': 1}
)

#everything here is defining the model structure and it's functions

class Length(layers.Layer):
    def call(self, inputs, **kwargs):
        return K.sqrt(K.sum(K.square(inputs), -1) + K.epsilon())

    def compute_output_shape(self, input_shape):
        return input_shape[:-1]

    def get_config(self):
        config = super(Length, self).get_config()
        return config

class Mask(layers.Layer):
    def call(self, inputs, **kwargs):
        if type(inputs) is list:  # true label is provided with shape = [None, n_classes], i.e. one-hot code.
            assert len(inputs) == 2
            inputs, mask = inputs
        else:  # if no true label, mask by the max length of capsules. Mainly used for prediction
            # compute lengths of capsules
            x = K.sqrt(K.sum(K.square(inputs), -1))
            mask = K.one_hot(indices=K.argmax(x, 1), num_classes=x.get_shape().as_list()[1])
        masked = K.batch_flatten(inputs * K.expand_dims(mask, -1))
        return masked

    def compute_output_shape(self, input_shape):
        if type(input_shape[0]) is tuple:  # true label provided
            return tuple([None, input_shape[0][1] * input_shape[0][2]])
        else:  # no true label provided
            return tuple([None, input_shape[1] * input_shape[2]])
    def get_config(self):
        config = super(Mask, self).get_config()
        return config

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

class CapsuleLayer(layers.Layer):
    def __init__(self, num_capsule, dim_capsule, routings=3,
                 kernel_initializer='glorot_uniform',
                 **kwargs):
        super(CapsuleLayer, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.kernel_initializer = keras.initializers.get(kernel_initializer)

    def build(self, input_shape):
        assert len(input_shape) >= 3, "The input Tensor should have shape=[None, input_num_capsule, input_dim_capsule]"
        self.input_num_capsule = input_shape[1]
        self.input_dim_capsule = input_shape[2]

        # Transform matrix
        self.W = self.add_weight(shape=[self.num_capsule, self.input_num_capsule,
                                        self.dim_capsule, self.input_dim_capsule],
                                 initializer=self.kernel_initializer,
                                 name='W')

        self.built = True

    def call(self, inputs, training=None):
        inputs_expand = K.expand_dims(inputs, 1)
        # Replicate num_capsule dimension to prepare being multiplied by W
        inputs_tiled = K.tile(inputs_expand, [1, self.num_capsule, 1, 1])
        inputs_hat = K.map_fn(lambda x: K.batch_dot(x, self.W, [2, 3]), elems=inputs_tiled)
        b = tf.zeros(shape=[K.shape(inputs_hat)[0], self.num_capsule, self.input_num_capsule])

        assert self.routings > 0, 'The routings should be > 0.'
        for i in range(self.routings):
            c = tf.nn.softmax(b, dim = 1)
            outputs = squash(K.batch_dot(c, inputs_hat, [2, 2]))  # [None, 10, 16]

            if i < self.routings - 1:
                b += K.batch_dot(outputs, inputs_hat, [2, 3])
        return outputs

    def compute_output_shape(self, input_shape):
        return tuple([None, self.num_capsule, self.dim_capsule])

    def get_config(self):
        config = {
            'num_capsule': self.num_capsule,
            'dim_capsule': self.dim_capsule,
            'routings': self.routings
        }
        base_config = super(CapsuleLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def PrimaryCap(inputs, dim_capsule, n_channels, kernel_size, strides, padding):
    output = layers.Conv2D(filters=dim_capsule * n_channels, kernel_size=kernel_size, strides=strides, padding=padding,
                           name='primarycap_conv2d')(inputs)
    outputs = layers.Reshape(target_shape=[-1, dim_capsule], name='primarycap_reshape')(output)
    return layers.Lambda(squash, name='primarycap_squash')(outputs)


def squash(x, axis=-1):
    s_squared_norm = K.sum(K.square(x), axis, keepdims=True) + K.epsilon()
    scale = K.sqrt(s_squared_norm) / (0.5 + s_squared_norm)
    return scale * x

# define our own softmax function instead of K.softmax
# because K.softmax can not specify axis.
'''
def softmax(x, axis=-1):
    ex = K.exp(x - K.max(x, axis=axis, keepdims=True))
    return ex / K.sum(ex, axis=axis, keepdims=True)
'''
# define the margin loss like hinge loss
def margin_loss(y_true, y_pred):
    lamb, margin = 0.5, 0.1
    return K.sum(y_true * K.square(K.relu(1 - margin - y_pred)) + lamb * (
        1 - y_true) * K.square(K.relu(y_pred - margin)), axis=-1)

def CapsNet(input_shape, n_class, routings):
    input_shape = input_shape
    n_class  = n_class
    routings = routings
    x = layers.Input(shape=input_shape)
    conv1 = layers.Conv2D(filters=256, kernel_size=9, strides=1, padding='same', activation='relu', name='conv1')(x)
    primarycaps = PrimaryCap(conv1, dim_capsule=16, n_channels=32, kernel_size=5, strides=2, padding='same')
    digitcaps = CapsuleLayer(num_capsule=n_class, dim_capsule=16, routings=routings, name='digitcaps')(primarycaps)
    out_caps = Length(name='capsnet')(digitcaps)

    # Decoder network.
    y = layers.Input(shape=(n_class,))
    masked_by_y = Mask()([digitcaps, y]) # The true label is used to mask the output of capsule layer. For training
    masked = Mask()(digitcaps) # Mask using the capsule with maximal length. For prediction

       # Shared Decoder model in training and prediction

    decoder = Sequential(name='decoder')
    decoder.add(layers.Dense(512, activation='relu', input_dim=16*n_class))
    decoder.add(layers.Dense(1024, activation='relu'))
    decoder.add(layers.Dense(np.prod(input_shape), activation='sigmoid'))
    decoder.add(layers.Reshape(target_shape=input_shape, name='out_recon'))

       # Models for training and evaluation (prediction)
    train_model = models.Model([x, y], [out_caps, decoder(masked_by_y)])
    eval_model = models.Model(x, [out_caps, decoder(masked)])

       # manipulate model
    noise = layers.Input(shape=(n_class, 16))
    noised_digitcaps = layers.Add()([digitcaps, noise])
    masked_noised_y = Mask()([noised_digitcaps, y])
    manipulate_model = models.Model([x, y, noise], decoder(masked_noised_y))
    return train_model, eval_model, manipulate_model

train_model, eval_model, manipulate_model = CapsNet(input_shape= (32, 32, 3), n_class =3 , routings = 3)
# compile the model
#train_model.compile(optimizer=optimizers.Adam(lr=0.001), loss=[margin_loss, 'mse'], #loss_weights=[1., 0.392]
#                   metrics={'capsnet': 'accuracy'})

eval_model.load_weights("C:/Users/roger/Downloads/temp/fweights.h5") #change dir


#eval_model = load_model("E:/test12345.h5",custom_objects={'CapsuleLayer': CapsuleLayer, 'Mask': Mask, 'Length': Length})
'''
img = cv2.imread("F:/additionalgreendata/additional_Type_3_v2/Type_3/330.jpg").astype('float32')/255
img = cv2.resize(img, (32, 32), interpolation=cv2.INTER_CUBIC)
img = img.reshape(1,32,32,3)
yPred,x = eval_model.predict(img)
pred = np.argmax(yPred)
if pred == 0:
    print("It belongs to type 1")
elif pred == 1:
    print("It belongs to type 2")
elif pred == 2:
    print("It belongs to type 3")
else:
    print("Error")
'''


#config for firebase

config = {
  "apiKey": "AIzaSyBQJ_-lipZ0-2JgTOjBoUzUx1QteLUlQuA",
  "authDomain": "major-project-d5c26.firebaseapp.com",
  "databaseURL": "https://major-project-d5c26.firebaseio.com",
  "storageBucket": "major-project-d5c26.appspot.com",
}

#runs the program once for caching(?) else it gives an error for some reason

firebase = pyrebase.initialize_app(config)


db = firebase.database()
Fn = db.child("Filename").get()

for vals in Fn.each():
	print(vals.key())
	print(vals.val())

vv = vals.val()
fname = None
ml = []
for key,value in vv.items():
	ml.append(value)
fname = ml[0]
#print(fname)
storage = firebase.storage()
storage.child("images/"+fname).download(filename= 'downloaded.jpg', path = "C:/Users/roger/Downloads/temp") #change dir
print("images/"+fname)
img = cv2.imread("downloaded.jpg").astype('float32')/255
img = cv2.resize(img, (32, 32), interpolation=cv2.INTER_CUBIC)
img = img.reshape(1,32,32,3)
yPred,x = eval_model.predict(img)
pred = np.argmax(yPred)

#function that predicts the image given once the predict button is pressed(actually works i confirmed)
def pr():
    firebase = pyrebase.initialize_app(config)
    
    
    db = firebase.database()
    Fn = db.child("Filename").get()
    
    for vals in Fn.each():
    	print(vals.key())
    	print(vals.val())
    
    vv = vals.val()
    fname = None
    ml = []
    for key,value in vv.items():
    	ml.append(value)
    fname = ml[0]
    #print(fname)
    storage = firebase.storage()
    storage.child("images/"+fname).download(filename= 'downloaded.jpg', path = "C:/Users/roger/Downloads/temp") #change dir
    print("images/"+fname)
    img = cv2.imread("downloaded.jpg").astype('float32')/255
    img = cv2.resize(img, (32, 32), interpolation=cv2.INTER_CUBIC)
    img = img.reshape(1,32,32,3)
    yPred,x = eval_model.predict(img)
    pred = np.argmax(yPred)
    return pred
 
#here-on it's flask handling the https requests
app = Flask(__name__)
#@app.route('/')
#def home():
#	return render_template('form.html')



@app.route('/', methods = ['GET','POST']) 
def result(): 
    if request.method =='POST':
        pred = pr() #calls the prediction function
        if pred == 0:
            prediction = "It belongs to type 1"
        elif pred == 1:
            prediction = "It belongs to type 2"
        elif pred == 2:
            prediction = "It belongs to type 3"
        else:
            prediction = "Error"
        response = app.response_class(response=json.dumps(prediction),
                                  status=200,
                                  mimetype='application/json')
        return response
        #return render_template("result.html", prediction = prediction) 

if __name__ == "__main__":
    app.run("localhost", "9999",debug=True) #CHANGE THE IP IF YOU NEED TOO

'''
h5py_path = 'Z:/majorProject/dataset_32(OG only(plus900inTrain) scmaz without removal).hdf5'
print("123")
dataset = h5py.File(h5py_path,mode='r')
xTrain = dataset['train_img']
yTrain = dataset['train_labels']
xTest = dataset['test_img']
yTest = dataset['test_labels']
print(yTrain[:3])
yTrain = np.array(to_categorical(yTrain))
yTest = np.array(to_categorical(yTest))
temp = xTest[0]
temp = temp.reshape(1,32,32,3)
print(temp.shape)
yPred,x = eval_model.predict(temp)
pred = np.argmax(yPred)
if pred is 0:
    print("It belongs to type 1")
elif pred is 1:
    print("It belongs to type 2")
else:
    print("It belongs to type 3")
print("yPred =",yPred)
print(yTrain[:3])
print(yPred.shape,"               ",x.shape)
print('Test acc:', np.sum(np.argmax(yPred, 1) == np.argmax(yTest, 1)) / yTest.shape[0])
pred = np.argmax(yPred)
print(pred)
#{'CapsuleLayer': CapsuleLayer, 'Mask': Mask, 'Length': Length, 'margin_loss': margin_loss}
'''
