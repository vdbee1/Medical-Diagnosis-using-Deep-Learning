import os
import h5py
import numpy as np
from keras import backend as K
import tensorflow as tf
from keras.layers import Layer
from keras import activations, optimizers
from keras.models import Model
from keras.layers import *
from keras import layers
from keras.models import Sequential
from keras import models
import cv2
from keras.utils import to_categorical
from keras.utils.vis_utils import plot_model
from keras import initializers
from sklearn.metrics import confusion_matrix, classification_report

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

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

class CapsuleLayer(layers.Layer):
    def __init__(self, num_capsule, dim_capsule, routings=3,
                 kernel_initializer='glorot_uniform',
                 **kwargs):
        super(CapsuleLayer, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.kernel_initializer = initializers.get(kernel_initializer)

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
            c = tf.nn.softmax(b, dim=1)
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
def softmax(x, axis=-1):
    ex = K.exp(x - K.max(x, axis=axis, keepdims=True))
    return ex / K.sum(ex, axis=axis, keepdims=True)

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
train_model.compile(optimizer=optimizers.Adam(lr=0.001), loss=[margin_loss, 'mse'], #loss_weights=[1., 0.392]
                    metrics={'capsnet': 'accuracy'})
#plot_model(train_model, to_file='capsNetArchitecture.png', show_shapes=True, show_layer_names=True)




eval_model.load_weights("C:/Users/roger/Desktop/temp/finalWEIGHTS(without Aug) - 24.03.2020 - 23-42-57.h5")
typ1Path = "Z:/majorProject/input/test_roi_0.15/Type_1/"
typ2Path = "Z:/majorProject/input/test_roi_0.15/Type_2/"
typ3Path = "Z:/majorProject/input/test_roi_0.15/Type_3/"


t = []



def pr(path):
    for i in os.listdir(path):
        #print(i)
        img = cv2.imread("Z:/majorProject/input/test_roi_0.15/Type_1/"+i).astype('float32')/255
        img = cv2.resize(img, (32, 32), interpolation=cv2.INTER_CUBIC)
        img = img.reshape(-1,32,32,3)
        img = img.astype('int8')
        #print(img)
        #print(img)
        yPred,x = eval_model.predict(img)
        #print(yPred.shape,"     123 ")
        print(yPred[0])
        #if path == typ1Path:
            #yPred[0][0] += 0.25
        pred = np.argmax(yPred,1)
        t.append(pred)
        if pred == 0:
            print("It belongs to type 1")
        elif pred == 1:
            print("It belongs to type 2")
        elif pred == 2:
            print("It belongs to type 3")
        else:
            print("Error")


#pr(typ1Path)


h5py_path = 'Z:/majorProject/datasetFinal(without Aug).hdf5'
print("123")
dataset = h5py.File(h5py_path,mode='r')
xTrain = dataset['train_img']
yTrain = dataset['train_labels']
xTest = dataset['test_img']
yTest = dataset['test_labels']
yTrain = np.array(to_categorical(yTrain))
yTest = np.array(to_categorical(yTest))
#temp = xTest[0]
#temp = temp.reshape(1,32,32,3)
#print(temp.shape)
yPred,x = eval_model.predict(xTest)
#print(yPred[:20])
pred = np.argmax(yPred,1)
truE = np.argmax(yTest,1)
print('12343')
#print(pred.shape)


#print("yPred =",yPred)
#print(yTrain[:3])
#print(yPred.shape,"               ",x.shape)
#print(np.argmax(yPred[:100],1))
print('Test acc:', np.sum(np.argmax(yPred, 1) == np.argmax(yTest, 1)) / yTest.shape[0])
#{'CapsuleLayer': CapsuleLayer, 'Mask': Mask, 'Length': Length, 'margin_loss': margin_loss}

#print(confusion_matrix(truE, pred))


print(classification_report(truE, pred))


'''
def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()

plot_confusion_matrix(cm = confusion_matrix(truE, pred),
                      normalize    = True,
                      target_names = ['1', '2', '3'],
                      title        = "Confusion Matrix")

'''