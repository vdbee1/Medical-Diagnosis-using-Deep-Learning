from keras import backend as K
import tensorflow as tf
from keras.layers import Layer
from keras import activations, optimizers
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import *
from keras.preprocessing.image import ImageDataGenerator
from keras import layers
from keras.models import Sequential
from keras import models
import h5py
from keras.backend.tensorflow_backend import set_session
from keras.callbacks import CSVLogger as csv
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from sklearn.utils import class_weight
import datetime
from time import localtime, strftime
import matplotlib.pyplot as plt


train_data_directory = "D:/T/test"
test_data_dir = "D:/T/train"
h5py_path = "Z:/majorProject/datasetFinal(without Aug).hdf5"
train_datagen = ImageDataGenerator(rescale=1. / 255)
test_datagen = ImageDataGenerator(rescale=1. / 255)

config = tf.ConfigProto(
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
    # device_count = {'GPU': 1}
)
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
set_session(session)

# the squashing function.
# we use 0.5 in stead of 1 in hinton's paper.
# if 1, the norm of vector will be zoomed out.
# if 0.5, the norm will be zoomed in while original norm is less than 0.5
# and be zoomed out while original norm is greater than 0.5.

class Length(layers.Layer):
    """
    Compute the length of vectors. This is used to compute a Tensor that has the same shape with y_true in margin_loss.
    Using this layer as model's output can directly predict labels by using `y_pred = np.argmax(model.predict(x), 1)`
    inputs: shape=[None, num_vectors, dim_vector]
    output: shape=[None, num_vectors]
    """
    def call(self, inputs, **kwargs):
        return K.sqrt(K.sum(K.square(inputs), -1) + K.epsilon())

    def compute_output_shape(self, input_shape):
        return input_shape[:-1]

    def get_config(self):
        config = super(Length, self).get_config()
        return config

class Mask(layers.Layer):
    """
    Mask a Tensor with shape=[None, num_capsule, dim_vector] either by the capsule with max length or by an additional
    input mask. Except the max-length capsule (or specified capsule), all vectors are masked to zeros. Then flatten the
    masked Tensor.
    For example:
        ```
        x = keras.layers.Input(shape=[8, 3, 2])  # batch_size=8, each sample contains 3 capsules with dim_vector=2
        y = keras.layers.Input(shape=[8, 3])  # True labels. 8 samples, 3 classes, one-hot coding.
        out = Mask()(x)  # out.shape=[8, 6]
        # or
        out2 = Mask()([x, y])  # out2.shape=[8,6]. Masked with true labels y. Of course y can also be manipulated.
        ```
    """

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
    """
    The capsule layer. It is similar to Dense layer. Dense layer has `in_num` inputs, each is a scalar, the output of the
    neuron from the former layer, and it has `out_num` output neurons. CapsuleLayer just expand the output of the neuron
    from scalar to vector. So its input shape = [None, input_num_capsule, input_dim_capsule] and output shape = \
    [None, num_capsule, dim_capsule]. For Dense Layer, input_dim_capsule = dim_capsule = 1.

    :param num_capsule: number of capsules in this layer
    :param dim_capsule: dimension of the output vectors of the capsules in this layer
    :param routings: number of iterations for the routing algorithm
    """

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
    """
    Apply Conv2D `n_channels` times and concatenate all capsules
    :param inputs: 4D tensor, shape=[None, width, height, channels]
    :param dim_capsule: the dim of the output vector of capsule
    :param n_channels: the number of types of capsules
    :return: output tensor, shape=[None, num_capsule, dim_capsule]
    """
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

dataset = h5py.File(h5py_path,mode='r')
xTrain = dataset['train_img']
yTrain = dataset['train_labels']
class_weights = class_weight.compute_class_weight('balanced', np.unique(yTrain), yTrain)
class_weight_dict = dict(enumerate(class_weights))
#print(class_weight_dict)
xTest = dataset['test_img']
yTest = dataset['test_labels']
yTrain = np.array(to_categorical(yTrain))
yTest = np.array(to_categorical(yTest))


train_model, eval_model, manipulate_model = CapsNet(input_shape=(32, 32, 3), n_class=3, routings=3)
# compile the model
train_model.compile(optimizer=optimizers.Adam(lr=0.0001), loss=[margin_loss, 'mse'], loss_weights=[1., 0.392], metrics={'capsnet': 'accuracy'})
# we use a margin loss
train_model.summary()

batchSize = 256
csv_logger = csv("log.csv", append=True, separator='-')
#best_model_file = f'C:/Users/roger/AppData/Local/Programs/Python/Python36/majorProject_Roger/Models/CapsNet/' \
                  #f't1t2t3 - finalModel(weights only, without Aug) - {strftime("%d.%m.%Y - %H-%M-%S",localtime())}.h5'
#best_model = ModelCheckpoint(best_model_file, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True)
#earlyStop = EarlyStopping(patience=10, restore_best_weights=True)
#train_model.load_weights("C:/Users/roger/AppData/Local/Programs/Python/Python36/majorProject_Roger/Models/CapsNet/finalWEIGHTS(without Aug).h5")
x = train_model.fit(x= [xTrain,yTrain], y= [yTrain,xTrain], batch_size=batchSize, epochs=150,validation_data=([xTest,yTest], [yTest, xTest]),
                verbose=1, shuffle='batch', callbacks=[csv_logger], class_weight=class_weight_dict)

#train_model.save_weights(f'C:/Users/roger/AppData/Local/Programs/Python/Python36/majorProject_Roger/Models/CapsNet/t1t2t3) -'
                         #f' {strftime("%d.%m.%Y - %H-%M-%S",localtime())}.h5')
#train_model.save('test123456Correct.h5')







