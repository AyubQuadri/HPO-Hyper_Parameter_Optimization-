import numpy as np
import keras
from keras import backend as K

batch_size = 128
num_classes = 10
epochs = 12
# input image dimensions
img_rows, img_cols = 28, 28

(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

x = np.concatenate((x_train, x_test), axis=0)
x.shape

y = np.concatenate((y_train, y_test), axis=0)
y.shape

from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, BatchNormalization, MaxPooling2D, Flatten, Activation

def fashion_mnist_fn(x_train, y_train, x_val, y_val, params):
    conv_dropout = float(params['conv_dropout'])
    dense1_neuron = int(params['dense1_neuron'])
    model = Sequential()
    model.add(BatchNormalization(input_shape=x_train.shape[1:]))
    model.add(Conv2D(64, (5, 5), padding='same', activation=params['activation']))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
    model.add(Dropout(conv_dropout))

    model.add(BatchNormalization(input_shape=x_train.shape[1:]))
    model.add(Conv2D(128, (5, 5), padding='same', activation=params['activation']))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(conv_dropout))

    model.add(BatchNormalization(input_shape=x_train.shape[1:]))
    model.add(Conv2D(256, (5, 5), padding='same', activation=params['activation']))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
    model.add(Dropout(conv_dropout))

    model.add(Flatten())
    model.add(Dense(dense1_neuron))
    model.add(Activation(params['activation']))
    model.add(Dropout(0.5))
    model.add(Dense(10))
    model.add(Activation('softmax'))
  
    model.compile(
        optimizer='adam', # tf.train.AdamOptimizer(learning_rate=1e-3, )
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )


    out = model.fit(
        x, y, epochs=10, batch_size=32, 
        verbose=0,
        validation_data=[x_val, y_val]
    )
    return out, model

import talos as ta

para = {
    'dense1_neuron': [256, 512],
    'activation': ['relu', 'elu'],
    'conv_dropout': [0.25, 0.4]
}

import time
start = time.time()
scan_results = ta.Scan(x, y, para, fashion_mnist_fn)
end = time.time()
print(end-start)