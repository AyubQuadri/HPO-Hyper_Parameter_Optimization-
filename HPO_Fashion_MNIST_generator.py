import numpy as np
import keras
from random import shuffle
import cv2
from sklearn.utils import shuffle
import os 
from matplotlib import pyplot as plt
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, BatchNormalization, MaxPooling2D, Flatten, Activation
import talos as ta

batch_size = 128
num_classes = 10
# input image dimensions
img_rows, img_cols = 28, 28

def load_data_Fashan_Mnist():
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
    y = np.concatenate((y_train, y_test), axis=0)

    return x, y

def batch_iter(data, labels, batch_size, shuffle=True):
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1

    def data_generator():
        data_size = len(data)
        while True:
            # Shuffle the data at each epoch
            if shuffle:
                shuffle_indices = np.random.permutation(np.arange(data_size))
                shuffled_data = data[shuffle_indices]
                shuffled_labels = labels[shuffle_indices]
            else:
                shuffled_data = data
                shuffled_labels = labels

            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * batch_size
                end_index = min((batch_num + 1) * batch_size, data_size)
                X, y = shuffled_data[start_index: end_index], shuffled_labels[start_index: end_index]
                yield X, y

    return num_batches_per_epoch, data_generator()

def fashion_mnist_fn(x_train, y_train, x_val, y_val, params):
    fit = False # if want to run normal fit funciton from talos.
    gen = True  # if want to run fit_generator function from talos. 
    print('Parameters\n',params)
    
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
    model.add(Dropout(conv_dropout))
    model.add(Dense(10))
    model.add(Activation('softmax'))
  
    model.compile(
        optimizer=params['optimizer'], 
        loss=params['losses'],
        metrics=['accuracy']
    )

    
    # To-Do 
    # call data_generator get x_batch, y_batch yeild 
    # call model.fit 
    if fit == True:
        
        out = model.fit(x_train, y_train, 
                        epochs=params['epochs'], 
                        batch_size=params['batch_size'], 
                        verbose=0,
                        validation_data=[x_val, y_val])
        
    elif gen == True:
        train_steps, train_batches = batch_iter(x_train, y_train, params['batch_size'])
        valid_steps, valid_batches = batch_iter(x_val, y_val, params['batch_size'])

        out = model.fit_generator(train_batches, train_steps, 
                                epochs=params['epochs'],
                                validation_data=valid_batches, 
                                validation_steps=valid_steps)
        
        
    return out, model

p = {'lr': [0.0001, 0.001, 0.005],
     'batch_size': [32],
     'epochs': [10],
     'conv_dropout': [0.25, 0.4, 0.5],
     'optimizer': ['adam', 'sgd'],
     'losses': ['categorical_crossentropy', 'logcosh'],
     'activation': ['relu', 'elu'],
     'last_activation': ['softmax'],
     'dense1_neuron': [512,1024]
     }


if __name__ == "__main__":
    x, y = load_data_Fashan_Mnist()
    h = ta.Scan(x, y,
            params=p,
            dataset_name='first_test',
            experiment_no='1',
            model=fashion_mnist_fn)

    
