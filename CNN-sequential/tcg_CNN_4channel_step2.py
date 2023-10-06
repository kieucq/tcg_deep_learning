import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
import pickle
from tensorflow.keras.callbacks import TensorBoard
import pickle
import time
#
# read in data
#
pickle_in = open("tcg_X.pickle","rb")
X = pickle.load(pickle_in)
pickle_in = open("tcg_y.pickle","rb")
y = pickle.load(pickle_in)
y = np.array(y)
print('Recheck input shape of the X data: ',X.shape)
print('Recheck input shape of the y data: ',y.shape)
print('Full shape of input data X is: ',X.shape[1:])
#
# nornamlize the X data here
#
maxvalue = X[:,:,:,0].flat[np.abs(X[:,:,:,0]).argmax()]
print('Normalization factor for 950 mb abs vort is: ',maxvalue)  
X[:,:,:,0] = X[:,:,:,0]/maxvalue
maxvalue = X[:,:,:,1].flat[np.abs(X[:,:,:,1]).argmax()]
print('Normalization factor for relative humidity is: ',maxvalue)
X[:,:,:,1] = X[:,:,:,1]/maxvalue
maxvalue = X[:,:,:,2].flat[np.abs(X[:,:,:,2]).argmax()]
print('Normalization factor for surface pressure is: ',maxvalue)
X[:,:,:,2] = X[:,:,:,2]/maxvalue
maxvalue = X[:,:,:,3].flat[np.abs(X[:,:,:,3]).argmax()]
print('Normalization factor for 750 mb temperature is: ',maxvalue)
X[:,:,:,3] = X[:,:,:,3]/maxvalue
#
# build a CNN model
#
dense_layers = [0]
layer_sizes = [128]
conv_layers = [3]
for dense_layer in dense_layers:
    for layer_size in layer_sizes:
        for conv_layer in conv_layers:
            NAME = "{}-conv-{}-nodes-{}-dense-{}".format(conv_layer, layer_size, dense_layer, int(time.time()))
            print('--> Running configuration: ',NAME)

            model = Sequential()
            model.add(Conv2D(layer_size, (3, 3), input_shape=X.shape[1:],data_format="channels_last"))
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
           
            for l in range(conv_layer-1): 
                model.add(Conv2D(layer_size, (3, 3)))
                model.add(Activation('relu'))
                model.add(MaxPooling2D(pool_size=(2, 2)))
            
            model.add(Flatten())

            for _ in range(dense_layer):
                model.add(Dense(layer_size))
                model.add(Activation('relu'))
            
            model.add(Dense(1))
            model.add(Activation('sigmoid'))

            tensorboard = TensorBoard(log_dir="./logs/{}".format(NAME))
            
            model.compile(loss='binary_crossentropy',
                          optimizer='adam',
                          metrics=['accuracy'])

            model.fit(X, y, batch_size=90, epochs=30, validation_split=0.1, callbacks=[tensorboard])

model.save('tcg_CNN.model')
