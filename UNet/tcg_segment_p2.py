# NOTE: This deep learning model is for predicting TC formation, using
#       input dataset in the NETCDF format and U-Net architecture based on
#       segmentation of TC genesis location. The program treats different
#       meteorological 2D input fields as different channels of an image.
#       This U-Net model is currently designed for inpud data of 12 2D-variables
#       (12-channel image) and consists of four different stages
#       - Stage 0: generating segment mask data for training, validation and
#                  test sets.
#       - Stage 1: reading NETCDF input and generating (X,Y) data with a
#                  given image sizes, which are then saved by pickle;
#       - Stage 2: import the saved pickle (X,Y) pair and build a U-Net model
#                  with a given training/validation ratio, and then save
#                  the train model under tcg_CNN.model.
#       - Stage 3: import the trained model from Stage 2, and make a list
#                  of prediction from normalized test data.
#
# INPUT: This Stage 2 script requires input datasets containing 12-channel 
#        arrays (X) and corresponding segmentation data (Y) in pickle format
#        that are re-sized for U-Net model.
#
# REMARKS: Note that this script is designed for a specific size (N,64,128,12)
#        for input. The script has to be re-run for different lead times. Also
#        the output class will be 0/1 for no/yes TC location. 
#
# OUTPUT:  A best model obtained from U-Net training.
#
# HIST: - 25, Oct 22: Created by CK
#       - 04, Aug 23: Revised for Segmentation processing data
#       - 23, Aug 23: cleaned up for better workflow by CK
#
# AUTH: Chanh Kieu (Indiana University, Bloomington. Email: ckieu@iu.edu)
#
#==========================================================================
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
import numpy as np
import pickle
import time
from tensorflow.keras.callbacks import TensorBoard
import os
import matplotlib.pyplot as plt
from tensorflow.keras.utils import load_img, img_to_array
import cv2
print("Ready")
#
# read in data output from Part 1
#
pickle_in = open("tcg_segment_X.pickle","rb")
X = pickle.load(pickle_in)
pickle_in = open("tcg_segment_Y.pickle","rb")
Y = pickle.load(pickle_in)
number_channels=X.shape[3]
print('Input shape of the X features data: ',X.shape)
print('Input shape of the Y mask target data: ',Y.shape)
print('Number of input channel extracted from X is: ',number_channels)
#
# normalize the data before training the model
#
nsample = X.shape[0]
for i in range(nsample):
    for var in range(number_channels):    
        maxvalue = X[i,:,:,var].flat[np.abs(X[i,:,:,var]).argmax()]
        #print('Normalization factor for sample and channel',i,var,', is: ',abs(maxvalue))
        X[i,:,:,var] = X[i,:,:,var]/abs(maxvalue)
        maxnew = X[i,:,:,var].flat[np.abs(X[i,:,:,var]).argmax()]
        #print('-->After normalization of sample and channel',i,var,', is: ',abs(maxnew))
        #input('Enter to continue...')
print("Finish normalization...")
#
# split for val/training
# 
num_val_samples=25
train_input_imgs = X[:-num_val_samples]
train_targets = Y[:-num_val_samples]
val_input_imgs = X[-num_val_samples:]
val_targets = Y[-num_val_samples:]
img_size = (train_input_imgs.shape[1],train_input_imgs.shape[2],train_input_imgs.shape[3])
print(train_input_imgs.shape,train_targets.shape,img_size)
#
# build a Unet model
#
num_classes = 2
inputs = keras.Input(shape=img_size)
x = layers.Conv2D(64,3,strides=2,activation="relu",padding="same",name="my_first_conv2d")(inputs)
x = layers.Conv2D(64,3,activation="relu",padding="same",name="my_second_conv2d")(x)
x = layers.Conv2D(128,3,strides=2,activation="relu",padding="same",name="my_third_conv2d")(x)
x = layers.Conv2D(128,3,activation="relu",padding="same",name="my_fouth_conv2d")(x)
x = layers.Conv2D(256,3,strides=2,activation="relu",padding="same",name="my_fifth_conv2d")(x)
x = layers.Conv2D(256,3,activation="relu",padding="same",name="my_sixth_conv2d")(x)

x = layers.Conv2DTranspose(256,3,activation="relu",padding="same",name="my_first_conv2dtranpose")(x)
x = layers.Conv2DTranspose(256,3,strides=2,activation="relu",padding="same",name="my_second_conv2dtranpose")(x)
x = layers.Conv2DTranspose(128,3,activation="relu",padding="same",name="my_third_conv2dtranpose")(x)
x = layers.Conv2DTranspose(128,3,strides=2,activation="relu",padding="same",name="my_fouth_conv2dtranpose")(x)
x = layers.Conv2DTranspose(64,3,activation="relu",padding="same",name="my_fifth_conv2dtranpose")(x)
x = layers.Conv2DTranspose(64,3,strides=2,activation="relu",padding="same",name="my_sixth_conv2dtranpose")(x)

outputs = layers.Conv2D(num_classes,3,activation="softmax", padding="same")(x)
model = keras.Model(inputs,outputs)
model.summary()
keras.utils.plot_model(model)
#
# traing model
#
model.compile(optimizer="rmsprop",loss="sparse_categorical_crossentropy")
callbacks=[keras.callbacks.ModelCheckpoint("tcg_segment_model.keras",save_best_only=True)]
history = model.fit(train_input_imgs,train_targets,
                    epochs=100,
                    callbacks=callbacks,
                    batch_size=64,
                    validation_data=(val_input_imgs,val_targets))

#
# make a test prediction now
#
model = keras.models.load_model("tcg_segment_model.keras")
test_image = val_input_imgs[8]
prediction_batch = model.predict(np.expand_dims(test_image,axis=0))
prediction = prediction_batch[0,:,:,:]
print(prediction.shape,test_image.shape)
print(prediction[:,:,0])

