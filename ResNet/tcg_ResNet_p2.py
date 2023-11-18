#
# NOTE: This ResNet model is for predicting TC formation, using the 
#       architechture provided in deep-learning coursera. The model treats 
#       different 2D input fields as input channels of an image. This specific
#       program requires a set of input data from Stage 2 of the following
#       3-stage workflow
#       - Stage 1: reading NETCDF input and generating (X,y) data with a
#                  given image sizes, which are then saved by pickle;
#       - Stage 2: import the saved pickle (X,y) pair and build a CNN model
#                  with a given training/validation ratio, and then save
#                  the train model under tcg_CNN.model.
#       - Stage 3: import the trained model from Stage 2, and make a list
#                  of prediction from normalized test data.
#
# INPUT: This Stage 2 script requires two specific input datasets that are
#        generated from Step 1, including
#        1. tcg_ResNet_X.pickle: data contains all images of yes/no TCG events, 
#           each of these images must have 12 channels
#        2. tcg_ResNet_y.pickle: data contains all labels of each image (i.e., 
#           yes or no) of TCG corresponding to each data in X.
#
#        Remarks: Note that each channel must be normalized separately. Also,
#        the script requires a large memory allocation. So users need to have
#        GPU version to run this.
#
# OUTPUT: The best ResNet model built from Keras that is saved under 
#        tcg_ResNet.model
#
# HIST: - 27, May 23: Created by CK from the open source ResNet50 model in 
#                     the deep learning course.
#       - 12, Jun 23: Added ResNet-20, and ResNet-22 model and re-organized the
#                     workflow for better fit with the TCG prediction problem.
#       - 18, Nov 23: re-designed the workflow for better maintenance in the 
#                     future. Also the histories are saved in pickle for Stage 3
#                     to make it easier to check.
#
# AUTH: Chanh Kieu (Indiana University, Bloomington. Email: ckieu@iu.edu)
#===============================================================================
import tensorflow as tf
import numpy as np
import scipy.misc
from tensorflow import keras
from tensorflow.keras.applications.resnet_v2 import ResNet50V2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet_v2 import preprocess_input, decode_predictions
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.initializers import random_uniform, glorot_uniform, constant, identity
from tensorflow.python.framework.ops import EagerTensor
from matplotlib.pyplot import imshow
import pickle
import sys
import libtcg_utils as tcg_utils
#
# Building the identity_block for ResNet with 3 convolutional layers
#
def identity_block(X, f, filters, training=True, initializer=random_uniform):        
    # Retrieve Filters
    F1, F2, F3 = filters
    
    # Save the input value. Will need this later to add back to the main path. 
    X_shortcut = X
    
    # First component of the main path
    X = Conv2D(filters = F1, kernel_size = 1, strides = (1,1), padding = 'valid', kernel_initializer = initializer(seed=0))(X)
    X = BatchNormalization(axis = 3)(X, training = training) # Default axis
    X = Activation('relu')(X)
    
    # Second component of main path    
    X = Conv2D(filters = F2, kernel_size = f, strides = (1,1), padding = 'same', kernel_initializer = initializer(seed=0))(X)
    X = BatchNormalization(axis = 3)(X, training = training)
    X = Activation('relu')(X)

    # Third component of main path
    X = Conv2D(filters = F3, kernel_size = 1, strides = (1,1), padding = 'valid', kernel_initializer = initializer(seed=0))(X)
    X = BatchNormalization(axis = 3)(X, training = training) 
    
    # Final step: Add shortcut value to main path, and pass it through a RELU activation
    X = Add()([X_shortcut,X])
    X = Activation('relu')(X) 
    return X
# 
# Building the convolutional_block for ResNet with 3 convolutional layers
#
def convolutional_block(X, f, filters, s = 2, training=True, initializer=glorot_uniform):    
    # Retrieve Filters
    F1, F2, F3 = filters
    
    # Save the input value
    X_shortcut = X
    
    # First component of main path glorot_uniform(seed=0)
    X = Conv2D(filters = F1, kernel_size = 1, strides = (s, s), padding='valid', kernel_initializer = initializer(seed=0))(X)
    X = BatchNormalization(axis = 3)(X, training=training)
    X = Activation('relu')(X)
    
    # Second component of main path
    X = Conv2D(filters = F2, kernel_size = f, strides = (1, 1), padding='same', kernel_initializer = initializer(seed=0))(X)
    X = BatchNormalization(axis = 3)(X, training=training)
    X = Activation('relu')(X)

    # Third component of main path
    X = Conv2D(filters = F3, kernel_size = 1, strides = (1, 1), padding='valid', kernel_initializer = initializer(seed=0))(X) 
    X = BatchNormalization(axis = 3)(X, training=training)
    
    # Shortcut path 
    X_shortcut = Conv2D(filters = F3, kernel_size = 1, strides = (s, s), padding='valid', kernel_initializer = initializer(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis = 3)(X_shortcut, training=training)

    # Final step: Add shortcut value to main path and pass it through a RELU activation
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)
    return X
# 
# Building the ResNet-40 model for TCG classifications. The default input shape is a 
# tuple (30,30,12), but the actual shape is passed from the function call below. 
# Likewise, the default number of classes is 1.
#
def ResNet40(input_shape = (30, 30, 12), classes = 1):
    # Define the input as a tensor with shape input_shape
    X_input = Input(input_shape)
    
    # Zero-Padding
    X = ZeroPadding2D((2, 2))(X_input)
    
    # Stage 1 - 1 layer
    X = Conv2D(64, (5, 5), strides = (2, 2), kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 1)(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)

    # Stage 2 - 9 layers
    X = convolutional_block(X, f = 3, filters = [64, 64, 256], s = 1)
    X = identity_block(X, 3, [64, 64, 256])
    X = identity_block(X, 3, [64, 64, 256])
    
    # Stage 3 - 12 layers
    X = convolutional_block(X, f = 3, filters = [128, 128, 512], s = 2)
   
    # the 3 `identity_block` with correct values of `f` and `filters` for this stage
    X = identity_block(X, 3, [128, 128, 512])
    X = identity_block(X, 3, [128, 128, 512])
    X = identity_block(X, 3, [128, 128, 512])

    # Stage 4 - 18 layers    
    X = convolutional_block(X, f = 3, filters = [256, 256, 1024], s = 2)
    
    # the 5 `identity_block` with correct values of `f` and `filters` for this stage
    X = identity_block(X, 3, [256, 256, 1024])
    X = identity_block(X, 3, [256, 256, 1024])
    X = identity_block(X, 3, [256, 256, 1024])
    X = identity_block(X, 3, [256, 256, 1024])
    X = identity_block(X, 3, [256, 256, 1024])

    # Stage 5 - 9 layers    
    #X = convolutional_block(X, f = 3, filters = [512, 512, 2048], s = 2)
    
    # the 2 `identity_block` with correct values of `f` and `filters` for this stage
    #X = identity_block(X, 3, [512, 512, 2048])
    #X = identity_block(X, 3, [512, 512, 2048])

    # AVGPOOL (≈1 line). Use "X = AveragePooling2D()(X)"
    X = AveragePooling2D()(X)

    # output layer - 1 dense layer
    X = Flatten()(X)
    X = Dense(classes, activation='sigmoid', kernel_initializer = glorot_uniform(seed=0))(X)    
    
    # Create model
    model = Model(inputs = X_input, outputs = X)
    return model
#
# ResNet-22 model
#
def ResNet22(input_shape = (30, 30, 12), classes = 1):
    # Define the input as a tensor with shape input_shape
    X_input = Input(input_shape)

    # Zero-Padding
    X = ZeroPadding2D((2, 2))(X_input)

    # Stage 1 - 1 layer
    X = Conv2D(64, (5, 5), strides = (2, 2), kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 1)(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)

    # Stage 2 - 9 layers
    X = convolutional_block(X, f = 3, filters = [64, 64, 256], s = 1)
    X = identity_block(X, 3, [64, 64, 256])
    X = identity_block(X, 3, [64, 64, 256])

    # Stage 3 - 12 layers
    X = convolutional_block(X, f = 3, filters = [128, 128, 512], s = 2)

    # the 3 `identity_block` with correct values of `f` and `filters` for this stage
    X = identity_block(X, 3, [128, 128, 512])
    X = identity_block(X, 3, [128, 128, 512])
    X = identity_block(X, 3, [128, 128, 512])

    # AVGPOOL (≈1 line). Use "X = AveragePooling2D()(X)"
    X = AveragePooling2D()(X)

    # output layer - 1 dense layer
    X = Flatten()(X)
    X = Dense(classes, activation='sigmoid', kernel_initializer = glorot_uniform(seed=0))(X)

    # Create model
    model = Model(inputs = X_input, outputs = X)
    return model
#
# ResNet-20 model
#
def ResNet20(input_shape = (30, 30, 12), classes = 1):
    # Define the input as a tensor with shape input_shape
    X_input = Input(input_shape)

    # Zero-Padding
    X = ZeroPadding2D((2, 2))(X_input)

    # Stage 1 - 1 layer
    X = Conv2D(64, (5, 5), strides = (2, 2), kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 1)(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)

    # Stage 2 - 9 layers
    X = convolutional_block(X, f = 3, filters = [64, 64, 256], s = 1)
    X = identity_block(X, 3, [64, 64, 256])
    X = identity_block(X, 3, [64, 64, 256])

    # Stage 3 - 9 layers
    X = convolutional_block(X, f = 3, filters = [128, 128, 512], s = 2)
    X = identity_block(X, 3, [128, 128, 512])
    X = identity_block(X, 3, [128, 128, 512])    

    # AVGPOOL (≈1 line). Use "X = AveragePooling2D()(X)"
    X = AveragePooling2D()(X)

    # output layer - 1 dense layer
    X = Flatten()(X)
    X = Dense(classes, activation='sigmoid', kernel_initializer = glorot_uniform(seed=0))(X)

    # Create model
    model = Model(inputs = X_input, outputs = X)
    return model
#
# call ResNet model and printout the summary. Note that the number of parameters for the
# batch normalization is computed as 4x # of filter due to the use of 4 parameters:
# [gamma weights, beta weights, moving_mean(non-trainable), moving_variance(non-trainable)]
# for each filter normalization.
#
def main(resnet_models=['ResNet20'],X=[],y=[],lead_time='00'):
    histories = []
    for resnet in resnet_models:
        NAME = "model_{}_{}h".format(resnet,lead_time)
        print('--> Running configuration: ',NAME)
        if resnet == "ResNet20":
            model = ResNet20(input_shape = (X.shape[1], X.shape[2], X.shape[3]), classes = 1)
        elif resnet == "ResNet22":
            model = ResNet22(input_shape = (X.shape[1], X.shape[2], X.shape[3]), classes = 1)
        elif resnet == "ResNet40":
            model = ResNet40(input_shape = (X.shape[1], X.shape[2], X.shape[3]), classes = 1)
        model.summary()
        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=[tf.keras.metrics.BinaryAccuracy(name="binary_accuracy", dtype=None, threshold=0.3)])
        callbacks=[keras.callbacks.ModelCheckpoint("tcg_" + resnet + ".model_" + str(lead_time),save_best_only=True)]
        hist = model.fit(X, Y, epochs = 100, batch_size = 128, validation_split=0.1, callbacks=callbacks)
        histories.append(hist.history)
    return histories
#
# Visualize the output of the training model (work for jupyter notebook only)
#
def view_history(history):
    import matplotlib.pyplot as plt
    val_accuracy = history.history['val_binary_accuracy']
    accuracy = history.history['binary_accuracy']
    epochs = history.epoch
    plt.plot(epochs,val_accuracy,'r',label="val binary_accuracy")
    plt.plot(epochs,accuracy,'b',label="train binary_accuracy")
    plt.legend()

    plt.figure()
    val_loss = history.history['val_loss']
    loss = history.history['loss']
    plt.plot(epochs,val_loss,'r',label="val loss")
    plt.plot(epochs,loss,'b',label="train loss")
    plt.legend()
    plt.show()
#
# main function
#
if __name__ == '__main__':
    n = len(sys.argv)
    print("Total arguments input are:", n)
    print("Name of Python script:", sys.argv[0])
    if n < 2:
       print("Need a forecast lead time to process...Stop")
       print("+ Example: tcg_ResNet_p2.py 00")
       exit()
    leadtime = str(sys.argv[1])
    print("Forecast lead time to run is: ",leadtime)
    #sys.exit()
    #
    # read in data output from Part 1 and normalize it
    #
    pickle_in = open("tcg_ResNet_X.pickle","rb")
    X = pickle.load(pickle_in)
    pickle_in = open("tcg_ResNet_y.pickle","rb")
    y = pickle.load(pickle_in)
    Y = np.array(y)
    number_channels=X.shape[3]
    print('Input shape of the X features data: ',X.shape)
    print('Input shape of the y label data: ',Y.shape)
    print('Number of input channel extracted from X is: ',number_channels)

    x_train,y_train = tcg_utils.normalize_channels(X,Y)
    print ("number of input examples = " + str(X.shape[0]))
    print ("X shape: " + str(X.shape))
    print ("Y shape: " + str(Y.shape))
    #
    # define the model architecture
    #
    resnets = ['ResNet20', 'ResNet22', 'ResNet40']
    histories = main(resnet_models=resnets,X=x_train,y=y_train,lead_time=leadtime)
    with open('./tcg_histories_resnet.pickle', 'wb') as out:
        pickle.dump(histories,out)

    check_visualization = "no"
    if check_visualization== "yes":
        view_history(histories)
