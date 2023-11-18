#
# NOTE: This workflow is for predicting weather, using convolutional LSTM
#       architechture. The training data is a set of reanalysis data in the
#       past for one specific domain that are regular gridded at regular
#       intervals in the NETCDF format. This system consists of three stages
#       as given below:
#
#       - Stage 1: reading NETCDF input and generating training dataset with a
#                  given image sizes, number of frames, number of sample, and
#                  number of channels, which are saved by pickle;
#       - Stage 2: import the saved pickle data and split this data into a lag
#                  pair (X,Y), with the lag time (forecast lead time) prescribed
#                  in advance). This stage will then build a convolutional LSTM
#                  model with a given training/validation ratio, and then save
#                  the train model under the name "nwp_model_hhh", where hhh is
#                  forecast lead time. It also saves the history of training
#                  in the form of pickle format for later analysis.
#       - Stage 3: testing the performance of the model by importing the best
#                  trained model from Stage 2, and make a list of prediction
#                  to be validated with the test data. Note that this stage is
#                  best to run in the Jupyter notebook mode so the prediction
#                  can be visuallly checked.
#
# INPUT: This Stage 2 script requires an input dataset in the pickle that are 
#        produced from Stage 1. Note that the input data are given in each year
#        separately, because Stage 1 takes a long time ro process and so it is
#        designed to process year by year only. 
#
# OUTPUT: A best trained model that is saved as "nwp_model_hhh", where hhh is
#         forecast lead time, and the training history in the form of pickle 
#         format for later analysis.
#
#         Remark: this script should only be run in the job submission mode 
#         instead of Jupyter notebook, as it requires a lot of time and memory
#         that will be crashed easily if running with Jupyter.
#
# HIST: - 12, Oct 23: Created by CK
#       - 10, Nov 23: revised for a better workflow for future upgrades
#                     and sharing
#       - 16, Nov 23: added number of channels option 
#
# AUTH: Chanh Kieu (Indiana University, Bloomington. Email: ckieu@iu.edu)
#
#==========================================================================
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pickle
import time
from tensorflow.keras.callbacks import TensorBoard
import libtcg_utils as tcg_utils
#
# function to read in data from pickle. Note that we have to flip the frame here
# as the data saved in pickle is arraged as (t,t-1,t-2,...t-n). We need to have
# here as an output the order (t-n,...,t-2,t-1,t)
#
def readindata(year_list):
    for i,yyyy in enumerate(year_list):
        #print("Openning year data: ",yyyy)
        pickle_in = open("nwp_convlstm_"+yyyy+".pickle","rb")
        X = pickle.load(pickle_in)
        nc = X.shape[-1]
        nf = X.shape[1]
        nx = X.shape[3]
        ny = X.shape[2]
        nb = X.shape[0]
        #print('--->Input shape of the X features data: ',X.shape)
        #print('--->Number of input channel extracted from X is: ',nc)
        #print('--->Number of input frames extracted from X is: ',nf)
        if i == 0:
            #datain = np.zeros((nb,nf,ny,nx,nc))
            datain = X
        else:
            datain = np.concatenate((datain,X),axis=0)
        reversed_datain = np.flip(datain,axis=1)    
        print("Year and its input data shape:",yyyy,datain.shape)
    return reversed_datain
#
# Split into train and validation sets using indexing to optimize memory, using
# 90% for training and 10% for validation
#
def data_split_normalize(dataset,ratio=0.8):
    indexes = np.arange(dataset.shape[0])
    np.random.shuffle(indexes)
    train_index = indexes[: int(ratio*dataset.shape[0])]
    val_index = indexes[int(ratio*dataset.shape[0]):]
    train_dataset = dataset[train_index]
    val_dataset = dataset[val_index]

    train_dataset = tcg_utils.normalize_frame_data(train_dataset)
    val_dataset = tcg_utils.normalize_frame_data(val_dataset)
    #print("Check normalized max for channel 1",tcg_utils.maxval_framedata(train_dataset,1,1))
    #print("Check normalized max for channel 5",tcg_utils.maxval_framedata(train_dataset,5,1))
    #print("Check normalized max for channel 9",tcg_utils.maxval_framedata(train_dataset,9,1))
    return train_dataset,val_dataset
#
# We'll define a helper function to shift the frames, where
# `x` is frames 0 to n - 1, and `y` is frames 1 to n. Note that this function
# will be important as it defines the lead times. Shifting 1 means 1-lead time
# ahead. Likewise, shifting k means k-lead time ahead.
#
def create_shifted_frames(data,shift=1):
    print("Input for the shift function is: ",data.shape)
    x = data[:, 0:(data.shape[1]-shift), :, :, :]
    y = data[:, shift:data.shape[1], :, :, :]
    print("After shifting produces:", x.shape,y.shape)
    return x, y
#
# function to quick check visualization
#
def visulization(train_dataset,channel=1):
    fig, axes = plt.subplots(2, 4, figsize=(12, 4))
    data_choice = np.random.choice(range(len(train_dataset)), size=1)[0]
    #data_choice = 3    
    print("random choice of data to be plot is: ",data_choice)
    for idx, ax in enumerate(axes.flat):
        #print("Plotting figures: ",data_choice,idx,ax)
        cs=ax.contourf(train_dataset[data_choice,idx,:,:,channel],cmap='coolwarm')#,vmin=0.8,vmax=1)
        ax.set_title(f"Frame {idx + 1}")
        ax.axis("off")        
        fig.colorbar(cs)

    print(f"Displaying frames for example {data_choice}.")
    plt.show()
#
# A function to define a ConvLSTM model. Note that the input layer has no fixed frame size.
# The model will consist of 3 `ConvLSTM2D` layers with batch normalization, followed by a `Conv3D` 
# layer for the spatiotemporal outputs, which is similar to Keras' video prediction model based on 
# ConvLSTM. 
# 
# Note that the loss function is important here. Need to implement it right, or otherwise the model
# will not converge. Check https://neptune.ai/blog/keras-loss-functions for some loss functions.
#
def convlstm_prediction_model(x_train,number_channels=3):
    input = layers.Input(shape=(None, *x_train.shape[2:]))
    print("Input shape is", input.shape)
    
    x = layers.ConvLSTM2D(filters=16, kernel_size=(3, 3), padding="same", return_sequences=True, activation="relu")(input)
    x = layers.BatchNormalization()(x)
    
    x = layers.ConvLSTM2D(filters=32,kernel_size=(3, 3),padding="same",return_sequences=True,activation="relu")(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.ConvLSTM2D(filters=64,kernel_size=(1, 1),padding="same",return_sequences=True,activation="relu")(x)
    x = layers.BatchNormalization()(x)

    x = layers.ConvLSTM2D(filters=64,kernel_size=(1, 1),padding="same",return_sequences=True,activation="relu")(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.Conv3D(filters=number_channels, kernel_size=(3, 3, 3), activation="sigmoid", padding="same")(x)
        
    model = keras.models.Model(input,x)
    #model.compile(loss=keras.losses.binary_crossentropy, optimizer=keras.optimizers.Adam())
    #model.compile(loss=large_scale_loss_function,optimizer='adam')
    model.compile(loss=tf.keras.losses.MeanAbsolutePercentageError(), 
                  optimizer=keras.optimizers.Adam(),
                  metrics=[tf.keras.metrics.RootMeanSquaredError()])
    model.summary()
    return model
#
# create our own lost function for the problem
#
def large_scale_loss_function(y_true, y_pred):
   squared_difference = tf.square(y_true - y_pred)
   return tf.reduce_mean(squared_difference, axis=-1)
#
# This is main program that builds a Convolutional LSTM model. It requires a list of
# year to be processed, which are generated from Stage 1 in the pickle format. See
# the year_list below for any changes needed. It also requires pre-setting the 
# batch size and epochs for training.
#
if __name__ == '__main__':
    #
    # read in data output from Part 1, which is in the format yyyy_nc with nc being
    # the number of channels. 
    #
    #year_list = ["2008","2009","2010","2011","2012","2013","2014",
    #             "2015","2016","2017","2018","2019","2020","2021"]
    year_list = ["2008_3","2009_3"]
    epochs = 200
    batch_size = 32 
    number_channels = 3
    datain = readindata(year_list)
    #
    # normalize and split data into train/validation
    #
    train_dataset,val_dataset = data_split_normalize(datain,ratio=0.9)
    print("train and val dataset shapes are:",train_dataset.shape,val_dataset.shape)
    #
    # Apply the processing function to the datasets.
    #
    x_train, y_train = create_shifted_frames(train_dataset,1)
    x_val, y_val = create_shifted_frames(val_dataset,1)
    print("Training dataset (x,y) shape: " + str(x_train.shape) + ", " + str(y_train.shape))
    print("Validation dataset (x,y) Shape: " + str(x_val.shape) + ", " + str(y_val.shape))
    #
    # quick visaluzation
    #
    visualize = "no"
    if visualize == "yes":
        visulization(train_dataset,channel=4)
        #visulization(x_train,channel=4)
    #
    # define ConvLSTM model and callbacks here before fitting the model
    #
    model = convlstm_prediction_model(x_train,number_channels=number_channels)
    bestmodel = "nwp_model_06h_"+str(number_channels)
    early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=10)
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=5)    
    save_best_model = keras.callbacks.ModelCheckpoint(bestmodel,save_best_only=True)
    history = model.fit(x_train,y_train,batch_size=batch_size,epochs=epochs,
                        validation_data=(x_val, y_val),callbacks=[early_stopping,reduce_lr,save_best_model])
    with open('./nwp_convlstm_history.pickle', 'wb') as out:
        pickle.dump(history.history,out)
