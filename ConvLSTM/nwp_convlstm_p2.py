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
def convlstm_prediction_model(x_train):
    input = layers.Input(shape=(None, *x_train.shape[2:]))
    print("Input shape is", input.shape)
    
    x = layers.ConvLSTM2D(filters=16, kernel_size=(3, 3), padding="same", return_sequences=True, activation="relu")(input)
    x = layers.BatchNormalization()(x)
    
    x = layers.ConvLSTM2D(filters=32,kernel_size=(3, 3),padding="same",return_sequences=True,activation="relu")(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.ConvLSTM2D(filters=32,kernel_size=(1, 1),padding="same",return_sequences=True,activation="relu")(x)
    
    x = layers.Conv3D(filters=12, kernel_size=(3, 3, 3), activation="sigmoid", padding="same")(x)
        
    model = keras.models.Model(input,x)
    #model.compile(loss=keras.losses.binary_crossentropy, optimizer=keras.optimizers.Adam())
    #model.compile(loss=large_scale_loss_function,optimizer='adam')
    model.compile(loss=tf.keras.losses.MeanAbsolutePercentageError(), optimizer=keras.optimizers.Adam())
    model.summary()
    return model
#
# creat our own lost function for the problem
#
def large_scale_loss_function(y_true, y_pred):
   squared_difference = tf.square(y_true - y_pred)
   return tf.reduce_mean(squared_difference, axis=-1)
#
# main program
#
if __name__ == '__main__':
    #
    # read in data output from Part 1 and normalize it
    #
    year_list = ["2020","2021"]
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
    model = convlstm_prediction_model(x_train)
    early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=10)
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=5)    
    epochs = 100
    batch_size = 32   
    model.fit(x_train,y_train,batch_size=batch_size,epochs=epochs,
              validation_data=(x_val, y_val),callbacks=[early_stopping, reduce_lr])
