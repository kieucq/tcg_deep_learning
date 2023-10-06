import tensorflow as tf
import numpy as np
import os
import pandas as pd
#import imageio
import imageio.v2 as imageio
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dropout 
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import concatenate
#
# reading data
#
path = '/N/u/ckieu/Carbonate/model/deep-learning/Unet'
image_path = os.path.join(path, './data/images/')
mask_path = os.path.join(path, './data/target/')
image_list_orig = os.listdir(image_path)
image_list = [image_path+i for i in image_list_orig]
temp_list = [mask_path+i for i in image_list_orig]
mask_list = [sub.replace('.jpg', '.png') for sub in temp_list]
print('image_list_orig',image_list_orig[-2:])
print('image_list',image_list[-2:])
print('mask_list',mask_list[-2:])
#
# plot a few sample
#
visualization="NO"
if visualization=="YES":
    N = 2
    img = imageio.imread(image_list[N])
    mask = imageio.imread(mask_list[N])
    fig, arr = plt.subplots(1, 2, figsize=(14, 10))
    arr[0].imshow(img)
    arr[0].set_title('Image')
    arr[1].imshow(mask[:, :])
    arr[1].set_title('Segmentation')
    print(mask[250,:])
    print(img.shape,mask.shape)
#
# bundle the image and target into a single TF dataset type
#
image_filenames = tf.constant(image_list)
masks_filenames = tf.constant(mask_list)

dataset = tf.data.Dataset.from_tensor_slices((image_filenames, masks_filenames))
for image, mask in dataset.take(3):
    print(image)
    print(mask)
print(len(list(dataset)),dataset.cardinality().numpy())    
#
# preprocess the data and normalize such that the data values are now betwee  [0,1]
#
def process_path(image_path, mask_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1)
    mask = tf.math.reduce_max(mask, axis=-1, keepdims=True)
    return img, mask

def preprocess(image, mask):
    input_image = tf.image.resize(image, (64, 64), method='nearest')
    input_mask = tf.image.resize(mask, (64, 64), method='nearest')
    print(image.shape,input_image.shape)
    return input_image, input_mask

image_ds = dataset.map(process_path)
processed_image_ds = image_ds.map(preprocess)
#
# Define U-Net encoder block, which consists of 2 Conv and output next_layer + skip block
#
def conv_block(inputs=None, n_filters=32, dropout_prob=0, max_pooling=True):
    """
    Convolutional downsampling block
    
    Arguments:
        inputs -- Input tensor
        n_filters -- Number of filters for the convolutional layers
        dropout_prob -- Dropout probability
        max_pooling -- Use MaxPooling2D to reduce the spatial dimensions of the output volume
    Returns: 
        next_layer, skip_connection --  Next layer and skip connection outputs
    """

    conv = Conv2D(n_filters, # Number of filters
                  3,         # Kernel size   
                  activation='relu',
                  padding='same',
                  kernel_initializer='he_normal')(inputs)
    conv = Conv2D(n_filters, # Number of filters
                  3,         # Kernel size
                  activation='relu',
                  padding='same',
                  # set 'kernel_initializer' same as above
                  kernel_initializer='he_normal')(conv)
    
    # if dropout_prob > 0 add a dropout layer, with the variable dropout_prob as parameter
    if dropout_prob > 0:
        conv = Dropout(dropout_prob)(conv)     
        
    # if max_pooling is True add a MaxPooling2D with 2x2 pool_size
    if max_pooling:
        next_layer = MaxPooling2D(2,strides=2)(conv) 
    else:
        next_layer = conv
        
    skip_connection = conv
    
    return next_layer, skip_connection

def conv_block_new(inputs=None, n_filters=32, dropout_prob=0, max_pooling=True):
    """
    Convolutional downsampling block
    
    Arguments:
        inputs -- Input tensor
        n_filters -- Number of filters for the convolutional layers
        dropout_prob -- Dropout probability
        max_pooling -- Use MaxPooling2D to reduce the spatial dimensions of the output volume
    Returns: 
        next_layer, skip_connection --  Next layer and skip connection outputs
    """

    conv = Conv2D(n_filters, # Number of filters
                  3,         # Kernel size   
                  activation='relu',
                  padding='same')(inputs)
    conv = Conv2D(n_filters, # Number of filters
                  3,         # Kernel size
                  activation='relu',
                  padding='same')(conv)
    
    # if dropout_prob > 0 add a dropout layer, with the variable dropout_prob as parameter
    if dropout_prob > 0:
        conv = Dropout(dropout_prob)(conv)     
        
    # if max_pooling is True add a MaxPooling2D with 2x2 pool_size
    if max_pooling:
        next_layer = MaxPooling2D(2,strides=2)(conv) 
    else:
        next_layer = conv
        
    skip_connection = conv
    
    return next_layer, skip_connection
#
# Define U-Net decoder block
#
def upsampling_block(expansive_input, contractive_input, n_filters=32):
    """
    Convolutional upsampling block
    
    Arguments:
        expansive_input -- Input tensor from previous layer
        contractive_input -- Input tensor from previous skip layer
        n_filters -- Number of filters for the convolutional layers
    Returns: 
        conv -- Tensor output
    """
    
    up = Conv2DTranspose(
                 n_filters,    # number of filters
                 3,            # Kernel size
                 strides=2,
                 padding='same')(expansive_input)
    
    # Merge the previous output and the contractive_input
    merge = concatenate([up, contractive_input], axis=3)
    conv = Conv2D(n_filters,   # Number of filters
                 3,            # Kernel size
                 activation='relu',
                 padding='same',
                 kernel_initializer='he_normal')(merge)
    conv = Conv2D(n_filters,  # Number of filters
                 3,   # Kernel size
                 activation='relu',
                 padding='same',
                  # set 'kernel_initializer' same as above
                 kernel_initializer='he_normal')(conv)
    
    return conv

def upsampling_block_new(expansive_input, contractive_input, n_filters=32):
    """
    Convolutional upsampling block
    
    Arguments:
        expansive_input -- Input tensor from previous layer
        contractive_input -- Input tensor from previous skip layer
        n_filters -- Number of filters for the convolutional layers
    Returns: 
        conv -- Tensor output
    """
    
    up = Conv2DTranspose(
                 n_filters,    # number of filters
                 3,            # Kernel size
                 strides=2,
                 padding='same')(expansive_input)
    
    # Merge the previous output and the contractive_input
    merge = concatenate([up, contractive_input], axis=3)
    conv = Conv2D(n_filters,   # Number of filters
                 3,            # Kernel size
                 activation='relu',
                 padding='same')(merge)
    conv = Conv2D(n_filters,  # Number of filters
                 3,   # Kernel size
                 activation='relu',
                 padding='same')(conv)
    
    return conv
#
# define U-Net model finally
#
def unet_model(input_size=(64, 64, 3), n_filters=32, n_classes=3):
    """
    Unet model
    
    Arguments:
        input_size -- Input shape 
        n_filters -- Number of filters for the convolutional layers
        n_classes -- Number of output classes
    Returns: 
        model -- tf.keras.Model
    """
    inputs = Input(input_size)
    # Contracting Path (encoding)
    # Add a conv_block with the inputs of the unet_ model and n_filters
    cblock1 = conv_block(inputs=inputs, n_filters=n_filters)
    # Chain the first element of the output of each block to be the input of the next conv_block. 
    # Double the number of filters at each new step
    cblock2 = conv_block(inputs=cblock1[0], n_filters=n_filters*2)
    cblock3 = conv_block(inputs=cblock2[0], n_filters=n_filters*4)
    cblock4 = conv_block(inputs=cblock3[0], n_filters=n_filters*8, dropout_prob=0.3) 
    # Include a dropout_prob of 0.3 for this layer, and avoid the max_pooling layer
    cblock5 = conv_block(inputs=cblock4[0], n_filters=n_filters*16, dropout_prob=0.3, max_pooling=False) 
    
    # Expanding Path (decoding)
    # Add the first upsampling_block.
    # Use the cblock5[0] as expansive_input and cblock4[1] as contractive_input and n_filters * 8
    ublock6 = upsampling_block(cblock5[0],cblock4[1], n_filters=n_filters*8)
    # Chain the output of the previous block as expansive_input and the corresponding contractive block output.
    # Note that you must use the second element of the contractive block i.e before the maxpooling layer. 
    # At each step, use half the number of filters of the previous block 
    ublock7 = upsampling_block(ublock6,cblock3[1], n_filters=n_filters*4)
    ublock8 = upsampling_block(ublock7,cblock2[1], n_filters=n_filters*2)
    ublock9 = upsampling_block(ublock8,cblock1[1], n_filters=n_filters)

    conv9 = Conv2D(n_filters,
                 3,
                 activation='relu',
                 padding='same',
                 # set 'kernel_initializer' same as above exercises
                 kernel_initializer='he_normal')(ublock9)

    # Add a Conv2D layer with n_classes filter as output, kernel size of 1 and a 'same' padding
    conv10 = Conv2D(n_classes, 1, activation='softmax', padding='same')(conv9)
    
    model = tf.keras.Model(inputs=inputs, outputs=conv10)

    return model
#
# define a different simipler U-Net model
#
def unet_model_simple(input_size=(64, 64, 3), n_filters=32, n_classes=3):
    """
    Unet model
    
    Arguments:
        input_size -- Input shape 
        n_filters -- Number of filters for the convolutional layers
        n_classes -- Number of output classes
    Returns: 
        model -- tf.keras.Model
    """
    inputs = Input(input_size)
    # Contracting Path (encoding)    
    cblock1 = conv_block_new(inputs=inputs, n_filters=n_filters)    
    cblock2 = conv_block_new(inputs=cblock1[0], n_filters=n_filters*2)
    cblock3 = conv_block_new(inputs=cblock2[0], n_filters=n_filters*4, dropout_prob=0.3, max_pooling=False) 
    
    # Expanding Path (decoding)    
    ublock4 = upsampling_block_new(cblock3[0],cblock2[1], n_filters=n_filters*2)    
    ublock5 = upsampling_block_new(ublock4,cblock1[1], n_filters=n_filters)        

    #conv9 = Conv2D(n_filters,
    #             3,
    #             activation='relu',
    #             padding='same',
    #             # set 'kernel_initializer' same as above exercises
    #             kernel_initializer='he_normal')(ublock5)

    # Add a Conv2D layer with n_classes filter as output, kernel size of 1 and a 'same' padding
    conv6 = Conv2D(n_classes, 1, activation='softmax', padding='same')(ublock5)
    
    model = tf.keras.Model(inputs=inputs, outputs=conv6)

    return model
#
# Call UNet model now
#
img_height = 64
img_width = 64
num_channels = 3

unet = unet_model_simple((img_height, img_width, num_channels),n_filters=8)
unet.summary()

#unet.compile(optimizer=tf.keras.optimizers.Adam(clipnorm=1.0),
#              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#              metrics=['accuracy'])

unet.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=['accuracy'])

def display(display_list):
    plt.figure(figsize=(15, 15))

    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()

if visualization=="YES":    
    print(processed_image_ds.cardinality().numpy())
    for image, mask in image_ds.take(1):    
        sample_image, sample_mask = image, mask
        print(image.shape,mask.shape)
    display([sample_image, sample_mask])

    for image, mask in processed_image_ds.take(1):    
        sample_image, sample_mask = image, mask
        print(image.shape,mask.shape)
    display([sample_image, sample_mask])
#
# train Unet model now
#
EPOCHS = 5
VAL_SUBSPLITS = 5
BUFFER_SIZE = 100
BATCH_SIZE = 64
train_split=0.7
val_split=0.2
data_size=processed_image_ds.cardinality().numpy()
train_size = int(train_split*data_size)
val_size = int(val_split*data_size)    
print("train_size, val_size, data_size: ",train_size,val_size,data_size)

train_dataset = processed_image_ds.take(train_size).batch(BATCH_SIZE)    
val_dataset = processed_image_ds.skip(train_size).take(val_size).batch(BATCH_SIZE)
#train_dataset = processed_image_ds.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
print(processed_image_ds.element_spec)
model_history = unet.fit(train_dataset, epochs=EPOCHS, batch_size=BATCH_SIZE)
