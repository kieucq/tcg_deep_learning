import cv2
import tensorflow as tf
import os
from tqdm import tqdm
import netCDF4
import numpy as np
#
# Function to return input data as an numpy array
#
def prepare(filepath):
    IMG_SIZE = 30  
    f = netCDF4.Dataset(filepath)
    abv = f.variables['absvprs']
    nx = np.size(abv[0,0,:])
    ny = np.size(abv[0,:,0])
    nz = np.size(abv[:,0,0])
    #print('Dimension of input NETCDF is: ',nx,ny,nz)
    a2 = np.zeros((nx,ny,4))
    for i in range(a2.shape[0]):
        for j in range(a2.shape[1]):
           a2[i,j,0] = abv[1,j,i]    # abs vort at 950 mb
    rel = f.variables['rhprs']
    for i in range(a2.shape[0]):
        for j in range(a2.shape[1]):
           a2[i,j,1] = rel[7,j,i]    # RH at 750 mb
    sfc = f.variables['pressfc']
    for i in range(a2.shape[0]):
        for j in range(a2.shape[1]):
           a2[i,j,2] = sfc[j,i]      # surface pressure
    tmp = f.variables['tmpprs']
    for i in range(a2.shape[0]):
        for j in range(a2.shape[1]):
           a2[i,j,3] = tmp[13,j,i]   # temperature at 400 mb
    new_array = cv2.resize(a2, (IMG_SIZE, IMG_SIZE))
    #
    # normalize the data
    #
    maxvalue = new_array[:,:,0].flat[np.abs(new_array[:,:,0]).argmax()]
    #print('Normalization factor for 950 mb abs vort is: ',maxvalue)
    new_array[:,:,0] = new_array[:,:,0]/maxvalue
    maxvalue = new_array[:,:,1].flat[np.abs(new_array[:,:,1]).argmax()]
    #print('Normalization factor for relative humidity is: ',maxvalue)
    new_array[:,:,1] = new_array[:,:,1]/maxvalue
    maxvalue = new_array[:,:,2].flat[np.abs(new_array[:,:,2]).argmax()]
    #print('Normalization factor for surface pressure is: ',maxvalue)
    new_array[:,:,2] = new_array[:,:,2]/maxvalue
    maxvalue = new_array[:,:,3].flat[np.abs(new_array[:,:,3]).argmax()]
    #print('Normalization factor for 750 mb temperature is: ',maxvalue)
    new_array[:,:,3] = new_array[:,:,3]/maxvalue
    out_array = np.reshape(new_array, (-1, IMG_SIZE, IMG_SIZE, 4))
    #print('reshape new_array returns: ',out_array.shape)
    return out_array
#
# call a CNN trained model and make a prediction
#
CATEGORIES = ["No", "Yes"]
model = tf.keras.models.load_model("./tcg_CNN.model")
DATADIR = "/N/u/ckieu/Carbonate/python/logs"
category = "neg"
path = os.path.join(DATADIR,category)
print(path)
for img in tqdm(os.listdir(path)):
    try:
        img_dir = DATADIR + '/' + category + '/' + img
        print('Processing image:', img_dir)
        print('Input image dimension is: ',prepare(img_dir).shape)
        prediction = model.predict([prepare(img_dir)])
        print(prediction,round(prediction[0][0]),CATEGORIES[round(prediction[0][0])])
        #input("Press Enter to continue...")
    except Exception as e: 
        pass
