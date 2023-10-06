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
    a2 = np.zeros((nx,ny))
    print(nx,ny,nz)
    for i in range(a2.shape[0]):
        for j in range(a2.shape[1]):
            a2[i,j] = abv[1,j,i]
    #if nx == 30 and ny == 30:
    #    array_raw = np.array(a2)
    array_raw = np.array(a2)
    new_array = cv2.resize(array_raw, (IMG_SIZE, IMG_SIZE))
    #
    # nornamlize the X data here
    #
    maxvalue = new_array.flat[np.abs(new_array).argmax()]
    print('Normalization factor is: ',maxvalue)
    new_array = new_array/maxvalue
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
#
# call a CNN trained model and make a prediction
#
CATEGORIES = ["No", "Yes"]
model = tf.keras.models.load_model("./tcg_CNN.model")
DATADIR = "/N/u/ckieu/Carbonate/python/logs"
category = "pos"
path = os.path.join(DATADIR,category)
print(path)
for img in tqdm(os.listdir(path)):
    try:
        img_dir = DATADIR + '/' + category + '/' + img
        print('Processing image:', img_dir)
        prediction = model.predict([prepare(img_dir)])
        print(prediction,CATEGORIES[int(prediction[0][0])])
    except Exception as e: 
        pass
