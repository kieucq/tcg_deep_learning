import netCDF4
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
#
# define data source and reading data
#
#datadirp="./logs/pos/"
#datadirn="./logs/neg/"
datadirp="/N/project/pfec_climo/qmnguyen/tc_prediction/binary_datasets/ncep_WP_tc_binary/pos"
datadirn="/N/project/pfec_climo/qmnguyen/tc_prediction/binary_datasets/ncep_WP_tc_binary/neg"
print('Positive data dir is: ',datadirp)
print('Negative data dir is: ',datadirn)
array_raw = []
for img in tqdm(os.listdir(datadirp)):
    try:
        print('Processing file:', img)
        file=datadirp+'/'+img
        f = netCDF4.Dataset(file)
        abv = f.variables['absvprs']
        nx = np.size(abv[0,0,:])
        ny = np.size(abv[0,:,0])
        nz = np.size(abv[:,0,0])
        print('nx = ',nx,' ny = ',ny )         
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
        #plt.imshow(a2, cmap='gray')
        #plt.show()
        if nx == 30 and ny == 30:
            array_raw.append([a2, 1])
    except Exception as e:
        pass
for img in tqdm(os.listdir(datadirn)):
    try:
        print('Processing file:', img)
        file=datadirn+'/'+img
        f = netCDF4.Dataset(file)
        abv = f.variables['absvprs']
        nx = np.size(abv[0,0,:])
        ny = np.size(abv[0,:,0])
        nz = np.size(abv[:,0,0])
        print('nx = ',nx,' ny = ',ny )
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
        if nx == 30 and ny == 30:
            array_raw.append([a2, 0])
    except Exception as e:
        pass
array_np = np.array(array_raw)
#print(array_np)
print(array_np.shape)
#
# randomize data and generate training data (X,y)
#
import random
np.random.shuffle(array_np)
#for sample in array_np[:2]:
#    print(sample[0],sample[1])
X = []
y = []
for features,label in array_np: 
    X.append(features)
    y.append(label)
IMG_SIZE = 30
#print(X[0].reshape(-1, IMG_SIZE, IMG_SIZE, 4))
X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 4)
print(X.shape)
print(y)
#
# save training data to an output for later use
#
import pickle
pickle_out = open("tcg_X.pickle","wb")
pickle.dump(X, pickle_out)
pickle_out.close()
pickle_out = open("tcg_y.pickle","wb")
pickle.dump(y, pickle_out)
pickle_out.close()
#pickle_in = open("test_p16_X.pickle","rb")
#X = pickle.load(pickle_in)
#pickle_in = open("test_p16_y.pickle","rb")
#y = pickle.load(pickle_in)

