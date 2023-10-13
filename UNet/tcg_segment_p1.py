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
# INPUT: This Stage 1 script requires one specific input datasets, that contains
#       a 12-channel input data (X) and corresponding segmentation data (Y).  
#        
# REMARKS: Note that this script needs the input data for each forecast lead time. 
#       Also, the script is tailored to a specific input data with 12-channel 
#       training dataset (X). Both the segmentation data and corresponding X data
#       must be located under the same dir "rootdir".  
#
# OUTPUT:  A binary set (X,Y) saved as pickle files that matchs an image X with
#       its correpsonding segmentation Y, but re-sized to match with U-Net 
#       structure. 
#
# HIST: - 25, Oct 22: Created by CK
#       - 04, Aug 23: Revised for Segmentation processing data
#       - 23, Aug 23: cleaned up for better workflow by CK
#       - 03, Sep 23: Tested for BR200
#
# AUTH: Chanh Kieu (Indiana University, Bloomington. Email: ckieu@iu.edu) 
#
#==========================================================================
import netCDF4
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
#
# define data source and reading pos/neg data 
#
IMG_SIZE_X = 128
IMG_SIZE_Y = 64
rootdir="/N/slate/ckieu/tmp/output/ncep_extracted_41x161_12h/train/"
tcg_class = ['seg','pos']
array_input = []
array_target = []
for tcg in tcg_class:    
    datadir=rootdir + tcg + '/'
    print('Input data dir is: ',datadir)
    if tcg == "pos":
        for img in tqdm(os.listdir(datadir)):
            try:
                print('Processing file:', img)           
                file=datadir+'/'+img
                f = netCDF4.Dataset(file)
                abv = f.variables['absvprs']
                nx = np.size(abv[0,0,:])
                ny = np.size(abv[0,:,0])
                nz = np.size(abv[:,0,0])
                print('nx = ',nx,' ny = ',ny )             
                a2 = np.zeros((nx,ny,12))         
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
                        a2[i,j,3] = tmp[15,j,i]   # temperature at 400 mb
                tsf = f.variables['tmpsfc']
                for i in range(a2.shape[0]):
                    for j in range(a2.shape[1]):
                        a2[i,j,4] = tsf[j,i]      # surface temperature
                ugr = f.variables['ugrdprs']
                for i in range(a2.shape[0]):
                    for j in range(a2.shape[1]):
                        a2[i,j,5] = ugr[3,j,i]    # u-wind at 900 mb
                        a2[i,j,6] = ugr[17,j,i]   # u-wind at 300 mb
                vgr = f.variables['vgrdprs']
                for i in range(a2.shape[0]):
                    for j in range(a2.shape[1]):
                        a2[i,j,7] = vgr[3,j,i]    # v-wind at 900 mb
                        a2[i,j,8] = vgr[17,j,i]   # v-wind at 300 mb
                hgt = f.variables['hgtprs']
                for i in range(a2.shape[0]):
                    for j in range(a2.shape[1]):
                        a2[i,j,9] = hgt[3,j,i]    # geopotential at 850 mb
                wgr = f.variables['vvelprs']
                for i in range(a2.shape[0]):
                    for j in range(a2.shape[1]):
                        a2[i,j,10] = wgr[3,j,i]   # w-wind at 900 mb
                        a2[i,j,11] = wgr[17,j,i]  # w-wind at 300 mb
                a3 = cv2.resize(a2, (IMG_SIZE_X, IMG_SIZE_Y)) 
                print('a3 shape is :',a3.shape)
                #input('Enter to continue...')
                array_input.append(a3)
            except Exception as e:
                pass
    else:
        for img in tqdm(os.listdir(datadir)):
            try:
                print('Processing file:', img)           
                file=datadir+'/'+img
                f = netCDF4.Dataset(file)                
                nx = f.dimensions['lon'].size
                ny = f.dimensions['lat'].size             
                print('nx = ',nx,' ny = ',ny )             
                a2 = np.zeros((nx,ny))         
                segment = f.variables['mask']
                a2 = segment[:,:] 
                a3 = cv2.resize(a2, (IMG_SIZE_X, IMG_SIZE_Y)) 
                print('a3 shape is :',a3.shape)
                #input('Enter to continue...')
                array_target.append(a3)
            except Exception as e:
                pass                          
#
# output data (X,Y) in a separate pickle
#
import random
import pickle
y = np.array(array_target)
X = np.array(array_input)
Y = np.expand_dims(y, axis=3)
#random.Random(1332).shuffle(X)
#andom.Random(1332).shuffle(Y)
print(X.shape,Y.shape)
pickle_out = open("tcg_segment_X.pickle","wb")
pickle.dump(X, pickle_out)
pickle_out.close()
pickle_out = open("tcg_segment_Y.pickle","wb")
pickle.dump(Y, pickle_out)
pickle_out.close()
#
# visualize a few variables for checking the input data. Check the jupiter-notebook version
# of this step 1.
#
