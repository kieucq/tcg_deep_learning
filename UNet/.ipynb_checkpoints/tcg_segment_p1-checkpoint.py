#
# NOTE: This machine learning program is for predicting TC formation, using
#       input dataset in the NETCDF format and U-Net architechture based on
#       segmentation. The program treats different 2D input fields as different 
#       channels of an image. This specific 
#       program requires a set of 12 2D-variables (12-channel image) and 
#       consists of three stages
#       - Stage 1: reading NETCDF input and generating (X,y) data with a 
#                  given image sizes, which are then saved by pickle;
#       - Stage 2: import the saved pickle (X,y) pair and build a CNN model
#                  with a given training/validation ratio, and then save
#                  the train model under tcg_CNN.model.
#       - Stage 3: import the trained model from Stage 2, and make a list
#                  of prediction from normalized test data. 
#
# INPUT: This Stage 1 script requires two specific input datasets, including
#       1. 7 meterological vars u, v,abs vort, tmp, RH, vvels, sst, cape  
#          corresponding to negative cases (i.e. no TC formation within the 
#          domain). 
#       2. Similar data but for positive cases (i.e., there is a TC centered
#          on the domain)  
#        Remarks: Note that these data must be on the standard 19 vertical
#        levels 1000, 975, 950, 925, 900, 850, 800, 750, 700, 650, 600, 
#        550, 500, 450, 400, 350, 300, 250, 200. Also, all field vars must
#        be resize to cover an area of 30x30 around the TC center for the 
#        positive data cases.
#
# OUTPUT: A set of pairs (X,y) needed for CNN training
#
# HIST: - 25, Oct 22: Created by CK
#       - 04, Aug 23: Revised for Segmentation processing data
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
