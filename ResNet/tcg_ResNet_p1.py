#
# NOTE: This deep learning program is for predicting TC formation, using
#       input dataset in the NETCDF format and the ResNet50 architechture. 
#       The program treats different 2D input fields as different channels 
#       of an image. This ResNet program requires a set of 12 2D-variables 
#       (12-channel image) and consists of three stages
#       - Stage 1: reading NETCDF input and generating (X,y) data with a 
#                  given image sizes, which are then saved by pickle;
#       - Stage 2: import the saved pickle (X,y) pair and build a CNN model
#                  with a given training/validation ratio, and then save
#                  the train model under tcg_CNN.model.
#       - Stage 3: import the trained model from Stage 2, and make a list
#                  of prediction from normalized test data. 
#
# INPUT: This Stage 1 script requires two specific input datasets, including
#       1. 12 meterological vars u, v, abs vort, tmp, RH, vvels, sst, cape  
#          corresponding to negative cases (i.e. no TC formation within the 
#          domain). The data must be given as an array of 30x30, with a 
#          resolution of 1 degree.
#       2. Similar data but for positive cases (i.e., there is a TC centered
#          on the domain)  
# 
#        Remarks: Note that these data must be on the standard 19 vertical
#        levels 1000, 975, 950, 925, 900, 850, 800, 750, 700, 650, 600, 
#        550, 500, 450, 400, 350, 300, 250, 200. Also, all field vars must
#        be resize to cover an area of 30x30 around the TC center for the 
#        positive data cases.
#
# OUTPUT: A set of pairs (X,y) needed for ResNet training
#
# HIST: - 29, May, 23: Updated for ResNet architechture from the CNN model
#       - 08, Jun, 23: refined for better workflow
#       - 17, Nov, 23: re-worked on flow and structure,
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
import random
import pickle
import sys
import libtcg_netcdfloader as tcg_loader
import libtcg_utils as tcg_utils
#
# This function returns a list composing of [numpy data, label] that reads from the
# set of NETCDF data under netcdf data path
#
def main(datapath,IMG_SIZE=30,number_channels=12):
    tcg_class = ['pos','neg']
    array_raw = []
    for tcg in tcg_class:
        if tcg == "pos":
            datadir=datapath + 'pos'
        else:
            datadir=datapath + 'neg'
        print('Input data dir is: ',datadir)
        for img in tqdm(os.listdir(datadir)):
            try:
                file=datadir+'/'+img
                print('Processing file:', file)
                if number_channels == 12:
                    a3 = tcg_loader.create12channels(file,new_nx=IMG_SIZE,new_ny=IMG_SIZE,number_channels=12)
                elif number_channels == 3:
                    a3 = tcg_loader.create3channels(file,new_nx=IMG_SIZE,new_ny=IMG_SIZE,number_channels=3)
                else:
                    print("Number of channels must be 12 or 3 at the moment... exit")
                    exit()
                print('Data shape is :',a3.shape)
                #input('Enter to continue...')
                if tcg == "pos":
                    array_raw.append([a3, 1])
                else:
                    array_raw.append([a3, 0])
            except Exception as e:
                pass
    return array_raw
#
# This function reads in a list of 4 dim and plot a random field for quick check
#
def check_visual(array_raw,plot_sample=1):
    print("Plotting one example from raw data input")
    temp = np.array(array_raw[plot_sample][0])
    plt.figure(figsize=(11, 3))
    plt.subplot(1,3,1)
    CS = plt.contour(temp[:,:,4])
    plt.clabel(CS, inline=True, fontsize=10)
    plt.title('SST')
    plt.grid()

    plt.subplot(1,3,2)
    CS = plt.contour(temp[:,:,1])
    plt.clabel(CS, inline=True, fontsize=10)
    plt.title('RH')
    plt.grid()

    plt.subplot(1,3,3)
    CS = plt.contour(temp[:,:,9])
    plt.clabel(CS, inline=True, fontsize=10)
    plt.title('850 mb geopotential')
    plt.grid()
    plt.show()
#
# this is the main function
#
if __name__ == '__main__':
    IMG_SIZE = 32
    nchannels = 12
    #leadtime = "00"
    #datapath="/N/project/hurricane-deep-learning/data/ncep_extracted_binary_30x30/WP/"+leadtime+"/"
    n = len(sys.argv)
    print("Total arguments input are:", n)
    print("Name of Python script:", sys.argv[0])
    if n < 3:
       print("Need an input data path argument for the year to process...Stop")
       print("+ Example: tcg_ResNet_p1.py 00 /N/project/hurricane-deep-learning/data/ncep_extracted_binary_30x30/WP")
       exit()
    leadtime = str(sys.argv[1])
    datapath = str(sys.argv[2]) + "/" + leadtime + "/"
    print("Input data path to run is: ",datapath)

    array_raw = main(datapath=datapath,IMG_SIZE=IMG_SIZE,number_channels=nchannels)
    print("Raw input data shape (nsample,ny,nx,nchannel) is: ",len(array_raw),
           len(array_raw[0][0]),len(array_raw[0][0][0]),len(array_raw[0][0][0][0]))
    #
    # visualize a few variables for checking the input data
    #
    check_visualization = "no"
    if check_visualization== "yes":
        check_visual(array_raw,plot_sample=7)
    #
    # randomize data and generate training data (X,y)
    #
    np.random.shuffle(array_raw)
    X = []
    y = []
    for features,label in array_raw:
        X.append(features)
        y.append(label)
    X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 12)
    if check_visualization== "yes":
        print(X.shape)
        print(y)
    #
    # save training data to an output for subsequent use
    #
    pickle_out = open("tcg_ResNet_X.pickle","wb")
    pickle.dump(X, pickle_out)
    pickle_out.close()
    pickle_out = open("tcg_ResNet_y.pickle","wb")
    pickle.dump(y, pickle_out)
    pickle_out.close()
