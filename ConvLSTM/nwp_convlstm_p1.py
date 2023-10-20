#
# NOTE: This machine learning program is for predicting TC formation, using
#       input dataset in the NETCDF format. The program treats different 
#       2D input fields as different channels of an image. This specific 
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
#       - 27, Oct 22: Added a classification loop to simplify the code
#       - 01, Nov 22: Modified to include more channels  
#       - 02, Feb 23: Revised for jupiter-notebook workflow
#       - 20, Jun 23: Updated for augmentation/dropout layers
#       - 11, Oct 23: revised for a better workflow for future upgrades
#                     and sharing
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
import libtcg_netcdfloader as tcg_loader
import libtcg_utils as tcg_utils
import libtcg_fnl as tcg_fnl
import sys
import re
#
# This function returns a kist composing of [numpy data, label] that reads from the
# set of NETCDF data under netcdf data path
#
def main(rootdir,interval,nx,ny,number_channels,nframe,yyyy):
    print('Input data dir is: ',rootdir)
    frame = np.zeros([1,nframe,ny,nx,number_channels])
    i = 0 
    j = 0
    for img in tqdm(os.listdir(rootdir)):
        try:
            infile=rootdir+'/'+img
            cycle = tcg_fnl.path2cycle(infile) 
            print('Processing file:',infile,cycle,interval*(nframe-1),frame.shape)
            last_cycle = tcg_utils.add_hour(cycle,interval*(nframe-1))
            last_file = tcg_fnl.cycle2path(rootdir,last_cycle)
            if len(yyyy) >= 4:
                match_year = re.search("fnl_"+yyyy, infile) 
            else:
                match_year = re.search("fnl_", infile)
            #print('---> Last cycle/file:',last_cycle,last_file,match_year)
            if os.path.isfile(last_file) and match_year:
                a = tcg_loader.frame12channels(rootdir,cycle,interval=interval,
                    nx=nx,ny=ny,number_channels=number_channels,nframe=nframe)
                print('Data shape is :',a.shape,cycle)
                if i == 0:
                    frame[0,:,:,:,:] = a[:,:,:,:]
                else:
                    b = np.expand_dims(a, axis=0)
                    frame = np.concatenate((frame, b), axis=0)
                    del b
                i = i + 1
            else:
                print('Do not have enough cycles for frames or unmatch year...skip',cycle,last_cycle)
        except Exception as e:
            pass
        if match_year: j += 1
        if j > 199: 
            print("Save the first 200 frames only... stop now")
            break
    return frame
#
# This function reads in a list of 4 dim and plot a random field for quick check
#
def check_visual(array_raw,plot_sample=1):
    print("Plotting one example from raw data input")
    temp = np.array(array_raw[plot_sample])
    plot_channel = 2
    fig, axs = plt.subplots(2, 4, layout="constrained",figsize=(13, 5))
    for i,ax in enumerate(axs.flat):
        CS = ax.contourf(temp[i,:,:,plot_channel])
        #ax.clabel(CS, inline=True, fontsize=10)
        #ax.colorbars()
        ax.set_title('t = 0')
        ax.grid()

    #plt.figure(figsize=(11, 8))
    #plt.subplot(1,4,2)
    #CS = plt.contour(temp[2,:,:,1])
    #plt.clabel(CS, inline=True, fontsize=10)
    #plt.title('t=-2')
    #plt.grid()

    plt.show()

if __name__ == '__main__':
    n = len(sys.argv)
    print("Total arguments input are:", n)
    print("Name of Python script:", sys.argv[0])
    if n < 2:
       print("Need one input argument for the year to process...Stop")
       print("+ Example for year 2007: nwp_convlstm_p1.py 2007")
       print("+ Example for all years: nwp_convlstm_p1.py _")
       exit()
    yyyy = str(sys.argv[1])

    rootdir="/N/project/hurricane-deep-learning/data/ncep_extracted_41x161_13vars/"
    img_nx = 161             # number of lon points/width/col
    img_ny = 41              # number of lat points/depth/row
    number_channels = 12     # number of channels 
    nframe = 9               # number of time frames
    interval_hr = -6         # hor interval between frames
    array_raw = main(rootdir,interval_hr,img_nx,img_ny,number_channels,nframe,yyyy)
    print("Raw output shape (nsample,nframe,ny,nx,nchannel) is: ",array_raw.shape)
    #
    # visualize a few variables for checking the input data
    #
    check_visualization = "no"
    if check_visualization== "yes":
        check_visual(array_raw,plot_sample=0)
    #
    # randomize data and save training data to an output for subsequent use
    #
    np.random.shuffle(array_raw)
    outfile = "nwp_convlstm_"+yyyy+".pickle"
    pickle_out = open(outfile,"wb")
    pickle.dump(array_raw, pickle_out)
    pickle_out.close()
