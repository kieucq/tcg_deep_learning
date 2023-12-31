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
# INPUT: This Stage 1 script requires an input dataset in the NETCDF that contains
#        regular time frequency (e.g, every 3 or 6 hours), and should include
#        all basic meterological variables such as u, v, T, RH, pressure,...  
#
#        Remarks: Note that these data should be on the standard 19 vertical
#        levels 1000, 975, 950, 925, 900, 850, 800, 750, 700, 650, 600, 
#        550, 500, 450, 400, 350, 300, 250, 200. Also, all field vars must
#        be resized to cover an area of interest.  
#
# OUTPUT: A set of training data in the with shape (sample_size, ny, nx, nchannel).
#        Note that array setting for NETCDF and Python are arraged such that
#        ny is the number of rows (depth), while nx is the number of col (width).  
#
# HIST: - 12, Oct 23: Created by CK
#       - 10, Nov 23: revised for a better workflow for future upgrades
#                     and sharing
#       - 16, Nov 23: Added 3-channel option for data generator
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
                if number_channels == 12:  
                    a = tcg_loader.frame12channels(rootdir,cycle,interval=interval,
                        nx=nx,ny=ny,number_channels=number_channels,nframe=nframe)
                elif number_channels == 3:
                    a = tcg_loader.frame3channels(rootdir,cycle,interval=interval,
                        nx=nx,ny=ny,number_channels=number_channels,nframe=nframe)
                else:
                    print("Channels must be 3 or 12 at the moment...exit")
                    exit()
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
#
# This is the main program. Need to edit several parameters including
# rootdir, img_nx, img_ny, number_channels, nframe, interval_hr. See the
# section below for where to change these parameters.
#
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
    number_channels = 3      # number of channels 
    nframe = 9               # number of time frames
    interval_hr = -6         # hor interval between frames
    array_raw = main(rootdir,interval_hr,img_nx,img_ny,number_channels,nframe,yyyy)
    print("Raw output shape (nsample,nframe,ny,nx,nchannel) is: ",array_raw.shape)
    #
    # visualize a few variables for checking the input data. SHould be "no" 
    # if running in the job submission mode at all times.
    #
    check_visualization = "no"
    if check_visualization== "yes":
        check_visual(array_raw,plot_sample=0)
    #
    # randomize data and save training data to an output for subsequent use
    #
    np.random.shuffle(array_raw)
    outfile = "nwp_convlstm_"+yyyy+"_"+str(number_channels)+".pickle"
    pickle_out = open(outfile,"wb")
    pickle.dump(array_raw, pickle_out)
    pickle_out.close()
