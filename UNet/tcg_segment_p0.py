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
# INPUT: This Stage 0 script requires two specific input datasets, including
#       1. CSV files for training, validation, and test. 
#       2. Input post-processed NETCDF data with a given domain size.  
#        
# REMARKS: Note that this script needs the CSV input data for each forecast lead time. 
#       Also, Stage 0 will produce the segmentation data with the same size as the 
#       positive label, due to the nature of U-Net model. These CSV input data can 
#       be produced, using Quan's workflow.
#
#       For the post-process input NETCDF data, this is dataset will provide the 
#       domain size to define the segmentation for TC genesis, with label 1 for
#       the area of 5x5 degree around TC genesis location, and 0 otherwise.
#
# OUTPUT: A segmentation dir for each training, validation, and test data needed for 
#       the U-Net training, which is defined by "dataout_path".  
#
# HIST: - 25, Oct 22: Created by CK
#       - 04, Aug 23: Revised for Segmentation processing data
#       - 23, Aug 23: cleaned up for better workflow by CK
#       - 03, Sep 23: Tested for Big Red 200.
#
# AUTH: Chanh Kieu (Indiana University, Bloomington. Email: ckieu@iu.edu) 
#
#==========================================================================
import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
import os
from netCDF4 import Dataset
#
# reading CSV data that is output from Quan's workflow
#
hh = '6h'
#datain_path='/N/slate/ckieu/tmp/output/2020/'
#dataout_path='/N/slate/ckieu/tmp/output/2020_binary/'
datain_path='/N/project/hurricane-deep-learning/data/ncep_extracted_41x161_13vars/'
dataout_path='/N/project/hurricane-deep-learning/data/ncep_extracted_segmentation_41x161/' + hh + '/'
df_train = pd.read_csv(datain_path+'tc_' + hh +'_train.csv')
df_val = pd.read_csv(datain_path+'tc_' + hh + '_val.csv')
df_test = pd.read_csv(datain_path+'tc_' + hh + '_test.csv')
df_all = pd.read_csv(datain_path+'tc_' + hh + '.csv')
#
# create pos/neg data from CSV
#
df_all.replace(False,'neg', inplace=True)
df_all.replace(True,'pos', inplace=True)
data_all_full=df_all[['Genesis','Path']]
all_label = list(data_all_full['Genesis'])
all_file = list(data_all_full['Path'])
all_lat = list(df_all['Latitude'])
all_lon = list(df_all['Longitude'])

df_train.replace(False,'neg', inplace=True)
df_train.replace(True,'pos', inplace=True)
data_train_full=df_train[['Genesis','Path']]
train_label = list(data_train_full['Genesis'])
train_file = list(data_train_full['Path'])
train_lat = list(df_train['Latitude'])
train_lon = list(df_train['Longitude'])

df_test.replace(False,'neg', inplace=True)
df_test.replace(True,'pos', inplace=True)
data_test_full=df_test[['Genesis','Path']]
test_label = list(data_test_full['Genesis'])
test_file = list(data_test_full['Path'])
test_lat = list(df_test['Latitude'])
test_lon = list(df_test['Longitude'])

df_val.replace(False,'neg', inplace=True)
df_val.replace(True,'pos', inplace=True)
data_val_full=df_val[['Genesis','Path']]
val_label = list(data_val_full['Genesis'])
val_file = list(data_val_full['Path'])
val_lat = list(df_val['Latitude'])
val_lon = list(df_val['Longitude'])

print(data_test_full.head(5))
#
# function to create a segmentation mask for all TCG events
#
def create_segmentation(outdir,filepath,clat,clon,rscale):
    x = filepath.split('/')
    segfile = x[-1].replace("fnl","seg")
    print("Creating a segment file: ",segfile)
    #
    # creating NETCDF file for the segmentation output and write
    # it to outdir location.
    #
    f = Dataset(filepath)
    ny = f.dimensions['lat'].size
    nx = f.dimensions['lon'].size
    y = f.variables['lat']
    x = f.variables['lon']
    dx = x[1]-x[0]
    dy = y[1]-y[0]
    try: ncfile.close()  
    except: pass
    ncfile = Dataset(outdir+'/'+segfile,mode='w',format='NETCDF4_CLASSIC')
    #
    # create dimension
    #
    lat_dim = ncfile.createDimension('lat', ny)      # latitude axis
    lon_dim = ncfile.createDimension('lon', nx)      # longitude axis
    #time_dim = ncfile.createDimension('time', None) # unlimited axis (can be appended to).
    for dim in ncfile.dimensions.items():
        print(dim)
    #
    # create attribute
    #
    ncfile.title='Segementation 2D data for Unet model'
    ncfile.mask="Segementation values are 0 (outside) or 1 (inside) of TC genesis location"
    ncfile.storm_center = "Storm center location in lon/lat degree "+str(clon)+' , '+str(clat)
    ncfile.rscale = "Storm mask scale (in degree) for segmentation is "+str(rscale)
    #print(ncfile) 
    #
    # create variables now with _ values
    #
    lat = ncfile.createVariable('lat', np.float32, ('lat',))
    lat.units = 'degrees_north'
    lat.long_name = 'latitude'
    lon = ncfile.createVariable('lon', np.float32, ('lon',))
    lon.units = 'degrees_east'
    lon.long_name = 'longitude'
    #time = ncfile.createVariable('time', np.float64, ('time',))
    #time.units = 'hours since 1800-01-01'
    #time.long_name = 'time'
    mask = ncfile.createVariable('mask',np.float64,('lat','lon')) # note: unlimited dimension is leftmost
    mask.units = 'Dimensionless'
    mask.standard_name = 'Mask data <0/1> for a single time only'
    #print(mask)
    #
    # assign values for lat,lon,mask now with slice value filling.
    #
    lat[:] = y[:]
    lon[:] = x[:]
    print("dx, dy, clon, clat, scale: ",dx,dy,clon,clat,rscale)
    for j in range(ny):
      for i in range(nx):
        #radius = np.sqrt(((lat[ny-j-1]-clat)*dx)**2 + ((lon[i]-clon)*dy)**2)
        radius = np.sqrt(((lat[j]-clat)*dx)**2 + ((lon[i]-clon)*dy)**2)
        if radius <= rscale:
          mask[j,i] = 1
        else:
          mask[j,i] = 0
    f.close()
    ncfile.close()
    return 
#
# soft link all data into correspondng pos/neg dirs
#
def create_binary(path_in,path_out,tcg_label,tcg_path,tcg_lat,tcg_lon):
    outputdir_pos = path_out+'pos/'
    outputdir_neg = path_out+'neg/'
    outputdir_seg = path_out+'seg/'

    if os.path.exists(outputdir_pos):
        print("positive dir exists. Skip:",outputdir_pos)
    else:    
        print("creating a positive dir:",outputdir_pos)
        os.makedirs(outputdir_pos)    

    if os.path.exists(outputdir_neg):
        print("negative dir exists. Skip:",outputdir_neg)
    else:    
        print("creating a negative dir:",outputdir_neg)
        os.makedirs(outputdir_neg)    

    if os.path.exists(outputdir_seg):
        print("segmentation dir exists. Skip:",outputdir_seg)
    else:
        print("creating a segmentation dir:",outputdir_seg)
        os.makedirs(outputdir_seg)

    for i in range(len(tcg_label)):
        outfile = tcg_path[i].split("/")[-1] 
        infile = path_in+outfile
        #print(i,tcg_label[i],outfile)
        if tcg_label[i] == "neg":                             
            os.system('ln -sf ' + infile + ' ' + outputdir_neg+outfile)
            #os.symlink(path_in+outfile, outputdir_neg+outfile)
        else:
            os.system('ln -sf ' + infile + ' ' + outputdir_pos+outfile)
            create_segmentation(outputdir_seg,infile,tcg_lat[i],tcg_lon[i],rscale=5.)
    return

#create_binary(datain_path,dataout_path,all_label,all_file,all_lat,all_lon)
create_binary(datain_path,dataout_path+'/train/',train_label,train_file,train_lat,train_lon)     
create_binary(datain_path,dataout_path+'/test/',test_label,test_file,test_lat,test_lon)     
create_binary(datain_path,dataout_path+'/validation/',val_label,val_file,val_lat,val_lon)     
print("Done")

