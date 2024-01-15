#
# NOTE: This machine learning program is for predicting TC formation, using
#       input dataset in the NETCDF format. The program treats different
#       2D input fields as different channels of an image. This specific
#       program requires a set of 12 2D-variables (12-channel image) and
#       consists of three stages
#       - Stage 1: reading NETCDF input and generating (X,y) data with a
#                  given image sizes, which are then saved by pickle;
#       - Stage 2: import the saved pickle (X,y) pair and build a ResNet model
#                  with a given training/validation ratio, and then save
#                  the train model under tcg_ResNet.model.
#       - Stage 3: import the trained model from Stage 2, and make a list
#                  of prediction from normalized test data.
#
# INPUT: This Stage 3 script reads in the ResNet-trained model "tcg_ResNet.model"
#        that is generated from Step 2.
#
#        Remarks: Note that the input data for this script must be on the
#        same as in Step 1 with standard 19 vertical
#        levels 1000, 975, 950, 925, 900, 850, 800, 750, 700, 650, 600,
#        550, 500, 450, 400, 350, 300, 250, 200. Also, all field vars must
#        be resized to cover an area of 30x30 around the TC center for the
#        positive data cases.
#        Similar to Step 2, Step 3 needs to also have a large mem
#        allocation so that it can be run properly.
#
# OUTPUT: A list of probability forecasts with the same dimension as the
#        number of input 12-channel images.
#
# HIST: - 01, Nov 22: Created by CK
#       - 02, Nov 22: Modified to optimize it
#       - 05, Jun 23: Rechecked and added F1 score function for a list of model
#       - 12, Jun 23: Customized for ResNet from the ResNet functional model
#       - 18, Nov 23: re-designed for better check
#       - 14, Jan 24: streamlined further the workflow for job script submission
#
# AUTH: Chanh Kieu (Indiana University, Bloomington. Email: ckieu@iu.edu)
#
#==========================================================================
import cv2
import tensorflow as tf
import os
from tqdm import tqdm
import netCDF4
import numpy as np
import libtcg_netcdfloader as tcg_loader
import libtcg_utils as tcg_utils
import pickle
import sys
#
# Visualize the output of the training model (work for jupyter notebook only)
#
def view_history(histories):
    import matplotlib.pyplot as plt
    val_accuracy1 = histories[0]['val_binary_accuracy']
    accuracy1 = histories[0]['binary_accuracy']
    val_accuracy2 = histories[1]['val_binary_accuracy']
    accuracy2 = histories[1]['binary_accuracy']
    val_accuracy3 = histories[2]['val_binary_accuracy']
    accuracy3 = histories[2]['binary_accuracy']
    epochs = np.arange(len(val_accuracy1))
    plt.plot(epochs,val_accuracy1,'r',label="val binary_accuracy resnet20")
    plt.plot(epochs,accuracy1,'r--',label="train binary_accuracy resnet20")
    plt.plot(epochs,val_accuracy2,'b',label="val binary_accuracy resnet22")
    plt.plot(epochs,accuracy2,'b--',label="train binary_accuracy resnet22")
    plt.plot(epochs,val_accuracy3,'g',label="val binary_accuracy resnet40")
    plt.plot(epochs,accuracy3,'g--',label="train binary_accuracy resnet40")
    plt.legend()

    plt.figure()
    val_loss1 = histories[0]['val_loss']
    loss1 = histories[0]['loss']
    val_loss2 = histories[1]['val_loss']
    loss2 = histories[1]['loss']
    val_loss3 = histories[2]['val_loss']
    loss3 = histories[2]['loss']
    plt.plot(epochs,val_loss1,'r',label="val loss resnet20")
    plt.plot(epochs,loss1,'r--',label="train loss resnet20")
    plt.plot(epochs,val_loss2,'b',label="val loss resnet22")
    plt.plot(epochs,loss2,'b--',label="train loss resnet22")
    plt.plot(epochs,val_loss3,'g',label="val loss resnet40")
    plt.plot(epochs,loss3,'g--',label="train loss resnet40")
    plt.legend()
    plt.show()
#
# loop thru all best-saved ResNet-trained models and make a prediction. Note that prediction is applied one by one instead 
# of a batch input. 
#
def main(DATADIR="",bestmodels=[],test_sample=10):
    #
    # Test data with an input structure of pos/neg dirs
    #
    CATEGORIES = ["neg", "pos"]
    F1_performance = []
    for bestmodel in bestmodels:
        model = tf.keras.models.load_model(bestmodel)
        prediction_total = 0
        prediction_yes = 0
        prediction_history = []
        truth_history = []
        for category in CATEGORIES:
            path = os.path.join(DATADIR,category)
            for img in tqdm(os.listdir(path)):    
                try:
                    img_dir = DATADIR + '/' + category + '/' + img
                    print('Processing image:', img_dir)
                    indata, _ = tcg_loader.prepare12channels(img_dir,IMG_SIZE=32)
                    prediction = model.predict([indata])
                    print("TC formation prediction is",prediction,round(prediction[0][0]),CATEGORIES[round(prediction[0][0])])
                    prediction_history.append(prediction[0][0])
                    if round(prediction[0][0]) == 1:
                        prediction_yes += 1
                    if category == "pos":
                        truth_history.append(1)
                    else:
                        truth_history.append(0)
                    prediction_total += 1                        
                    if prediction_total > test_sample:
                        prediction_total = 0
                        break
                except Exception as e:
                    pass   
        #
        # Compute F1 score for each best model and save it
        #
        F1_performance.append([bestmodel,tcg_utils.F1_score(truth_history,prediction_history,1,0.5)]) 
    return F1_performance

#
# main fucntion
#
if __name__ == '__main__':
    n = len(sys.argv)
    print("Total arguments input are:", n)
    print("Name of Python script:", sys.argv[0])
    if n < 3:
       print("Need a forecast lead time and datapath to process...Stop")
       print("+ Example: tcg_ResNet_p3.py 00 /N/project/hurricane-deep-learning/data/ncep_extracted_binary_30x30/EP/")
       exit()
    leadtime = str(sys.argv[1])
    datadir = str(sys.argv[2]) + "/" + leadtime + "/"
    print("Forecast lead time to run is: ",leadtime)
    print("Datapath is: ",datadir)

    history_check = "no"
    if history_check == "yes":
        hist = open("tcg_histories_resnet.pickle","rb")
        histories = pickle.load(hist)
        view_history(histories)

    basemodels = ["tcg_ResNet20.model_ZZ","tcg_ResNet22.model_ZZ","tcg_ResNet40.model_ZZ"]
    bestmodels = []
    for model in basemodels:
        a=model.replace('ZZ', leadtime)
        bestmodels.append(a)
    print(bestmodels)

    performance = main(DATADIR=datadir,bestmodels=bestmodels,test_sample=50)
    print("Summary of the ResNet model performance for forecast lead time: ",leadtime)
    for i in range(len(bestmodels)):
        print("Model:",bestmodels[i]," --- F1, Recall, Presision:",np.round(performance[i][1],2))   
