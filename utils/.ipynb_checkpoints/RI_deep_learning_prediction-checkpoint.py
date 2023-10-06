import pandas as pd
import math
import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn import preprocessing, svm, neighbors
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
#
# reading input data and make prediction from pre-trained models using all 24-h
#
num_predictors = 15
sequence_length = 5
df = pd.read_csv('/N/u/ckieu/Carbonate/python/DOKSURI05W_Master.csv')
df.drop(['Time','Basin','OHC','OHC(+6h)','OHC(+12h)','OHC(+18h)','OHC(+24h)'], axis=1, inplace=True)
df.replace('?',-99999, inplace=True)
x_fcst = np.array(df.drop(['Storm','class'],axis=1))
y_true = np.array(df['class'])
x_tlag = x_fcst.reshape((-1,sequence_length,num_predictors))
print('External input SHIP data length is: ',len(x_fcst))
#print(x_fcst[0])
print(y_true)
#
# loading three different pre-trained models using all 24-h input SHIP data
#
model_RNN_1d = keras.models.load_model("RI_model_RNN_24h.keras")
model_logistics_1d = keras.models.load_model("RI_model_logistics_24h.keras")
model_GRU_1d = keras.models.load_model("RI_model_GRU_24h.keras")
fcst_logistics_1d = model_logistics_1d.predict(x_fcst)
fcst_GRU_1d = model_GRU_1d.predict(x_tlag)
fcst_RNN_1d = model_RNN_1d.predict(x_tlag)
for i in range(len(x_fcst)):
    print(f"Logistic, RNN, GRU probability predictions: {float(fcst_logistics_1d[i]):.3f},{float(fcst_RNN_1d[i]):.3f},{float(fcst_GRU_1d[i]):.3f}")
#
# loading three different pre-trained models using all 00-h input SHIP data
#
x_0d = np.array(df[['lat','lon','MaxWind','RMW','MIN_SLP','SHR_MAG','SHR_HDG','STM_SPD','STM_HDG',
                   'SST','TPW','LAND','850TANG','850VORT','200DVRG']])
x_0d_extent = np.expand_dims(x_0d, axis=1)
print(x_0d.shape,x_0d_extent.shape)
model_RNN_0d = keras.models.load_model("RI_model_RNN_00h.keras")
model_logistics_0d = keras.models.load_model("RI_model_logistics_00h.keras")
model_GRU_0d = keras.models.load_model("RI_model_GRU_00h.keras")
fcst_logistics_0d = model_logistics_0d.predict(x_0d)
fcst_GRU_0d = model_GRU_0d.predict(x_0d_extent)
fcst_RNN_0d = model_RNN_0d.predict(x_0d_extent)
for i in range(len(x_fcst)):
    print(f"Logistic, RNN, GRU probability predictions 00h: {float(fcst_logistics_0d[i]):.3f},{float(fcst_RNN_0d[i]):.3f},{float(fcst_GRU_0d[i]):.3f}")    

