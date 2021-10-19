# -*- coding: utf-8 -*-
"""
Created on Mon May 17 10:42:22 2021

@author: ali.raza
"""

import socket
import pickle
import threading
import time
from mlsocket import MLSocket
import pygad
import pygad.nn
import pygad.gann
import numpy

import pygad
import pygad.nn
import pygad.gann
import numpy

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from keras.utils.np_utils import to_categorical
from sklearn.utils import class_weight
import warnings
from keras.callbacks import TensorBoard
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Model
from keras.models import Sequential
from keras.layers import Convolution1D, ZeroPadding1D, MaxPooling1D, BatchNormalization, Activation, Dropout, Flatten, Dense
from keras.layers import Conv1D, Dense, MaxPool1D, Flatten, Input
import tensorflow as tf
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from tensorflow.keras import datasets, layers, models
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.preprocessing import label_binarize
from tensorflow.keras.models import Sequential, load_model
import cv2
import h5py
from sklearn.model_selection import train_test_split
#-----------------------------------------Datan Preparation--------------------------------------------------------------------------
warnings.filterwarnings('ignore')
train_df=pd.read_csv('mitbih_train.csv',header=None)
test_df=pd.read_csv('mitbih_test.csv',header=None)

df_1=train_df[train_df[188]==1]
df_2=train_df[train_df[188]==2]
df_3=train_df[train_df[188]==3]
df_4=train_df[train_df[188]==4]
df_0=(train_df[train_df[188]==0]).sample(n=20000,random_state=42)

df_1_upsample=resample(df_1,replace=True,n_samples=20000,random_state=123)
df_2_upsample=resample(df_2,replace=True,n_samples=20000,random_state=124)
df_3_upsample=resample(df_3,replace=True,n_samples=20000,random_state=125)
df_4_upsample=resample(df_4,replace=True,n_samples=20000,random_state=126)

df=pd.concat([df_0,df_1_upsample,df_2_upsample,df_3_upsample,df_4_upsample])

dft_1=test_df[test_df[188]==1]
dft_2=test_df[test_df[188]==2]
dft_3=test_df[test_df[188]==3]
dft_4=test_df[test_df[188]==4]
dft_0=(test_df[test_df[188]==0]).sample(n=10000,random_state=42)

dft_1_upsample=resample(dft_1,replace=True,n_samples=10000,random_state=123)
dft_2_upsample=resample(dft_2,replace=True,n_samples=10000,random_state=124)
dft_3_upsample=resample(dft_3,replace=True,n_samples=10000,random_state=125)
dft_4_upsample=resample(dft_4,replace=True,n_samples=10000,random_state=126)

dft=pd.concat([dft_0,dft_1_upsample,dft_2_upsample,dft_3_upsample,dft_4_upsample])

data=pd.concat([dft,df])
y=data.iloc[:,-1]
X=data.iloc[:,:-1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


def add_gaussian_noise(signal):
    noise=np.random.normal(0,0.01,187)
    out=signal+noise
    return out
target_train=y_train
target_test=y_test

y_train=to_categorical(target_train)
y_test=to_categorical(target_test)

X_train=X_train
X_test=X_test
X_train_noise=np.array(X_train)
X_test_noise=np.array(X_test)


plt.plot(X_train[10])
plt.show()
for i in range(len(X_train_noise)):
    X_train_noise[i,:187]= add_gaussian_noise(X_train_noise[i,:187])
for i in range(len(X_test_noise)):
    X_test_noise[i,:187]= add_gaussian_noise(X_test_noise[i,:187])
    
X_train_noise= X_train_noise.reshape(len(X_train_noise), X_train_noise.shape[1],1)
X_train= X_train.reshape(len(X_train), X_train.shape[1],1)
X_test = X_test.reshape(len(X_test), X_test.shape[1],1)
X_test_noise= X_test_noise.reshape(len(X_test_noise), X_test_noise.shape[1],1)


X_train=np.float32(X_train)
X_train_noise=np.float32(X_train_noise)
X_test=np.float32(X_test)
X_test_noise=np.float32(X_test_noise)

plt.plot(X_train[10])
plt.plot(X_train_noise[10])
plt.show()

def featureNormalize(dataset):
    mu = np.mean(dataset,axis=0)
    sigma = np.std(dataset,axis=0)
    return (dataset-mu)/sigma
#normalize mean ok
X_train_noise=(X_train_noise-X_train_noise.mean())/X_train_noise.std()
X_test_noise=(X_test_noise-X_test_noise.mean())/X_test_noise.std()


plt.show()
s= MLSocket() 
HOST = '172.23.33.7'
PORT = 65432
s.connect((HOST, PORT)) # Connect to the port and host
autoencoder = s.recv(1024)
def scheduler(epoch, lr):
       if epoch < 40:
        return lr
       else:
        return lr * tf.math.exp(-0.1)

callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
autoencoder.compile(optimizer='RMSprop', loss='mse')
autoencoder.summary()

#-----------------------------------------Model Fit--------------------------------------------------------------------------
trained_autoencoder=autoencoder.fit(X_train_noise,X_train,epochs=50,
                    batch_size=100,
                    shuffle=True,
                    validation_data=(X_test_noise,X_test),
                    callbacks=[TensorBoard(log_dir='/tmp/autoencoder'),callback])
#-----------------------------------------Print autoencoder metrics------------------------------------
x_train_pred = autoencoder.predict(X_test_noise)
train_mae_loss = np.mean(np.abs(x_train_pred - X_test), axis=1)
#Print     
plt.hist(train_mae_loss, bins=50)
plt.xlabel("Train MAE loss")
plt.ylabel("No of samples")
plt.show()
loss=trained_autoencoder.history['loss']
val_loss=trained_autoencoder.history['val_loss']
plt.plot( loss, label='loss')
plt.plot( val_loss, label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
plt.figure()


s.send(autoencoder)
#-----------------------------------------Model Fit--------------------------------------------------------------------------

def scheduler2(epoch, lr):
       if epoch < 90:
        return lr
       else:
        return lr * tf.math.exp(-0.1)

callback = tf.keras.callbacks.LearningRateScheduler(scheduler2)

s= MLSocket() 
s.connect((HOST, PORT)) # Connect to the port and host
model = s.recv(1024)
print("received classifier")    
classifier=model

for l1,l2 in zip(classifier.layers[:5],autoencoder.layers[0:5]):
    l1.set_weights(l2.get_weights())
for layer in classifier.layers[0:5]:
    layer.trainable = False

opt = keras.optimizers.RMSprop(learning_rate=0.001)
classifier.compile(optimizer=opt,
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

classifier.summary()
input_data = keras.Input(shape=(X_train_noise.shape[1], 1))
classify_train = classifier.fit(X_train_noise,y_train,shuffle=True, batch_size=100,epochs=150,verbose=1,
                                validation_split=0.1,callbacks=[TensorBoard(log_dir='/tmp/autoencoder'),callback])
                    

s.send(classifier)
