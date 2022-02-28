# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 16:45:06 2021

@author: ali.raza
"""



import socket
import pickle
import threading
import time

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




def add_gaussian_noise(signal):
    noise=np.random.normal(0,0.01,187)
    out=signal+noise
    return out
target_train=df.iloc[:,-1]
target_test=dft.iloc[:,-1]

y_train=to_categorical(target_train)
y_test=to_categorical(target_test)

X_train=df.iloc[:,:-1].values
X_test=dft.iloc[:,:-1].values
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
plt.show()
plt.plot(X_train_noise[10],color='red')
plt.show()
plt.plot(X_train[300])
plt.show()
plt.plot(X_train_noise[300],color='red')
plt.show()
#normalize mean ok
X_train_noise=(X_train_noise-X_train_noise.mean())/X_train_noise.std()
X_test_noise=(X_test_noise-X_test_noise.mean())/X_test_noise.std()


plt.show()
#-----------------------------------------Autoencoder Design--------------------------------------------------------------------------
def encoder(input_data):
    #encoder
    
    x = layers.Conv1D(64, kernel_size=3,activation='relu', name='input')(input_data)
    x = layers.MaxPooling1D(2, padding='same')(x)
    x = layers.Conv1D(64, kernel_size=3, activation='relu', padding='same')(x)
    x = layers.MaxPooling1D(2, padding='same')(x)
    x = layers.Conv1D(4, kernel_size=3, activation='relu', padding='same')(x)
    encoded = layers.MaxPooling1D(2,padding='same')(x)
    return encoded
def decoder(encoded):    
    #decoder
    x = layers.Conv1D(4,  kernel_size=3, activation='relu', padding='same')(encoded)
    x = layers.UpSampling1D(2)(x)
    x = layers.Conv1D(64, kernel_size=3, activation='relu', padding='same')(x)
    x = layers.UpSampling1D(2)(x)
    x = layers.Conv1D(64, kernel_size=3, activation='relu' )(x)
    x = layers.UpSampling1D(2)(x)
    decoded = layers.Conv1D(1, 3, activation='relu',padding='same')(x)
    return decoded
#----------------------------------------------Model Compile---------------------------------------------------------------------
input_data = keras.Input(shape=(X_train_noise.shape[1], 1))
autoencoder = keras.Model(input_data, decoder(encoder(input_data)))
opt = keras.optimizers.RMSprop(learning_rate=0.00001)
def scheduler(epoch, lr):
       if epoch < 40:
        return lr
       else:
        return lr * tf.math.exp(-0.1)
callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
autoencoder.compile(optimizer='RMSprop', loss='mse')
autoencoder.summary()



#------------------------------------------classifier-------------------------------------------------
def fc(enco):
    x = layers.Conv1D(64, kernel_size=3, activation='relu', padding='same')(enco)
    
    x = layers.Conv1D(64, kernel_size=1, activation='relu', padding='same')(x)
    
    x=layers.Flatten()(x)
    x=layers.Dense(118, activation='relu')(x)
    
    x=layers.Dense(5,  activation='softmax')(x)
    
    return x
encode = encoder(input_data)
classifier=Model(input_data,fc(encode))
#-----------------------------------------Assign weights------------------------------------------------

from mlsocket import MLSocket
from keras.models import Sequential
from keras.layers import Dense
from sklearn import svm
import numpy as np

ip = 'xxx.xx.xx.xx' #write the IP of host (server)
port = xxxx #Port number of host opened for communication.

number_clients=x #reaplace  x with number of edge devices

global count
count=0
global counter1
counter1=0
global counter2
counter2=0
global ave_weights
global ave_weights_auto
import socket 
from threading import Thread 
import threading 
from _thread import *
import time

s=MLSocket()
try:
    s.bind((ip, port))
except s.error as e:
    print(e)
print("Listening to clients")
s.listen()  
# initial weights of model make to zer0.  
ave_weights=classifier.get_weights()
ave_weights=[i * 0 for i in ave_weights]

ave_weights_auto=autoencoder.get_weights()
ave_weights_auto=[i * 0 for i in ave_weights_auto]

weight_scalling_factor=100000/120000  # repalce n_i and , n with number of samples of a given edge and total number of samples.


def scale_model_weights(weight, scalar):
    '''function for scaling a models weights'''
    weight_final = []
    steps = len(weight)
    for i in range(steps):
        weight_final.append(scalar * weight[i])
    return weight_final



def sum_scaled_weights(scaled_weight_list):
    '''Return the sum of the listed scaled weights. The is equivalent to scaled avg of the weights'''
    avg_grad = list()
    #get the average grad accross all client gradients
    for grad_list_tuple in zip(*scaled_weight_list):
        layer_mean = tf.math.reduce_sum(grad_list_tuple, axis=0)
        avg_grad.append(layer_mean)
        
    return avg_grad
def classified_class(target_class):
    if target_class==0:
        print('Your ECG is classified as: Non-ecotic beats (normal beat) ')
    if target_class==1:
        print('Your ECG is classified as: Supraventricular ectopic beats ')
    if target_class==2:
        print('Your ECG is classified as: Ventricular ectopic beats ')
    if target_class==3:
        print('Your ECG is classified as: Fusion Beats ')
    if target_class==4:
        print('Your ECG is classified as: Unknown Beats ')
def Grad_cam(model, input_test,sample_number,autoencoder):
    
    array = np.array(input_test[sample_number])
    
    # We add a dimension to transform our array into a "batch"
  
    array = np.expand_dims(array, axis=0)
     
    predict = model.predict(array)
    target_class = np.argmax(predict[0])
    classified_class(target_class)
    
    last_conv = model.get_layer('conv1d_7') #last_conv
    grad_model = tf.keras.models.Model([model.inputs], [last_conv.output, model.output])

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(array) #get activations maps + predictions from last conv layer
        loss = predictions[:, target_class] # the variable loss gets the probability of belonging to the defined class (the predicted class on the model output)

    output = conv_outputs[0] #activations maps from last conv layer
    grads = tape.gradient(loss, conv_outputs) #function to obtain gradients from last conv layer

  
    pooled_grad= tf.reduce_mean(grads, axis=(0, 1))
    conv_outputs=conv_outputs.numpy()
    pooled_grad = pooled_grad.numpy()
    for i in range(pooled_grad.shape[-1]):
        conv_outputs[:, :, i] *= pooled_grad[i]
    heatmap = np.mean(conv_outputs, axis=-1)
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)

   
    #Upsample the small_heatmap into a big_heatmap with cv2:
    big_heatmap = cv2.resize(heatmap, dsize=(188, 100), 
                         interpolation=cv2.INTER_CUBIC)
    reconstructed_ecg = autoencoder.predict(array )
    plt.imshow(big_heatmap, cmap='rainbow')
    plt.colorbar()
    plt.plot(reconstructed_ecg[0]*80,color='black')


    plt.xlim(0,188)
    plt.ylim(0,100)
    plt.title('Grad_CAM Diagram of Classified ECG Signal')
    plt.savefig("draw_on_image_03.png")
    plt.show()
    return big_heatmap

# Multithreaded Python server : TCP Server Socket Thread Pool
def ClientThread_send(conn,address,weights):
    global count
    global counter1
    global counter2
    if counter1<number_clients:
        conn.send(autoencoder)
        counter1=counter1+1
        weight_scalling_factor=n_i/n  # repalce n_i and , n with number of samples of a given edge and total number of samples.
        print("Autoencoder Sent to :",address)
        print("waiting for weights")
        model_recv=conn.recv(1024)
        print("weights received from autoencoder:",address)
        global count
        global counter
        global ave_weights
        global ave_weights_auto
        scaled_local_weight_list = list()
        scaled_weights = scale_model_weights(model_recv.get_weights(), weight_scalling_factor)
        scaled_local_weight_list.append(scaled_weights)
        ave_weights_auto = sum_scaled_weights(scaled_local_weight_list)
        counter2=counter2+1
        print("counter 2 value:",counter2)
    
    else:
        while counter2<number_clients:
            time.sleep(3)
            print("Waiting for the autoencoder to complete the aggregation...")
            
        autoencoder.set_weights(ave_weights_auto) 
        x_train_pred = autoencoder.predict(X_test_noise)
        train_mae_loss = np.mean(np.abs(x_train_pred - X_test), axis=1)
        #Print     
        plt.hist(train_mae_loss, bins=50)
        plt.xlabel("Mean Absolute Error (MAE) of Global Autoencoder")
        plt.ylabel("No of samples")
        plt.show()
        
        
        for l1,l2 in zip(classifier.layers[:5],autoencoder.layers[0:5]):
            l1.set_weights(l2.get_weights())
        for layer in classifier.layers[0:5]:
            layer.trainable = False

        opt = keras.optimizers.RMSprop(learning_rate=0.001)
        classifier.compile(optimizer=opt,
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])  
        conn.send(classifier)
        weight_scalling_factor=100000/100000
        print("Classifier Sent to :",address)
        print("waiting for weights of classifier")
        model_recv=conn.recv(1024)
        print("weights received from classifier:",address)
        global count
        global ave_weights
        scaled_local_weight_list = list()
        scaled_weights = scale_model_weights(model_recv.get_weights(), weight_scalling_factor)
        scaled_local_weight_list.append(scaled_weights)
    
        ave_weights = sum_scaled_weights(scaled_local_weight_list)
    
        count=count+1
        conn.close()
        if count==number_clients:
            classifier.set_weights(ave_weights)
            autoencoder.set_weights(ave_weights_auto)
            x_train_pred = autoencoder.predict(X_test_noise)
            train_mae_loss = np.mean(np.abs(x_train_pred - X_test), axis=1)
            #Print     
            plt.hist(train_mae_loss, bins=50)
            plt.xlabel("Mean Absolute Error (MAE) of Global Autoencoder")
            plt.ylabel("No of samples")
            plt.show()
            print("Predicting....")
            x_train_pred = classifier.predict(X_test_noise)
            train_mae_loss = np.mean(np.abs(x_train_pred - y_test), axis=1)
     #Print     
            plt.hist(train_mae_loss, bins=50)
            plt.xlabel("Mean Absolute Error (MAE) of Global Calssifier")
            plt.ylabel("No of samples")
            plt.show()
            y_pred = classifier.predict(X_test_noise)
            #y_pred.reshape(250000,1)
            y_pred=np.argmax(y_pred, axis=1)
            print(classification_report(target_test, y_pred))
            target_names = list("NSVFQ")
            clf_report = classification_report(target_test, y_pred, target_names=target_names,output_dict=True)
            
            
            plt.show(sns.heatmap(pd.DataFrame(clf_report).iloc[:-1, :].T, annot=True))
            
            heat=Grad_cam(classifier, X_test_noise,1,autoencoder)
            heat=Grad_cam(classifier, X_test_noise,3020,autoencoder)
            heat=Grad_cam(classifier, X_test_noise,1103,autoencoder)
            heat=Grad_cam(classifier, X_test_noise,648,autoencoder)
            heat=Grad_cam(classifier, X_test_noise,5,autoencoder)
            heat=Grad_cam(classifier, X_test_noise,320,autoencoder)
            heat=Grad_cam(classifier, X_test_noise,2034,autoencoder)
            heat=Grad_cam(classifier, X_test_noise,68,autoencoder)
            heat=Grad_cam(classifier, X_test_noise,175,autoencoder)
            heat=Grad_cam(classifier, X_test_noise,4000,autoencoder)
            heat=Grad_cam(classifier, X_test_noise,2303,autoencoder)
            heat=Grad_cam(classifier, X_test_noise,667,autoencoder)



    
while count<number_clients:
    conn, address = s.accept()
    start_new_thread(ClientThread_send,(conn,address,ave_weights))   
    print("count value", count)



    
    
