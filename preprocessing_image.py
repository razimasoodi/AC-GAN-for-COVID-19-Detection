#!/usr/bin/env python
# coding: utf-8

# In[1]:


import  numpy as np
import pandas as pd
import os
import time
import tensorflow as tf
from keras.models import Sequential
from keras.layers import (Input, Embedding, SimpleRNN, Dense, Activation,
                          TimeDistributed)

from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow import keras
from keras.layers import Dense
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
import os
import cv2 as cv
from tensorflow.keras.models import Sequential
from keras.layers import Dense, Conv2D , MaxPool2D , Flatten , Dropout 
from keras.optimizers import Adam
from sklearn.metrics import classification_report,confusion_matrix
from keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input,decode_predictions


# In[2]:


def read_data(name):
    DATADIR=name
    path=os.path.join(DATADIR)
    x=[]
    img_size=128
    for img in os.listdir(path):
        img_array=cv.imread(os.path.join(path,img))
        if img_array is not None:
            new_image=cv.resize(img_array,(img_size,img_size))
            x.append(new_image)
    X=np.array(x)         
    return X        


# In[3]:


pathes=['COVID_test','COVID_train','Non-COVID_test','Non-COVID_train']
samples=[]
labels=[]
for i in pathes:
    x=read_data(i)
    samples.append(x)
    if i=='COVID_test' or i=='COVID_train' :
        y=np.ones(len(x))
    else:
        y=np.zeros(len(x))
    labels.append(y)     
x_train=np.concatenate((samples[1],samples[3]), axis=0)
x_test=np.concatenate((samples[0],samples[2]), axis=0)
y_train=np.concatenate((labels[1],labels[3]), axis=0)
y_test=np.concatenate((labels[0],labels[2]), axis=0)    


# In[19]:


np.save('x_train.npy', x_train)
np.save('x_test.npy', x_test)
np.save('y_train.npy', y_train)
np.save('y_test.npy', y_test)

