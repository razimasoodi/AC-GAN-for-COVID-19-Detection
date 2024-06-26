# -*- coding: utf-8 -*-
"""ac_gan.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1JKZ6Ypw2ZriohE2jRqynXOFoAuGmBHOc
"""

import  numpy as np
import pandas as pd
import os
import time
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Input, Embedding, SimpleRNN, Dense, Activation,
                          TimeDistributed,Reshape)

from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow import keras
from tensorflow.keras.layers import Dense
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
import cv2 as cv
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D , MaxPool2D , Flatten , Dropout,Reshape, Concatenate, Conv2DTranspose, UpSampling2D
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report,confusion_matrix
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input,decode_predictions
import matplotlib.pyplot as plt
from tensorflow.keras.layers import BatchNormalization
from numpy.random import randn
import matplotlib.pyplot as plt
from matplotlib import pyplot
from keras.layers import UpSampling2D
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import RocCurveDisplay

x_test=np.load('/content/drive/MyDrive/x_test.npy')
x_train=np.load('/content/drive/MyDrive/x_train.npy')
y_test=np.load('/content/drive/MyDrive/y_test.npy')
y_train=np.load('/content/drive/MyDrive/y_train.npy')
ConvNet=tf.keras.models.load_model('/content/drive/MyDrive/generated/ConvNet3')

def discriminator_func(input_shape=(128,128,3), n_classes=2):
    init = tf.random_normal_initializer(stddev=0.02)
    input_image = Input(shape=input_shape)
    dis = Conv2D(32,(3,3),strides=(2,2), padding='same', kernel_initializer=init)(input_image)
    dis = tf.keras.layers.LeakyReLU(alpha=0.2)(dis)
    dis = Dropout(0.5)(dis)

    dis = Conv2D(64, (3,3), padding='same', kernel_initializer=init)(dis)
    dis = BatchNormalization()(dis)
    dis = tf.keras.layers.LeakyReLU(alpha=0.2)(dis)
    dis = Dropout(0.5)(dis)

    dis = Conv2D(128, (3,3),strides=(2,2), padding='same', kernel_initializer=init)(dis)
    dis = BatchNormalization()(dis)
    dis = tf.keras.layers.LeakyReLU(alpha=0.2)(dis)
    dis = Dropout(0.5)(dis)

    dis = Conv2D(256, (3,3), padding='same', kernel_initializer=init)(dis)
    dis = BatchNormalization()(dis)
    dis = tf.keras.layers.LeakyReLU(alpha=0.2)(dis)
    dis = Dropout(0.5)(dis)

    dis = Flatten()(dis)

    out1 = Dense(1, activation='sigmoid')(dis)
    out2 = Dense(n_classes, activation='softmax')(dis)

    model = tf.keras.models.Model(input_image, [out1, out2])

    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss=['binary_crossentropy', 'sparse_categorical_crossentropy'], optimizer=opt)
    return model

def generator_func(latent_dim, n_classes=2):

    in_label = Input(shape=(1,), dtype='int32')
    n_nodes = 32 * 32 * 3 
    li =  tf.keras.layers.Embedding( n_classes, 50)(in_label)
    li = Dense(n_nodes)(li)
    li = Reshape((32, 32, 3))(li)
  
    in_lat = Input(shape=(latent_dim,))

    n_nodes = 384 * 32 * 32
    gen = Dense(n_nodes)(in_lat)
    gen = BatchNormalization(momentum=0.9)(gen)
    gen = Activation('relu')(gen)
    
    gen = Reshape((32, 32, 384))(gen)
    gen = Dropout(0.4)(gen)
    
    merge = Concatenate()([gen, li])
    
    gen = Conv2DTranspose(128,(5,5), padding='same')(merge)
    gen = BatchNormalization(momentum=0.9)(gen)
    gen = Activation('relu')(gen)
    gen = UpSampling2D()(gen)

    gen = Conv2DTranspose(64,(5,5),padding='same')(gen)
    gen = BatchNormalization(momentum=0.9)(gen)
    gen = Activation('relu')(gen)
    gen = UpSampling2D()(gen)

    gen = Conv2DTranspose(32,(5,5),padding='same')(gen)
    gen = BatchNormalization(momentum=0.9)(gen)
    gen = Activation('relu')(gen)

    gen = Conv2DTranspose(3, (3,3), padding='same')(gen)
    out_layer = Activation('sigmoid')(gen)

    model = tf.keras.models.Model([in_lat, in_label], out_layer)
    return model

def gan_func(gen, dis):
    for layer in dis.layers:
        if not isinstance(layer, BatchNormalization):
            layer.trainable = False
    gan_out = dis(gen.output)
    model = tf.keras.models.Model(gen.input, gan_out)
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss=['binary_crossentropy', 'sparse_categorical_crossentropy'], optimizer=opt)
    return model

def generate_real_image(dataset, n_samples):
    X = np.expand_dims(dataset[0],axis=3)
    k = np.random.randint(0, dataset[0].shape[0],n_samples)
    X,labels = dataset[0][k],  dataset[1][k]
    y = np.ones((n_samples,1))
    return [X,labels], y

def generate_noise(latent_dim, n_samples, n_classes=2):
    x_input = randn(latent_dim * n_samples)
    z_input = x_input.reshape(n_samples, latent_dim)
    labels = np.random.randint(0, n_classes, n_samples)
    return [z_input, labels]

def generate_fake_image(generator, latent_dim, n_samples):
    z_input, labels_input = generate_noise(latent_dim, n_samples,n_classes=2)
    images = generator.predict([z_input, labels_input])
    y = np.zeros((n_samples, 1))
    return [images, labels_input], y

def plot(step, g_model, latent_dim, n_samples=100):
    [image,label], y = generate_fake_image(g_model, latent_dim, n_samples)
    image = (image + 1) / 2.0
    for i in range(16):
        plt.subplot(4, 4, 1 + i)
        plt.axis('off')
        plt.imshow(image[i, :, :, 0], cmap='gray_r')
   # plt.show()   
    return image, label

def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=20, n_batch=128):
    generated_image=[]
    generated_label=[]
    bat_per_epo = int(dataset[0].shape[0] / n_batch)
    steps = bat_per_epo * n_epochs
    half_batch = int(n_batch / 2)
    
    for i in range(steps):
        
        [X_real, labels_real], y_real = generate_real_image(dataset, half_batch)     
        _,d_r1,d_r2 = d_model.train_on_batch(X_real, [y_real, labels_real])      
        [X_fake, labels_fake], y_fake = generate_fake_image(g_model, latent_dim, half_batch)      
        _,d_f,d_f2 = d_model.train_on_batch(X_fake, [y_fake, labels_fake])      
        [z_input, z_labels] = generate_noise(latent_dim, n_batch)      
        y_gan = np.ones((n_batch, 1))    
        _,g_1,g_2 = gan_model.train_on_batch([z_input, z_labels], [y_gan, z_labels])
        print(i+1)

        if (i+1) % (bat_per_epo *10) == 0 :
            image,label=plot(i, g_model, latent_dim)
            generated_image.append(image)
            generated_label.append(label)
        
    return generated_image, generated_label

latent_dim = 100
dataset = [x_train,y_train]
discriminator = discriminator_func()
generator = generator_func(latent_dim)
gan_model = gan_func(generator, discriminator)
generated_image, generated_label =train(generator, discriminator, gan_model, dataset, latent_dim)

#np.save('/content/drive/MyDrive/generated/generated_image10.npy', np.array(generated_image))
#np.save('/content/drive/MyDrive/generated/generated_label10.npy', np.array(generated_label))
#x_test_generated=np.load('/content/drive/MyDrive/generated/generated_image1.npy')
#y_test_generated=np.load('/content/drive/MyDrive/generated/generated_label1.npy')

def preprocessing_generated_image(x_test_generated,y_test_generated):
    xtg=x_test_generated.reshape(x_test_generated.shape[0]*x_test_generated.shape[1],128,128,3)
    ytg=y_test_generated.reshape(y_test_generated.shape[0]*y_test_generated.shape[1],)
    p=int(y_test.shape[0]*0.2)
    q=np.random.choice(np.arange(y_test.shape[0]-1), p, replace=False)
    x=[]
    y=[]
    for i in q:
        x.append(x_test[i])
        y.append(y_test[i])
    xtg_20=np.concatenate((xtg,np.array(x)),axis=0)
    ytg_20=np.concatenate((ytg,np.array(y)),axis=0)  
    return xtg,ytg,xtg_20,ytg_20

def confusion_Roc(model,samples,label):
    predictions = model.predict_classes(samples)
    print(classification_report(label, predictions, target_names = ['Non-COVID(Class 0)','COVID (Class 1)']))
    predict_label = model.predict(samples)  
    fpr, tpr, thresholds = roc_curve(label,predict_label[ : ,1])
    roc_auc = auc(fpr, tpr)
    display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name='example estimator')
    display.plot()  
    plt.show()

#x_test_generated,y_test_generated
#np.array(generated_image),np.array(generated_label)
xtg,ytg,xtg_20,ytg_20 = preprocessing_generated_image(np.array(generated_image),np.array(generated_label))
xtg_all=np.concatenate((xtg,x_test),axis=0)
ytg_all=np.concatenate((ytg,y_test),axis=0)  
eva_test_g=ConvNet.evaluate(xtg,ytg, verbose=0)
confusion_Roc(ConvNet,xtg,ytg)
print('The accuracy of generated images is : ',eva_test_g[1]*100,'%')
eva_test_g20=ConvNet.evaluate(xtg_20,ytg_20, verbose=0)
confusion_Roc(ConvNet,xtg_20,ytg_20)
print('The accuracy of 20% of the original test images along with the generated images is : ',eva_test_g20[1]*100,'%')
eva_test_g_all=ConvNet.evaluate(xtg_all,ytg_all, verbose=0)
confusion_Roc(ConvNet,xtg_all,ytg_all)
print('The accuracy of all of the original test images along with the generated images is : ',eva_test_g_all[1]*100,'%')

np.save('/content/drive/MyDrive/generated/generated_image0.npy', np.array(generated_image))
np.save('/content/drive/MyDrive/generated/generated_label0.npy', np.array(generated_label))

