#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import keras as keras
from keras import Sequential
from keras.optimizers import Adam, RMSprop
from keras.layers import Activation, Dense, Flatten, Dropout
from keras.layers import Conv2D, MaxPooling2D, ConvRecurrent2D
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as k
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.image as mpimg
import cv2
import os
import sys
import PIL
from warnings import filterwarnings
filterwarnings('ignore')


# In[2]:


"""train -> no of categorical files 
\n\nvalidation-> no of categorical files"""


# #  model creation

# In[27]:


class detectmodel(object):
  
    def __init__(self,input_shape):
        self.input_shape=input_shape
        
        
    def build_dataset(self):
        self.train_dir=input("input the path of training directory: ")
        self.test_dir= input("input the path of test directory: ")
        training_datagen = ImageDataGenerator(
          rescale = 1./255,
          rotation_range=40,
          width_shift_range=0.2,
          height_shift_range=0.2,
          shear_range=0.2,
          zoom_range=0.2,
          horizontal_flip=True,
          fill_mode='nearest')
        validation_datagen = ImageDataGenerator(rescale = 1./255)
        global train_generator 
        train_generator = training_datagen.flow_from_directory(self.train_dir,target_size=(150,150),class_mode='categorical')
        global validation_generator
        validation_generator = validation_datagen.flow_from_directory(self.test_dir,target_size=(150,150),class_mode='categorical')    
    
    def create_model(self,z,loss,opt,metrics):
        #self.input_shape=input_shape
        self.z=z
        self.loss=loss
        self.opt=opt
        self.metrics=metrics
        global model
        model=Sequential()
        print("input shape= ",input_shape,'\n','output nodes= ',self.z)
        model.add(Conv2D(32,(3,3),input_shape=input_shape, activation='relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Conv2D(32,(3,3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Conv2D(64,(3,3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Conv2D(64,(3,3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Conv2D(128,(3,3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Flatten())
        model.add(Dense(128,activation='elu'))
        model.add(Dropout(0.4))
        model.add(Dense(self.z,activation='sigmoid'))
        model.compile(optimizer=self.opt,loss=self.loss,metrics=[self.metrics])
        print('succesfully created and compiled')
        model.summary()
        
    def Fit(self,xtrain,ytrain,epochs):
        self.epochs=epochs
        self.xtrain=xtrain
        self.ytrain=ytrain
        ##model.compile(optimizer=self.opt,loss=self.loss,metrics=[self.metrics])
        model.fit(self.xtrain,self.ytrain,validation_split=0.2,epochs=self.epochs)
        
        
    def selffit(self,epochs):
        self.epochs=epochs
        model.fit_generator(train_generator, epochs=self.epochs, validation_data = validation_generator, verbose = 1)
        
        
        
    
    
if "__main__":
    print("enter input_shape as tuple during object initialization")
    print("enter output node, loss, optimizer and metrics in method create_model")
    print("enter X and Y training set along with no of epochs in Fit method")


# In[28]:


isp=(150,150,3)
mm=detectmodel(isp)


# In[32]:


mm.build_dataset()


# In[8]:


input_shape=(150,150,3)


# In[30]:


mm.create_model(2,"categorical_crossentropy","RMSprop",'accuracy')


# In[31]:


history=mm.selffit(50)


# In[ ]:




