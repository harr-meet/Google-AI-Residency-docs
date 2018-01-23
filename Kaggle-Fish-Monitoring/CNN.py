
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
import os
import cv2
import time

from keras.models import Sequential
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.convolutional import ZeroPadding2D, AveragePooling2D
from keras.layers.core import Activation, Dense, Dropout, Flatten
from keras.optimizers import SGD
from keras.utils import np_utils

from keras import backend as K
K.set_image_dim_ordering('tf')


# In[2]:

folders = ['ALB','BET','DOL','LAG','NoF','OTHER','SHARK','YFT']
rows = 32
cols = 32


# In[3]:

def img_get(fish):
    path = '../train/{}/'.format(fish)
    im = [path+'{}'.format(i) for i in os.listdir(path) if '.jpg' in i]
    return im


def img_read(path):
    img = cv2.imread(path)
    resize = cv2.resize(img, (rows,cols), interpolation = cv2.INTER_LINEAR)
    return resize

def test_img_get():
    path = '../test/'
    im = [path+'{}'.format(i) for i in os.listdir(path) if '.jpg' in i]
    return im


# In[4]:

def make_train():
    start = time.time()
    X_train = []
    X_id = []
    y_train =[]
    
    for fish in folders: 
        fish_path = img_get(fish)
        
        for i in fish_path:
            fish_mtx = img_read(i)
            X_train.append(fish_mtx)
        
        path = '../train/{}/'.format(fish)
        X_id.extend([i for i in os.listdir(path) if '.jpg' in i])
        
        fish_index = np.tile(folders.index(fish), len(fish_path))
        y_train.extend(fish_index)
    end = time.time()
    t1 = end - start
    return X_train, y_train, X_id, t1


# In[5]:

def make_test():
    start = time.time()
    X_test = []
    X_test_id =[]
    
    paths = test_img_get()
        
    for i in paths:
        fish_mtx = img_read(i)
        X_test.append(fish_mtx)
        
    path_id = '../test/'
    X_test_id.extend([i for i in os.listdir(path_id) if '.jpg' in i])
    end = time.time()
    t2 = end - start
    return X_test, X_test_id, t2


# In[6]:

X_train, y_train, X_id, t1 = make_train()
X_test, X_test_id, t2 = make_test()

y_trainArr = np_utils.to_categorical(y_train, nb_classes=len(folders))
X_trainArr = np.array(X_train)

X_testArr = np.array(X_test)


# In[223]:

#1.61025
sgd = SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)


def cnn_model1():
    model = Sequential()
    model.add(Convolution2D(8,3,3,input_shape = (rows,cols,3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    
    model.add(Convolution2D(8,3,3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    
    model.add(Convolution2D(8,3,3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    
    model.add(Flatten())
    model.add(Dense(8))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(len(folders)))
    model.add(Activation('softmax'))
    
    model.compile(loss='categorical_crossentropy',optimizer='sgd')
    return model


# In[242]:

#1.56465
sgd = SGD(lr=0.01, momentum=0.5, decay=0.0, nesterov=False)


def cnn_model2():
    model = Sequential()
    model.add(Convolution2D(8,3,3,input_shape = (rows,cols,3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    
    model.add(Convolution2D(16,3,3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    
    model.add(Convolution2D(32,3,3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    
    model.add(Flatten())
    model.add(Dense(96))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(len(folders)))
    model.add(Activation('softmax'))
    
    model.compile(loss='categorical_crossentropy',optimizer='sgd')
    return model


# In[34]:

#1.49077
#The change from above is just momentum from 0.5 to 0.45.
sgd = SGD(lr=0.01, momentum=0.45, decay=0.0, nesterov=False)


def cnn_modelFinal():
    model = Sequential()
    model.add(Convolution2D(8,3,3,input_shape = (rows,cols,3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    
    model.add(Convolution2D(16,3,3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    
    model.add(Convolution2D(32,3,3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    
    model.add(Flatten())
    model.add(Dense(96))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(len(folders)))
    model.add(Activation('softmax'))
    
    model.compile(loss='categorical_crossentropy',optimizer='sgd')
    return model


# In[ ]:

#Above model with 64x64 size:
#Epoch 30/30
#3399/3399 [==============================] - 7s - loss: 1.1021 - val_loss: 1.8279
#Named:  submission10-97-13BESTONE

#Above model with 64x64 size:
#Epoch 30/30
#3399/3399 [==============================] - 30s - loss: 0.5726 - val_loss: 1.2458
#Named:   submission01-41-13BESTONE2.csv


# In[17]:

md = cnn_modelFinal()
md.fit(X_trainArr,y_trainArr,nb_epoch = 30,validation_split = 0.1,batch_size=100)


# In[43]:

preds = md.predict(X_testArr, batch_size=100, verbose=1)
predsFrame = pd.DataFrame(preds,columns = folders)
predsFrame.insert(0, 'image', X_test_id)
predsFrame.head()

predsFrame.to_csv('submissionCNN.csv',index=False)


# In[33]:



