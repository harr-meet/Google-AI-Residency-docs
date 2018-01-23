
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
import os
import cv2
import time

from keras.preprocessing.image import ImageDataGenerator
from sklearn.neural_network import MLPRegressor,MLPClassifier
from matplotlib import pyplot as plt
from matplotlib.pyplot import savefig
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.core import Activation, Dense, Dropout, Flatten
from keras.optimizers import SGD
from keras.utils import np_utils
from sklearn.model_selection import KFold
from sklearn.metrics import log_loss

from keras import backend as K
K.set_image_dim_ordering('tf')


import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import data, color, exposure


# In[ ]:

folders = ['ALB','BET','DOL','LAG','NoF','OTHER','SHARK','YFT']
rows = 32
cols = 32

def img_get(fish):
    path = '/modules/cs342/Assignment2/Data/train/{}/'.format(fish)
    im = [path+'{}'.format(i) for i in os.listdir(path) if '.jpg' in i]
    return im


def img_read(path):
    img = cv2.imread(path)
    resize = cv2.resize(img, (rows,cols), interpolation = cv2.INTER_LINEAR)
    return resize

def test_img_get():
    path = '/modules/cs342/Assignment2/Data/test/'
    im = [path+'{}'.format(i) for i in os.listdir(path) if '.jpg' in i]
    return im


datagen = ImageDataGenerator(
    rescale = 1./255,
    shear_range = 0.3,
    zoom_range = 0.3,
    height_shift_range = 0.3,
    width_shift_range = 0.3,
    horizontal_flip = True,
    channel_shift_range = 0.3,
    rotation_range = 40,
    fill_mode = 'nearest')

def gen_imgs(fish,img, amount):
    i = 0
    for batch in datagen.flow(img, batch_size=1, save_to_dir = '/modules/cs342/Assignment2/Data/train/{}/'.format(fish), 
                              save_prefix = 'img',save_format = 'jpeg'):
        i += 1
        if i > amount:
            break
            
def data_aug(fish):
    for k in range(0,10):
        img = img_read(img_get(fish)[k])
        img = img.reshape((1,) + img.shape)
        gen_imgs(fish,img, 50)

#We shall only add images to these classess, as ALB and YFT have many
data_aug('BET')
data_aug('DOL')
data_aug('LAG')
data_aug('NoF')
data_aug('OTHER')
data_aug('SHARK')

def make_train():
    X_train = []
    X_id = []
    y_train =[]
    
    for fish in folders: 
        fish_path = img_get(fish)
        
        for i in fish_path:
            fish_mtx = img_read(i)
            fish_mtx = np.ravel(fish_mtx)
            X_train.append(fish_mtx)
        
        path = '/modules/cs342/Assignment2/Data/train/{}/'.format(fish)
        X_id.extend([i for i in os.listdir(path) if '.jpg' in i])
        
        fish_index = np.tile(folders.index(fish), len(fish_path))
        y_train.extend(fish_index)
    return X_train, y_train, X_id

def make_test():
    X_test = []
    X_test_id =[]
    
    paths = test_img_get()
        
    for i in paths:
        fish_mtx = img_read(i)
        fish_mtx = np.ravel(fish_mtx)
        X_test.append(fish_mtx)
        
    path_id = '/modules/cs342/Assignment2/Data/test/'
    X_test_id.extend([i for i in os.listdir(path_id) if '.jpg' in i])
    return X_test, X_test_id


#Loading training set
X_train, y_train, X_id = make_train()
X_test, X_test_id = make_test()

y_trainArr = np_utils.to_categorical(y_train, nb_classes=len(folders))
X_trainArr = np.array(X_train)

X_testArr = np.array(X_test)


# In[ ]:

mlp = MLPClassifier(hidden_layer_sizes=(10,10,10,10),activation='logistic',solver='sgd',momentum = 0.5)    
mlp.fit(X_trainArr,y_trainArr)

def cVal(X,Y,mlp):
    k_fold = KFold(n_splits=10)
    l = 0
    X = pd.DataFrame(X)
    Y = pd.DataFrame(Y)
    for train,test in k_fold.split(X):
        mlp = mlp.fit(X.iloc[train],Y.iloc[train])
        p = mlp.predict_proba(X.iloc[test])
        l += log_loss(Y.iloc[test],p)
    logloss = l/10
    return logloss

loss = cVal(X_trainArr, y_trainArr, mlp)


# In[ ]:

preds = mlp.predict_proba(X_testArr)
predsFrame = pd.DataFrame(preds,columns = folders)
predsFrame.insert(0, 'image', X_test_id)
predsFrame.head()

predsFrame.to_csv('submissionAugment.csv',index=False)

