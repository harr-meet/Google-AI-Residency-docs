
# coding: utf-8

# In[2]:

import pandas as pd
import numpy as np
import os
import cv2
import time

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


# In[5]:

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


def make_train():
    X_train = []
    X_id = []
    y_train =[]
    
    for fish in folders: 
        fish_path = img_get(fish)
        
        for i in fish_path:
            fish_mtx = img_read(i)
            fish_mtx = make_hog(fish_mtx)
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
        fish_mtx = make_hog(fish_mtx)
        fish_mtx = np.ravel(fish_mtx)
        X_test.append(fish_mtx)
        
    path_id = '/modules/cs342/Assignment2/Data/test/'
    X_test_id.extend([i for i in os.listdir(path_id) if '.jpg' in i])
    return X_test, X_test_id

#Function to return histogram of ordinary gradients
def make_hog(img):
    image = color.rgb2gray(img)

    fd, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16),cells_per_block=(1, 1), visualise=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 0.02))
    return hog_image_rescaled


#Loading training set
X_train, y_train, X_id = make_train()
X_test, X_test_id = make_test()

y_trainArr = np_utils.to_categorical(y_train, nb_classes=len(folders))
X_trainArr = np.array(X_train)

X_testArr = np.array(X_test)



# In[4]:

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

predsFrame.to_csv('submissionHog.csv',index=False)

