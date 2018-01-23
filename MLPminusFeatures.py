
# coding: utf-8

# In[44]:

import pandas as pd
import numpy as np
import os
import cv2
import time

from sklearn.neural_network import MLPRegressor,MLPClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score as cv
from sklearn.model_selection import cross_val_predict
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import log_loss

from keras.optimizers import SGD
from keras.utils import np_utils


# In[46]:

folders = ['ALB','BET','DOL','LAG','NoF','OTHER','SHARK','YFT']
rows = 32
cols = 32

def img_get(fish):
    path = '../train/{}/'.format(fish)
    im = [path+'{}'.format(i) for i in os.listdir(path) if '.jpg' in i]
    return im


def img_read(path):
    img = cv2.imread(path)
    resize = cv2.resize(img, (rows,cols), interpolation = cv2.INTER_LINEAR)
    resize = resize.ravel()
    return resize

def test_img_get():
    path = '../test/'
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
            X_train.append(fish_mtx)
        
        path = '../train/{}/'.format(fish)
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
        X_test.append(fish_mtx)
        
    path_id = '../test/'
    X_test_id.extend([i for i in os.listdir(path_id) if '.jpg' in i])
    return X_test, X_test_id


#Loading training set
X_train, y_train, X_id = make_train()
X_test, X_test_id = make_test()

y_trainArr = np_utils.to_categorical(y_train, nb_classes=len(folders))
X_trainArr = np.array(X_train)

X_testArr = np.array(X_test)


# In[67]:

mlp = MLPClassifier(hidden_layer_sizes=(10,10,10,10),activation='logistic',solver='adam',momentum = 0.5)    
mlp.fit(X_trainArr,y_trainArr)

def cVal(X,Y,mlp):
    start = time.time()
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


# In[72]:

#mlp = MLPClassifier(hidden_layer_sizes=(10,10,10,10),activation='logistic',solver='adam',momentum = 0.5)    
#cVal = 2.2002026525318135

#mlp = MLPClassifier(hidden_layer_sizes=(10,10,10,10),activation='logistic',solver='sgd',momentum = 0.8)    
#cVal = 2.0922429092974477

#mlp = MLPClassifier(hidden_layer_sizes=(10,10,10,10),activation='logistic',solver='sgd',momentum = 0.5)
#cVal = 1.9838959032447363

#mlp = MLPClassifier(hidden_layer_sizes=(10,10,10),activation='logistic',solver='sgd',momentum = 0.5)    
#cVal = 2.0007166038443303

#mlp = MLPClassifier(hidden_layer_sizes=(10,10,10,10),activation='tanh',solver='sgd',momentum = 0.5)
#cVal = 2.1740325078100544


# In[55]:

preds = mlp.predict_proba(X_testArr)
predsFrame = pd.DataFrame(preds,columns = folders)
predsFrame.insert(0, 'image', X_test_id)
predsFrame.head()


# In[56]:

predsFrame.to_csv('submission.csv',index=False)


# In[ ]:



