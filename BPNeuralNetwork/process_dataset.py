# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
filename ='dataset/train.csv'
dataset = pd.read_csv(filename)
datamat = dataset.as_matrix()
trainMat = datamat[:28000,:]
testMat = datamat[28000:35000,:]
validaMat = datamat[35000:,:]
train_x = trainMat[:,1:]
train_y = trainMat[:,0]

test_x = testMat[:,1:]
test_y = testMat[:,0]
val_x = validaMat[:,1:]
val_y = validaMat[:,0]




