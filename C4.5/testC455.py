# -*- coding:utf-8 -*-
import numpy as np
import time
# 把源文件 转换为矩阵 便于后来的降维处理
def img2Matrix(filename):
    fr = open(filename)
    featuremat = np.zeros((32,32))
    for i in range(32):
        lineStr = fr.readline()    # 每行为长度为32的字符串，无任何间隔字符
        for j in range(32):
            featuremat[i,j] = int(lineStr[j])
    return featuremat
# 把元数据转换为 一维特征向量
def img2Vector(filename):
    fr = open(filename)
    fvector = np.zeros((1,1024))
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            fvector[0,32*i+j] = int(lineStr[j])
    return fvector

####手写图片二进制文件识别
handlabes = []
trainFilelist = []
# train 文件名，和标签
for i in range(10):
    for j in range(180):
        filename = str(i)+'_'+str(j)+'.txt'
        handlabes.append(i)
        trainFilelist.append(filename)
m = len(trainFilelist)
trainFvector = np.zeros((m,1024))  # 存储train向量
for i in range(m):
    trainFvector[i:] = img2Vector('../trainingDigits/'+trainFilelist[i])

################## 测试文件，向量################
testlabel = []
testFilelist = []
# 生成文件名，和测试标签
for i in range(10):
    for j in range(80):
        filename =str(i)+'_'+str(j)+'.txt'
        testlabel.append(i)
        testFilelist.append(filename)
n = len(testFilelist)
testFvector = np.zeros((n,1024))   # 存储测试向量
# 读取测试文件并转换为特征向量
for i in range(n):
    testFvector[i,:] = img2Vector('../testDigits/'+testFilelist[i])
# 测试识别手写识别算法
error = 0.0
start = time.clock()
from C455 import *
featureNames = range(1023)
dataSet = trainFvector
c45 = C45tree()
c45.dataSet = dataSet
c45.featureNames = featureNames
c45.train()
print c45.tree

