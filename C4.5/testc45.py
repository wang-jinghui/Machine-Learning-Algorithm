# -*- coding:utf-8 -*-
from C45 import *
path = '../data/dataset.dat'
featureNames = ['age','revenue','student','credit']

c45 = C45Tree()

c45.loadDataSet(path,featureNames)
testvec = c45.dataSet[128:512]
c45.train()
c455tree = c45.tree
lentest = len(testvec)
right = 0
for i in range(lentest):
    predict = c45.predict(c455tree,featureNames,testvec[i])
    if predict == testvec[i][-1]:
        print 'predicted classLabel is %s >>>the real is %s'%(predict,testvec[i][-1])
        right += 1
print 'the right rate is ',float(right)/lentest



