# -*- coding:utf-8 -*-
# c4.5 算法
from numpy import *
import math
import warnings
import copy
import cPickle as pickle
datapath = '../data/dataset.dat'
# 构造树
class C45Tree(object):
    def __init__(self):
        self.tree = {}
        self.dataSet = []
        self.featureNames = []
    def loadDataSet(self,path,featureNames):
        fp = open(path,'rb')
        content = fp.read()
        fp.close()
        rowlist = content.splitlines()
        recordlist = [row.split('\t') for row in rowlist if row.strip()]
        self.dataSet = recordlist
        self.featureNames = featureNames
    def train(self):
        featureNames = copy.deepcopy(self.featureNames)
        self.tree = self.buildTree(self.dataSet,featureNames)
    def buildTree(self,dataSet,featureNames):
        classLable = [data[-1] for data in dataSet]
        if classLable.count(classLable[0]) == len(classLable):
            return classLable[0]
        if dataSet[0] == 1:
            return self.maxLabel(classLable)
        bestFeatIndex = self.chooseBestFeat(dataSet)
        bestFeatName = featureNames[bestFeatIndex]
        c45Tree = {bestFeatName:{}}
        del featureNames[bestFeatIndex]
        featValues = set([data[bestFeatIndex] for data in dataSet])
        for featvalue in featValues:
            featNames = featureNames[:]
            subDataSet = self.splitDataSet(dataSet,bestFeatIndex,featvalue)
            nextDict = self.buildTree(subDataSet,featNames)
            c45Tree[bestFeatName][featvalue] = nextDict
        return c45Tree
    def maxLabel(self,classLabel):
        count = dict([(classLabel.count(i),i) for i in classLabel])
        return count[max(count.keys())]
    def chooseBestFeat(self,dataSet):
        numFeatures = len(dataSet[0]) -1
        lenDataSet = len(dataSet)
        baseEntory = self.computeEntory(dataSet)
        InfoGainRate = -1
        bestFeat = 0
        for featIndex in xrange(numFeatures):
             featVals= [example[featIndex] for example in dataSet]
             splitInfo = self.computeSplitInfo(featVals)
             featEntory = 0.0
             uniqueVals = set(featVals)
             for value in uniqueVals:
                 subDataSet = self.splitDataSet(dataSet,featIndex,value)
                 prob = float(len(subDataSet))/float(lenDataSet)
                 featEntory += prob*self.computeEntory(subDataSet)
             # 消除警告
             warnings.filterwarnings('ignore')
             gainRate = (baseEntory - featEntory)/splitInfo
             if gainRate > InfoGainRate:
                 InfoGainRate = gainRate
                 bestFeat = featIndex
        return bestFeat
    # 计算香农熵
    def computeEntory(self,dataSet):
        classLabels = [sample[-1] for sample in dataSet]
        lenClaLabel = float(len(classLabels))
        uniqueClassLabel = set(classLabels)
        shannonEntory = 0.0
        for label in uniqueClassLabel:
            prob = classLabels.count(label)/lenClaLabel    # 小心地板除
            shannonEntory -= prob * math.log(prob,2)
        return shannonEntory
    # 计算划分信息
    def computeSplitInfo(self,featVals):
        numFeatVal = float(len(featVals))
        uniqueFeatVal = set(featVals)
        counts = [featVals.count(val) for val in featVals]
        items = [float(count)/numFeatVal for count in counts]
        Info = [item*math.log(item,2) for item in items]
        splitInfo = -sum(Info)
        return splitInfo
    def splitDataSet(self,dataSet,featIndex,value):
        subDataSet = []
        for sample in dataSet :
            if sample[featIndex] == value:
                retlist = sample[:featIndex]
                retlist.extend(sample[featIndex+1:])
                subDataSet.append(retlist)
        return subDataSet
    def saveC45Tree(self,c45,path):
        obj = open(path,'w')
        pickle.dump(c45,obj)
        obj.close()
    def loadC45Tree(self,path):
        obj = open(path,'r')
        c45 = pickle.load(obj)
        obj.close()
        return c45
    def predict(self,c45,featureNames,testvec):
        root = c45.keys()[0]
        nextDict = c45[root]
        index = featureNames.index(root)
        value = testvec[index]
        if isinstance(nextDict[value],dict):
            classLabel = self.predict(nextDict[value],featureNames,testvec)
        else:
            classLabel = nextDict[value]
        return classLabel










