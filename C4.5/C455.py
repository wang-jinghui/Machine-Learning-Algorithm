# -*- coding:utf-8 -*-
from numpy import *
import math
import copy
import warnings
import cPickle as pickle
# 构造树
class C45tree(object):
    def __init__(self):
        self.tree = {}
        self.dataSet = []
        self.featureNames = []
    def loadDataSet(self,path,featureNames):
        fp = open(path,'r')
        content = fp.read()
        fp.close()
        rowlist = content.splitlines()
        dataSet = [row.split('\t') for row in rowlist if row.strip()]
        self.dataSet = dataSet
        self.featureNames = featureNames
    def train(self):
        featureNames = copy.deepcopy(self.featureNames)
        self.tree = self.buildTree(self.dataSet,featureNames)
    def buildTree(self,dataSet,featureNames):
        classLables = [sample[-1] for sample in dataSet]
        if classLables.count(classLables[0]) == len(classLables):
            return classLables[0]
        if len(dataSet[0]) ==1:
            return self.maxLable(classLables)
        bestFeat,uniqueVal = self.findFeature(dataSet)
        bestFeatName = featureNames[bestFeat]
        tree = {bestFeatName:{}}
        del (featureNames[bestFeat])

        for value in uniqueVal:
            featNames = featureNames[:]
            subDataSet = self.splitDataSet(dataSet,bestFeat,value)
            nextTree = self.buildTree(subDataSet,featNames)
            tree[bestFeatName][value]=nextTree
        return tree
    def maxLable(self,classLabels):
        uniqueLabel = set(classLabels)
        count = dict([(classLabels.count(label),label) for label in uniqueLabel])
        return count[max(count.keys())]
    def findFeature(self,dataSet):
        numFeature = len(dataSet[0])-1
        lenDataSet = float(len(dataSet))
        baseEntory = self.computeEntory(dataSet)
        featureEntory = []    # 特征的熵
        splitInfos = []        # 特征的划分
        uniqueValList = []    # 每个特征的，去重values
        for index in xrange(numFeature):
            featValList = [sample[index] for sample in dataSet]
            [splitInfo,uniqueVal] = self.splitInfo(featValList)
            splitInfos.append(splitInfo)       # append一个特征的信息划分
            uniqueValList.append(uniqueVal)    # 特征的去重特征值
            featEntory = 0.0
            for value in uniqueVal:
                subDataSet = self.splitDataSet(dataSet,index,value)
                prob = len(subDataSet)/lenDataSet
                featEntory += prob*self.computeEntory(subDataSet)    # 特征信息熵
            featureEntory.append(featEntory)
        infoGainArray = baseEntory*ones(numFeature)-array(featureEntory)
        warnings.filterwarnings('ignore')
        infoGainRate = infoGainArray/array(splitInfos)        #数组除法,信息增益率
        bestFeature = argsort(-infoGainRate)[0]
        return bestFeature,uniqueValList[bestFeature]
    def splitDataSet(self,dataSet,index,value):
        subDataSet = []
        for sample in dataSet:
            if sample[index] == value:
                retlist = sample[:index]
                retlist.extend(sample[index+1:])
                subDataSet.append(retlist)
        return subDataSet

    def computeEntory(self,dataSet):
        classLabels = [sample[-1] for sample in dataSet]
        numClassLabel = float(len(classLabels))
        uniqueLabel = set(classLabels)
        counts = [classLabels.count(label) for label in uniqueLabel]
        items = [count/numClassLabel for count in counts]
        shannonEntory = 0.0
        for item in items:
            shannonEntory -= item*math.log(item)
        return shannonEntory
    def splitInfo(self,featValList):
        uniqueVal = set(featValList)
        lenfeatVal = float(len(featValList))
        counts = [featValList.count(value) for value in uniqueVal]
        items = [count/lenfeatVal for count in counts]
        splitinfo = [item*math.log(item,2) for item in items]
        splitInfo = -sum(splitinfo)
        return splitInfo,uniqueVal
    def predict(self,tree,featureNames,testvec):
        root = tree.keys()[0]
        nextDict = tree[root]
        index = featureNames.index(root)
        key = testvec[index]
        if isinstance(nextDict[key],dict):
            return  self.predict(nextDict[key],featureNames,testvec)
        else:
            return nextDict[key]





