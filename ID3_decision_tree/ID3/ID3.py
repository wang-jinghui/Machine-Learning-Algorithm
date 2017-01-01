#-*- coding:utf-8 -*-
from numpy import *
import math
import copy
import cPickle as pickle

datapath = '../data/dataset.dat'
# 构造树
class ID3Tree(object):
    def __init__(self):
        self.tree = {}               # 生成的树
        self.dataSet = []            # 数据集
        self.featureNames = []       # 特征名字列表
    # 导入数据
    def loadDataSet(self,path,featureNames):
        fp = open(path,'rb')
        content = fp.read()
        fp.close()
        rowlist = content.splitlines()       # 按行转换为一列表（每行为列表元素）
        recordlist = [row.split('\t') for row in rowlist if row.strip()]  #row.strip() 没什么卵用
        self.dataSet = recordlist
        self.featureNames = featureNames     # 特征列表
#   执行决策树
    def train(self):
        featureNames = copy.deepcopy(self.featureNames)
        self.tree = self.buildTree(self.dataSet,featureNames)

    # 构建决策树
    def buildTree(self,dataSet,featureNames):
        cateList = [data[-1] for data in dataSet]            # 抽取标签
        if cateList.count(cateList[0]) == len(cateList):     # 标签只有一种标签，直接返回该标签
            return cateList[0]
        # 随着数据集的不断切分，只剩下一个特征
        if len(dataSet[0]) == 1:
            return self.maxCate(cateList)
        bestFeat = self.getBestFeat(dataSet)
        bestFeatLabel = featureNames[bestFeat]
        tree = {bestFeatLabel:{}}
        del featureNames[bestFeat]
        uniqueVals = set([data[bestFeat] for data in dataSet])   # 特征值去重
        for value in uniqueVals:
            sublabels = featureNames[:]
            subDataSet = self.splitDataSet(dataSet,bestFeat,value)
            subtree = self.buildTree(subDataSet,sublabels)
            tree[bestFeatLabel][value] = subtree
        return tree

    # cateList 标签列
    # 返回出项次数最多的标签
    def maxCate(self,cateList):
        count = dict([(cateList.count(i),i) for i in cateList])
        return count[max(count.keys())]
    # 得到最佳标签
    def getBestFeat(self,dataSet):
        numFeatures = len(dataSet[0] )-1            # 去掉标签列
        baseEntory = self.computeEntory(dataSet)    # 计算原始数据的 香农熵
        baseInfoGain = 0.0
        bestFeat = -1
        for featureIndex in xrange(numFeatures):    # 遍历每一个特征
            featValues = [data[featureIndex] for data in dataSet]     # 特征值列表
            featEntory = 0.0
            uniqueValues = set(featValues)           # 去重
            for value in uniqueValues:               # 遍历每一个特征值
                subDataSet = self.splitDataSet(dataSet,featureIndex,value)   # 切分数据
                prob = float(len(subDataSet))/len(dataSet)
                featEntory += prob * self.computeEntory(subDataSet)          # 每个特征的香农熵
            InforGain = baseEntory - featEntory
            if InforGain > baseInfoGain:
                baseInfoGain = InforGain
                bestFeat = featureIndex
        return bestFeat
    # 计算香农熵
    def computeEntory(self,dataSet):
        datalen = float(len(dataSet))
        cateList = [data[-1] for data in dataSet]           # 标签列表
        uniqueCateList = set(cateList)                      # 去重
        inforEntory = 0.0
        for label in uniqueCateList:
            prob = float(cateList.count(label))/datalen     # 每个标签的分类概率
            inforEntory -= prob*math.log(prob,2)            # 信息熵
        return inforEntory
    # 切分数据 （数据集，特征索引，特征值）
    def splitDataSet(self,dataSet,featureIndex,value):
        retlist = []
        for data in dataSet:
            if data[featureIndex] == value:
                reducelist = data[:featureIndex]               # 去掉最佳特征列
                reducelist.extend(data[featureIndex+1:])
                retlist.append(reducelist)    # 嵌套列表，存储取同一value的子集
        return retlist

    # 保存树
    def saveTree(self,tree,path):
        obj = open(path,'w')
        pickle.dump(tree,obj)
        obj.close()
    def loadTree(self,path):
        obj = open(path,'r')
        tree = pickle.load(obj)
        obj.close()
        return tree

    def predict(self,ID3tree,featureNames,testvec):
        root = ID3tree.keys()[0]                # 决策树的根节点
        secDict = ID3tree[root]                 # 以根节点为 key的字典
        featIndex = featureNames.index(root)    # 根节点在特征列，中的位置索引
        key = testvec[featIndex]         # 测试向量 在第一个分类特征上的取值（也就是第二个字典的key）
        if isinstance(secDict[key],dict):    # 如果第二个字典的value是字典，迭代下去
            classifyResult = self.predict(secDict[key],featureNames,testvec)
        else:
            classifyResult = secDict[key]
        return classifyResult


