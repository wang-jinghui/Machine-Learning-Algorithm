# -*- coding:utf-8 -*-
import numpy as np
def loadDataSet():
    postingList = [['my','dog','has','flea','problems','help','please'],
                   ['maybe','not','take','him','to','dog','park','stupid'],
                   ['my','dalmation','is','so','cute','I','love','him'],
                   ['stop','posting','stupid','worthless','garbage'],
                   ['mr','licks','ate','my','steak','how','to','stop','him'],
                   ['quit','buying','worthless','dog','food','stupid']]
    classVec = [0,1,0,1,0,1]
    return postingList,classVec
def createVocabList(dataset):
    vocabList = set([])
    for line in dataset:

        vocabList = vocabList | set(line)
    return list(vocabList)

# 根据vocabulary 弄出普通向量空间 和 Tf_idf 权重向量空间
def setOfWord2Vec(vocabList,dataset):
    numLine = len(dataset)
    lenVec = len(vocabList)
    matVec = np.zeros((numLine,lenVec))
    IdfVec = np.zeros((1,lenVec))
    TfVec = np.zeros((numLine,lenVec))
    for line in range(numLine):
        for word in dataset[line] :
            matVec[line,vocabList.index(word)] += 1
        TfVec[line]=matVec[line]/float(len(dataset[line]))
        for word in set(dataset[line]):
            IdfVec[0,vocabList.index(word)] += 1
    IdfVec = np.log(float(numLine)/IdfVec)
    return matVec
dataset ,classVec = loadDataSet()
vocabList = createVocabList(dataset)
matvec = setOfWord2Vec(vocabList,dataset)
# 计算出p(x|y0)*p(y0) 和 p(x|y1)*p(y1)
def Nbayes(matvec,classVec):
    p0vec = np.zeros((1,len(matvec[0])))
    p1vec = np.zeros((1,len(matvec[0])))
    numDoc = len(matvec)
    p1 = sum(classVec)/float(len(classVec))
    p0 = 1.0 - p1
    for i in range(numDoc):
         if classVec[i]==0:
             p0vec += matvec[i]
         else:
             p1vec += matvec[i]
    p0vec = p0vec/float(np.sum(p0vec))*p0
    p1vec = p1vec/float(np.sum(p1vec))*p1
    return p0vec,p1vec

p0vec,p1vec = Nbayes(matvec,classVec)

# 分类 就算分类概率
def classifyNB(testvec,p0vec,p1vec):
    p1value = np.sum(testvec*p1vec)
    p0value = np.sum(testvec*p0vec)
    if p1value > p0value:
        return 1
    else:
        return 0
def TestVec(testset):
    testvec = np.zeros((1,len(vocabList)))
    for word in testset:
        testvec[0,vocabList.index(word)] += 1
    return testvec
testset = ['stupid','garbage']
testvec =TestVec(testset)
print classifyNB(testvec,p0vec,p1vec)



