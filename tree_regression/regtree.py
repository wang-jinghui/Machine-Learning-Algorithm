# -*- coding:utf-8 -*-
from numpy import *
import cPickle as pickle
# 加载数据集
def loadDataSet(fileName):
    dataSet = []
    fr = open(fileName)
    for line in fr.readlines():
        curline = line.strip().split('\t')
        fltLine = map(float,curline)      # 每行为浮点数
        dataSet.append(fltLine)
    return dataSet       # 嵌套列表
# 根据特征值将数据切分为两个子集
def binSplitDataSet(dataSet,feature,value):
    mat0 = dataSet[nonzero(dataSet[:,feature] > value)[0],:]     
    mat1 = dataSet[nonzero(dataSet[:,feature] <=value)[0],:]
    return mat0,mat1

# 在不对数据进行切分时，返回目标变量的均值

def regLeaf(dataSet):
    return mean(dataSet[:,-1])
# 误差估计函数，返回总方差  (连续变量的混乱度）
def regError(dataSet):
    return var(dataSet[:,-1])*len(dataSet)

# 选择最佳切分特征和 特征值

def chooseBestSplit(dataSet,leafType = regLeaf,errType = regError,ops= (0,1)):
    tolE = ops[0]    #误差减小的最小值
    tolN = ops[1]    #划分子集的最小样本数 （限制条件有利于防止过拟合）
    # 目标变量在去重后，长度为1，即 所有值相等，推出
    if len(set(dataSet[:,-1].T.tolist()[0]))==1:
        return None,leafType(dataSet)    # 返回目标变量的均值
    m,n = shape(dataSet)
    varError = errType(dataSet)          # 目标变量的总方差
    bestVarError = inf
    bestIndex = 0
    bestValue = 0
    for featIndex in range(n-1):
        for value in set(array(dataSet)[:,featIndex]):
            mat0,mat1 = binSplitDataSet(dataSet,featIndex,value)
            # 判断划分的子集，是否符合最低要求
            if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):
                continue
            # 两个子集的总方差和
            newVarError = errType(mat0) + errType(mat1)
            if newVarError < bestVarError:
                bestIndex = featIndex
                bestValue = value
                bestVarError = newVarError
    # 如果bestVarError跟初始误差，差值太小，只返回叶节点的值（目标变量的均值） 误差下降的太小
    if  (varError - bestVarError) < tolE:
        return None,leafType(dataSet)
    mat0,mat1 = binSplitDataSet(dataSet,bestIndex,bestValue)
    # 根据找到的 beatIndex 和 bestValue ,切分的子集太小，返回叶子节点，feat = None
    if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):
        return None,leafType(dataSet)
    return bestIndex,bestValue


# creat tree
def creatTree(dataSet,leafType=regLeaf,errType=regError,ops=(0,1)):
    feat,val = chooseBestSplit(dataSet,leafType,errType,ops)
    # 如果没有找到最佳特征，返回叶子节点
    if feat == None:
        return val
    retTree = {}
    retTree['spInd'] = feat
    retTree['spal'] = val
    Lset ,Rset = binSplitDataSet(dataSet,feat,val)
    retTree['left'] = creatTree(Lset,leafType,errType,ops)
    retTree['right'] = creatTree(Rset,leafType,errType,ops)
    return retTree
# 持久化回归树
data = loadDataSet('data/ex2.txt')
matdata = mat(data)
tree = creatTree(matdata,ops=(1,15))
def saveTree(tree,path):
    obj = open(path,'w')
    pickle.dump(tree,obj)
    obj.close()
def loadTree(path):
    obj = open(path,'r')
    tree = pickle.load(obj)
    obj.close()
    return tree
