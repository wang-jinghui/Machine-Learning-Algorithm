# -*- coding:utf-8 -*-

import numpy as np

# theta1,theta2 为初始化的随机权重矩阵
# 传入的y需为一维数组
def nnCostFunction(theta1,theta2,X,y,num_labels,lamda=0):
    m,n=np.shape(X)
    theta1_grad = np.zeros(theta1.shape)
    theta2_grad = np.zeros(theta2.shape)
    Y = array_y(y,num_labels)    # if y[23]=3,Y[23,:]=00010000000
    X = biasplus(X)        # X m*(n+1) array
    z2 = X*theta1.T        # m*隐藏层神经元的个数
    a2 = sigmoid(z2)
    a2 = biasplus(a2)      # m*隐藏层神经元的个数+1
    z3 = a2*theta2.T       # m*输出层神经元的个数
    a3 = sigmoid(z3)
    eta1 = justRegular(theta1)    # theta0不参与正则化
    eta2 = justRegular(theta2)
    temp1 = np.multiply(eta1,eta1)   #矩阵转换为数组
    temp2 = np.multiply(eta2,eta2)
    partOfJ = np.multiply(Y,np.log(a3))+np.multiply((1-Y),np.log(1-a3))
    J = -1*sum(sum(partOfJ.A))/m+lamda*(sum(sum(temp1.A))+sum(sum(temp2.A)))/(2*m)

    # compute error
    error3 = np.mat(a3-Y)           # m*输出神经元的个数
    error = error3*theta2           # m*影藏层神经元个数+1
    error2 = np.multiply(error[:,1:],sigmoidGradient(z2))
    # calculate Gradient
    for j in xrange(m):
        theta1_grad = theta1_grad+error2[j,:].T*X[j,:]
        theta2_grad = theta2_grad+error3[j,:].T*a2[j,:]
    # regularization
    '''
    regular1 = lamda/m*eta1
    regular2 = lamda/m*eta2
    '''
    theta1_grad = theta1_grad/m
    theta2_grad = theta2_grad/m
    return J,theta1_grad,theta2_grad
   


def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))
def biasplus(array):
    m,n = array.shape
    temp = np.zeros((m,n+1))
    temp[:,0] = 1.0
    temp[:,1:] = array
    return temp

def sigmoidGradient(z):
    return np.multiply(sigmoid(z),(1-sigmoid(z)))
def justRegular(theta):
    theta[:,0] =0.0
    return theta

def array_y(y,num_labels):
    m = len(y)
    Y = np.zeros((m,num_labels))     # 是一个m*num_label的二维数组
    e = np.eye(num_labels)
    for i in xrange(num_labels):
        line_array = np.where(y==i)[0]
        Y[line_array,:] = e[i,:]
    return Y    






