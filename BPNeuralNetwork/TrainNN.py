# -*- coding:utf-8 -*-
import numpy as np
from NNCostFunction import *

def trainNN(theta1,theta2,X,y,num_labels,looptimes,mini_batch=100,lamda=0,alpha=3.0):
    J_list = []

    m,n = X.shape
    for i in xrange(looptimes):
        for j in xrange(0,m,mini_batch):
            mini_x = X[j:j+mini_batch,:]
            mini_y = y[j:j+mini_batch]
            J,theta1_grad,theta2_grad=nnCostFunction(theta1,theta2,mini_x,mini_y,num_labels,lamda)
            J_list.append(J)
            theta1 = theta1 - alpha/mini_batch*theta1_grad
            theta2 = theta2 - alpha/mini_batch*theta2_grad

    return J_list,theta1,theta2

# train ,validation,test error
# so x =train/validation/test
def lossfunction(x,y,num_labels,theta1,theta2):
    m,n = x.shape
    e = np.eye(num_labels)
    Y =  array_y(y,num_labels)
    x = biasplus(x)
    z2 = x*theta1.T
    a2 = sigmoid(z2)
    a2 = biasplus(a2)
    z3 = a2*theta2.T
    a3 = sigmoid(z3)                 # len(x)*output nuits
    loss = np.multiply(Y,np.log(a3))+np.multiply((1-Y),np.log(1-a3))
    loss = -1*sum(sum(loss.A))/m
    return loss
    






	 
	 
	 





