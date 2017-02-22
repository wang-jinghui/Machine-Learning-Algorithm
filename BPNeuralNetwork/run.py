# -*- coding:utf-8 -*-
import  cPickle as pickle
from TrainNN import *
import matplotlib.pyplot as plt
from process_dataset import train_x,train_y,val_x,val_y,test_x,test_y


initializa_theta1 = np.mat(np.random.rand(30,785))
initializa_theta2 = np.mat(np.random.rand(10,31))

print u'开始训练'
theta1_list=[]
theta2_list=[]   # 不同size训练数据得到的参数
for k in xrange(0,28000,1000):
    J_list,theta1,theta2 = trainNN(initializa_theta1,initializa_theta2,
                                   train_x[k:k+1000,:],train_y[k:k+1000],10,30,lamda=0,alpha=3.0)
    theta1_list.append(theta1)
    theta2_list.append(theta2)
    print u'%d个训练样本，训练完成'%(k+1000)

print 'wait moment.........'
print u'正在计算训练误差和交叉验证误差'
train_loss=[]
val_loss =[]
test_loss =[]
for theta1,theta2 in zip(theta1_list,theta2_list):
    t_loss = lossfunction(train_x,train_y,10,theta1,theta2)
    v_loss = lossfunction(val_x,val_y,10,theta1,theta2)
    te_loss = lossfunction(test_x,test_y,10,theta1,theta2)
    train_loss.append(t_loss)
    val_loss.append(v_loss)
    test_loss.append(te_loss)
print u'计算完成，正在绘制学习曲线。。。'
print train_loss
print val_loss
print test_loss
