# -*- coding:utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

class BPnet(object):
    def __init__(self):
        self.eb = 0.01    # error tolerance
        self.iterator = 0
        self.eta = 0.1     #learning efficiency
        self.mc = 0.01     # assign the proportion of the last gradient
        self.maxiter = 1000
        self.nHidden = 4
        self.nOut = 1

        self.errlist = []
        self.dataMat = 0
        self.classLabels = 0
        self.nSampNum = 0     #number of samples
        self.nSampDim = 0     # number of features
    # initizlize weight
    def init_hiddenWB(self):
        self.hi_wb = np.mat(np.random.rand(self.nHidden,self.nSampDim))

    def init_OutputWB(self):
        self.out_wb = np.mat(np.random.rand(self.nOut,self.nHidden+1))

    def loadDataset(self,filename):
        self.dataMat = []
        self.classLabels = []
        fr = open(filename)
        for line in fr.readlines():
            lineArr = line.strip().split('\t')
            # biases columns
            self.dataMat.append([1.0,float(lineArr[0]),float(lineArr[1])])
            self.classLabels.append(int(lineArr[2]))
        self.dataMat = np.mat(self.dataMat)
        m,n = self.dataMat.shape
        self.nSampNum = m
        self.nSampDim = n

    def normalize(self,dataMat):
        m,n = dataMat.shape
        for i in range(1,n):
            dataMat[:,i] = (dataMat[:,i]-np.mean(dataMat[:,i]))/(np.std(dataMat[:,i])+1.0e-10)
        return dataMat
    # add biases columns
    def addcol(self,matrix):
        m,n = matrix.shape
        mat = np.zeros((m,n+1))
        mat[:,0] = 1.0
        mat[:,1:] = matrix[:,:]
        return mat


    def logistic(self,z):
        return 1.0/(1.0+np.exp(-z))

    def errorfunc(self,vec):
        return np.sum(np.power(vec,2))*0.5

    def dlogistic(self,mat):
        return np.multiply(mat,(1.0-mat))
    #draw the data
    def drawClassScatter(self,plt):
        i = 0
        for linedata in self.dataMat:
            if self.classLabels[i] ==0:
                plt.scatter(linedata[0,1],linedata[0,2],c='blue',marker='o')
            else:
                plt.scatter(linedata[0,1],linedata[0,2],c='red',marker='s')
            i+=1

    def bpTrain(self):
        SampIn = self.dataMat.T                # n * m
        expected = np.mat(self.classLabels)    # real labels mat 1*m
        self.init_hiddenWB()                   # hi_wb
        self.init_OutputWB()                   # out_wb
        dout_wbold = 0.0
        dhi_wbold = 0.0                        # last time delta wb
        for i in xrange(self.maxiter):
            hi_input = self.hi_wb*SampIn             # len(hidden)*m
            hi_output = self.logistic(hi_input)
            hi2out = self.addcol(hi_output.T).T      # len(hidden)+1 * m
            out_input = self.out_wb*hi2out           # len(self.nOut) * m
            out_output = self.logistic(out_input)
            # compute error
            err = expected - out_output         # len(self.nOut)*m matrix
            sse = self.errorfunc(err)           # like Jcost

            self.errlist.append(sse)
            if sse <= self.eb:
                self.iterator = i+1
                break;
            #calculation error
            out_error = np.multiply(err, self.dlogistic(out_output))
            hid_error = np.multiply(self.out_wb[0,1:].T*out_error,self.dlogistic(hi_output))  # 4 * 307
            dout_wb = out_error * hi2out.T     # 1 * 5
            dhi_wb = hid_error * SampIn.T        # 4 * 3

            if i ==0:
                # 误差反向传播

                self.hi_wb = self.hi_wb + self.eta*dhi_wb
                self.out_wb = self.out_wb + self.eta*dout_wb
            # add the last gradient on this impact
            else:
                self.hi_wb = self.hi_wb + (1.0-self.mc)*self.eta*dhi_wb+self.mc*dhi_wbold
                self.out_wb = self.out_wb+ (1.0-self.mc)*self.eta*dout_wb+self.mc*dout_wbold

            dhi_wbold = dhi_wb
            dout_wbold = dout_wb

    def BPClassfier(self,start,end,steps=30):
        x = np.linspace(start,end,steps)
        xx = np.mat(np.ones((steps,steps)))
        xx[:,0:steps] = x
        yy = xx.T
        z = np.ones((len(xx),len(yy)))
        for i in range(len(xx)):
            for j in range(len(yy)):
                xi = []
                tauex = []
                tautemp = []
                np.mat(xi.append([1.0,xx[i,j],yy[i,j]]))
                hi_input = self.hi_wb*np.mat(xi).T
                hi_out = self.logistic(hi_input)
                taumrow ,taucol = hi_out.shape
                tauex = np.mat(np.ones((1,taumrow+1)))
                tauex[:,1:] = (hi_out.T)[:,0:taumrow]
                out_input = self.out_wb*(np.mat(tauex).T)
                out = self.logistic(out_input)
                z[i,j] = out
        return x,z

    def classfyLine(self,plt,x,z):
        plt.contour(x,x,z,1,colors='black')

    def TrendLine(self,plt,color='g'):
        X = np.linspace(0,self.maxiter,self.maxiter)
        Y = np.log2(self.errlist)
        plt.plot(X,Y,color)












