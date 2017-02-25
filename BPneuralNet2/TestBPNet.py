# -*- coding:utf-8 -*-
import operator
from BPNet import *

bpnet = BPnet()
# testSet2.txt has 307 rows,3 columns,last columns is labels
bpnet.loadDataset('testSet2.txt')
bpnet.dataMat = bpnet.normalize(bpnet.dataMat)

bpnet.drawClassScatter(plt)

bpnet.bpTrain()
print bpnet.hi_wb
print bpnet.out_wb

x,z = bpnet.BPClassfier(-3.0,3.0)
bpnet.classfyLine(plt,x,z)
plt.show()

bpnet.TrendLine(plt)
plt.show()