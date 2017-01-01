# -*- coding:utf-8 -*-
from ID3 import *
tree = ID3Tree()
tree.loadDataSet('../data/dataset.dat',['age','revenue','student','credit'])
tree.train()
tree.saveTree(tree.tree,'../tree/ID3.tree')  # 持久化树
id3 = tree.loadTree('../tree/ID3.tree')     # 加载树模型
result = tree.predict(id3,['age','revenue','student','credit'],['0','1','0','0'])

