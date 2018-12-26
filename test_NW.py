# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 11:04:10 2018

@author: zhangyawei_vendor
"""

from NeuralNetwork import NeuralNetwork
import numpy as np

# #2个输入，2层神经网络，1个输出
nn = NeuralNetwork([2,2,1],'tanh')
X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([0, 1, 1, 0])
nn.fit(X, y)
for i in [[0,0],[0,1],[1,0],[1,1]]:
    print (i, nn.predict(i))
