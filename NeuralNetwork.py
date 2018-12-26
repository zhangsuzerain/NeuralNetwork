# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 11:01:10 2018

@author: zhangyawei_vendor
"""
"""
Spyder Editor

This is a temporary script file.
"""

#!/usr/bin/python   
# -*- coding: utf-8-*-
# #Neural Network算法
import numpy as np

def tanh(x):    
    return np.tanh(x)

# #tanh()导数
def tanh_deriv(x):
    return 1.0 -np.tanh(x)*np.tanh(x)

def logistic(x):
    return 1/(1 + np.exp(-x))

# #logistic()导数
def logistic_derivative(x):
    return logistic(x)*(1 - logistic(x))

# #定义类NeuralNetwork
class NeuralNetwork:
    # #__init__ 构造函数   self相当于this当前类的指针
    def __init__(self, layers, activation = 'tanh'):
        """
        :param layers:a list 包含几个数表示有几层；数值表示对应层的神经元个数；
        :param activation: 指定函数；默认为tanh函数；
        """
        if activation == 'logistic':
            self.activation = logistic
            self.activation_deriv = logistic_derivative
        elif activation == 'tanh':
            self.activation = tanh
            self.activation_deriv = tanh_deriv

        self.weights = []  # 初始化weights
        for i in range(1, len(layers)-1):
            self.weights.append((2*np.random.random((layers[i - 1] + 1,layers[i] + 1)) - 1)*0.25)
            self.weights.append((2*np.random.random((layers[i] + 1,layers[i + 1])) - 1)*0.25)

    # #X为数据集，每一行为一个实例；y为label；抽取X中的一个样本进行更新，循环epochs次
    def fit(self, X, y, learning_rate=0.2, epochs=10000):
        # #确定X至少是二维的数据
        X = np.atleast_2d(X)
        # #初始化temp为1；与X同行数，列数比X多1
        temp = np.ones([X.shape[0], X.shape[1]+1])
        temp[:, 0:-1] = X
        X = temp
        # #转换y的数据类型
        y = np.array(y)

        # #每次循环从X中随机抽取1行，对神经网络进行更新
        for k in range(epochs):
            i = np.random.randint(X.shape[0])
            # #取X中的1行，即一个实例放入a中
            a = [X[i]]

            for l in range(len(self.weights)):
                # # a与weights内积
                a.append(self.activation(np.dot(a[l], self.weights[l])))
            # #实际的标记值减去标签值
            error = y[i] - a[-1]
            deltas = [error * self.activation_deriv(a[-1])]

            # #反向
            for l in range(len(a) - 2, 0, -1):
                deltas.append(deltas[-1].dot(self.weights[l].T)*self.activation_deriv(a[l]))

            deltas.reverse()
            for i in range(len(self.weights)):
                layer = np.atleast_2d(a[i])
                # #权重的更新
                delta =np.atleast_2d(deltas[i])
                self.weights[i] += learning_rate * layer.T.dot(delta)

    def predict(self,x):
        x = np.array(x)
        temp = np.ones(x.shape[0] + 1)
        temp[0:-1] = x
        a = temp
        for l in range(0, len(self.weights)):
            a = self.activation(np.dot(a,self.weights[l]))
        return a
