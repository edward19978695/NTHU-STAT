#!/usr/bin/env python
# coding: utf-8

# In[7]:


import numpy as np

class test_NN():
    def __init__(self,params,activation):
        self.params = params
        self.cache = dict()
        
        if activation == "relu":
            self.activation = self.relu
        elif activation == "sigmoid":
            self.activation = self.sigmoid
        elif activation == "hard_tanh":
            self.activation = self.hard_tanh
        elif activation == "tanh":
            self.activation = self.tanh
        
    def accuracy(self, y, output):
        return np.mean(np.argmax(y, axis=-1) == np.argmax(output.T, axis=-1))
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def sigmoid(self, x):
        return 1/(1+np.exp(-x))

    def hard_tanh(self, x):
        return np.maximum(np.minimum(x,1),-1)
    
    def tanh(self,x):
        t = (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
        return t
    
    def softmax(self,x):
        # Numerically stable with large exponentials
        exps = np.exp(x - x.max())
        return exps/np.sum(exps, axis=0)
    
    def feed_forward(self, x):
        self.cache["X"] = x
        self.cache["Z1"] = np.matmul(self.params["W1"], self.cache["X"].T) + self.params["b1"]
        self.cache["A1"] = self.activation(self.cache["Z1"])
        self.cache["Z2"] = np.matmul(self.params["W2"], self.cache["A1"]) + self.params["b2"]
        self.cache["A2"] = self.activation(self.cache["Z2"])
        self.cache["Z3"] = np.matmul(self.params["W3"], self.cache["A2"]) + self.params["b3"]
        self.cache["A3"] = self.softmax(self.cache["Z3"])
        return self.cache["A3"]
    
    def cross_entropy_loss(self, y, output):
        l_sum = np.sum(np.multiply(y.T, np.log(output)))
        m = y.shape[0]
        l = -(1/m) * l_sum
        return l
    
    def test(self, x_test, y_test):
        output = self.feed_forward(x_test)
        test_acc = self.accuracy(y_test, output)
        test_loss = self.cross_entropy_loss(y_test, output)
        print("test acc={:.4f}, test loss={:.4f}".format(test_acc, test_loss))



# Load testing data
dataset = np.load("data_set.npy.npz")
x_test, y_test = dataset["x3"], dataset["y3"]

# Load model weights
weight = np.load("weight.npy",allow_pickle='TRUE').item()

# Print test accuracy and Cross-entropy loss
# (you have to type in which act function had been used while training)
result = test_NN(weight, activation="relu")
result.test(x_test,y_test)

