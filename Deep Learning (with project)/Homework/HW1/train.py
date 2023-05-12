#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt

class ShallowNeuralNetwork():
    def __init__(self, sizes, epochs, activation="sigmoid"):
        self.sizes = sizes
        self.epochs = epochs
    
        # Choose activation function
        if activation == "relu":
            self.activation = self.relu
        elif activation == "sigmoid":
            self.activation = self.sigmoid
        elif activation == "hard_tanh":
            self.activation = self.hard_tanh
        elif activation == "tanh":
            self.activation = self.tanh

        # Save all weights
        self.params = self.initialize()
        # Save all intermediate values, i.e. activations
        self.cache = {}
        # Save all train_acc for each epoch
        self.train_acc = []
        # Save all train_loss for each epoch
        self.train_loss = []
        # Save all val_acc for each epoch
        self.val_acc = []
        # Save all val_loss for each epoch
        self.val_loss = []
    
   
    
    # Different kinds of activation function and their derivative(will be used in backpropagation)
    def relu(self, x, derivative=False):
        if derivative:
            x = np.where(x<0, 0, 1)
            #x = np.where(x>=0, 1, x)
            return x
        return np.maximum(0, x)
  

    def sigmoid(self, x, derivative=False):
        if derivative:
            return (np.exp(-x))/((np.exp(-x)+1)**2)
        return 1/(1+np.exp(-x))
    
    def tanh(self, x, derivative=False):
        t = (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
        if derivative:
            return 1-t**2
        return t


    def hard_tanh(self, x, derivative=False):
        if derivative:
            x = np.where((-1<=x)&(x<1),1,0)
            return x
        return np.maximum(np.minimum(x,1),-1)

  

    def softmax(self, x):
        # Numerically stable with large exponentials
        exps = np.exp(x - x.max())
        return exps/np.sum(exps, axis=0)
  

    #Initialize
    def initialize(self):
        # number of nodes in each layer
        input_layer = self.sizes[0]
        hidden_layer1 = self.sizes[1]
        hidden_layer2 = self.sizes[2]
        output_layer = self.sizes[3]

        params = {
            "W1" : np.random.randn(hidden_layer1, input_layer) * np.sqrt(1/input_layer), 
            "b1" : np.zeros((hidden_layer1, 1)), 
            "W2" : np.random.randn(hidden_layer2, hidden_layer1) * np.sqrt(1/hidden_layer1),
            "b2" : np.zeros((hidden_layer2, 1)),
            "W3" : np.random.randn(output_layer, hidden_layer2) * np.sqrt(1/hidden_layer2), 
            "b3" : np.zeros((output_layer, 1))
        }
        return params

    def initialize_momentum_optimizer(self):
        momemtum_opt = {
            "W1": np.zeros(self.params["W1"].shape),
            "b1": np.zeros(self.params["b1"].shape),
            "W2": np.zeros(self.params["W2"].shape),
            "b2": np.zeros(self.params["b2"].shape),
            "W3": np.zeros(self.params["W3"].shape),
            "b3": np.zeros(self.params["b3"].shape)
        }
        return momemtum_opt
  

    # Feedforward
    def feed_forward(self, x):
        self.cache["X"] = x
        self.cache["Z1"] = np.matmul(self.params["W1"], self.cache["X"].T) + self.params["b1"]
        self.cache["A1"] = self.activation(self.cache["Z1"])
        self.cache["Z2"] = np.matmul(self.params["W2"], self.cache["A1"]) + self.params["b2"]
        self.cache["A2"] = self.activation(self.cache["Z2"])
        self.cache["Z3"] = np.matmul(self.params["W3"], self.cache["A2"]) + self.params["b3"]
        self.cache["A3"] = self.softmax(self.cache["Z3"])
        return self.cache["A3"]
  

    # Backpropagation
    def back_propagate(self, y, output):
        current_batch_size = y.shape[0]

        dZ3 = output - y.T
        dW3 = (1/current_batch_size) * np.matmul(dZ3, self.cache["A2"].T)
        db3 = (1/current_batch_size) * np.sum(dZ3, axis=1, keepdims=True)

        dA2 = np.matmul(self.params["W3"].T, dZ3)
        dZ2 = dA2 * self.activation(self.cache["Z2"], derivative=True)
        dW2 = (1/current_batch_size) * np.matmul(dZ2, self.cache["A1"].T)
        db2 = (1/current_batch_size) * np.sum(dZ2, axis=1, keepdims=True)

        dA1 = np.matmul(self.params["W2"].T, dZ2)
        dZ1 = dA1 * self.activation(self.cache["Z1"], derivative=True)
        dW1 = (1/current_batch_size) * np.matmul(dZ1, self.cache["X"])
        db1 = (1/current_batch_size) * np.sum(dZ1, axis=1, keepdims=True)

        self.grads = {"W1":dW1, "b1":db1, "W2":dW2, "b2":db2, "W3":dW3, "b3":db3}
        return self.grads
  
    # Compute cross entropy loss
    def cross_entropy_loss(self, y, output):
        l_sum = np.sum(np.multiply(y.T, np.log(output)))
        m = y.shape[0]
        l = -(1/m) * l_sum
        return l
  
    # Choose the optimizing method
    def optimize(self, l_rate=0.1, beta=0.9):
        if self.optimizer == "sgd":
            for key in self.params:
                self.params[key] = self.params[key] - l_rate*self.grads[key]

        elif self.optimizer == "momentum":
            for key in self.params:
                self.momentum_opt[key] = (beta*self.momentum_opt[key] + (1-beta)*self.grads[key])
                self.params[key] = self.params[key] - l_rate*self.momentum_opt[key]
  

    # Compute accuracy
    def accuracy(self, y, output):
        return np.mean(np.argmax(y, axis=-1) == np.argmax(output.T, axis=-1))
  

    # Training
    def train(self, x_train, y_train, x_val, y_val, 
              batch_size=64, optimizer="momentum", l_rate=0.1, beta=0.9):
        # Hyperparameters
        self.batch_size = batch_size
        num_batches = -(-x_train.shape[0] // self.batch_size)

        # Initialize optimizer
        self.optimizer = optimizer
        if self.optimizer == "momentum":
            self.momentum_opt = self.initialize_momentum_optimizer()


        template = "Epoch {}: train acc={:.3f}, train loss={:.3f}, val acc={:.3f}, val loss={:.3f}"

        # Train
        for i in range(self.epochs):
            # Shuffle
            permutation = np.random.permutation(x_train.shape[0])
            x_train_shuffled = x_train[permutation]
            y_train_shuffled = y_train[permutation]

            for j in range(num_batches):
                # Batch
                begin = j * self.batch_size
                end = min(begin + self.batch_size, x_train.shape[0]-1)
                x = x_train_shuffled[begin:end]
                y = y_train_shuffled[begin:end]

                # Forward
                output = self.feed_forward(x)
                # Backprop
                grad = self.back_propagate(y, output)
                # Optimize
                self.optimize(l_rate=l_rate, beta=beta)

            # Evaluate performance
            # Training data
            output = self.feed_forward(x_train)
            train_acc = self.accuracy(y_train, output)
            self.train_acc.append(train_acc)
            train_loss = self.cross_entropy_loss(y_train, output)
            self.train_loss.append(train_loss)
            # Validation data
            output = self.feed_forward(x_val)
            val_acc = self.accuracy(y_val, output)
            self.val_acc.append(val_acc)
            val_loss = self.cross_entropy_loss(y_val, output)
            self.val_loss.append(val_loss)

            print(template.format(i+1, train_acc, train_loss, val_acc, val_loss))

    # Acc and Loss of test data
    def test(self, x_test, y_test):
        output = self.feed_forward(x_test)
        test_acc = self.accuracy(y_test, output)
        test_loss = self.cross_entropy_loss(y_test, output)
        print("test acc={:.4f}, test loss={:.4f}".format(test_acc, test_loss))
       

    
# Load dataset from data_set.npy.npz
dataset = np.load("data_set.npy.npz")
x_train, y_train = dataset["x1"], dataset["y1"]
x_val, y_val = dataset["x2"], dataset["y2"]
x_test, y_test = dataset["x3"], dataset["y3"]


np.random.seed(11031)
'''
    You can control hyperparameter (except the number of layers), 
    the activation function and the optimizing method here 
'''
snn = ShallowNeuralNetwork(sizes=[784,600,400,10], epochs=500, activation="relu")
snn.train(x_train, y_train, x_val, y_val, batch_size=500, optimizer="momentum", l_rate=0.01, beta=0.8)

n_iter = len(snn.train_acc)

# visualization loss and accuracy
plt.plot(range(1, n_iter+1), snn.train_loss, label = 'Training', color = 'blue', linewidth = 1)
plt.plot(range(1, n_iter+1), snn.val_loss, label = 'Validation', color = 'red', linewidth = 1)
plt.legend(loc='upper right')
plt.xlabel('Model complexity (Epoch)')
plt.ylabel('Cross Entropy Loss')
plt.tight_layout()
# plt.savefig('fig1.png', dpi=300)
plt.show()

plt.plot(range(1, n_iter+1)[100:], snn.train_loss[100:], label = 'Training', color = 'blue', linewidth = 1)
plt.plot(range(1, n_iter+1)[100:], snn.val_loss[100:], label = 'Validation', color = 'red', linewidth = 1)
plt.legend(loc='upper right')
plt.xlabel('Model complexity (Epoch)')
plt.ylabel('Cross Entropy Loss')
plt.tight_layout()
# plt.savefig('fig2.png', dpi=300)
plt.show()

plt.plot(range(1, n_iter+1), snn.train_acc, label = 'Training', color = 'blue', linewidth = 1)
plt.plot(range(1, n_iter+1), snn.val_acc, label = 'Validation', color = 'red', linewidth = 1)
plt.hlines(y=0.87, xmin=1, xmax=n_iter, color='gray', linewidth=1, linestyle='--')
plt.legend(loc='lower right')
plt.xlabel('Model complexity (Epoch)')
plt.ylabel('Accuracy')
plt.tight_layout()
# plt.savefig('fig3.png', dpi=300)
plt.show()

plt.plot(range(1, n_iter+1)[100:], snn.train_acc[100:], label = 'Training', color = 'blue', linewidth = 1)
plt.plot(range(1, n_iter+1)[100:], snn.val_acc[100:], label = 'Validation', color = 'red', linewidth = 1)
plt.hlines(y=0.87, xmin=100, xmax=n_iter, color='gray', linewidth=1, linestyle='--')
plt.legend(loc='lower right')
plt.xlabel('Model complexity (Epoch)')
plt.ylabel('Accuracy')
plt.tight_layout()
# plt.savefig('fig4.png', dpi=300)
plt.show()

# Save the model weight as .npy file
weight = snn.params
np.save("weight.npy",weight)

