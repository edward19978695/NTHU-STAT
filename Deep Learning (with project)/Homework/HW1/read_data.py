#!/usr/bin/env python
# coding: utf-8



import numpy as np
import gzip

#fashion mnist dataset path      
url_train_image = 'Fashion_MNIST_data/train-images-idx3-ubyte.gz'
url_train_labels = 'Fashion_MNIST_data/train-labels-idx1-ubyte.gz'
url_test_image = 'Fashion_MNIST_data/t10k-images-idx3-ubyte.gz'
url_test_labels = 'Fashion_MNIST_data/t10k-labels-idx1-ubyte.gz'

#use gzip open .gz to get ubyte
train_image_ubyte = gzip.open(url_train_image,'r')
test_image_ubyte = gzip.open(url_test_image,'r')
train_label_ubyte = gzip.open(url_train_labels,'r')
test_label_ubyte = gzip.open(url_test_labels,'r')

##START YOUR CODE
train_image_ubyte.read(16)
buf = train_image_ubyte.read(28*28*60000)
x = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
x = x.reshape(60000,28*28)

train_label_ubyte.read(8)
buf = train_label_ubyte.read(60000)
y = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
y = y.reshape(60000)

test_image_ubyte.read(16)
buf = test_image_ubyte.read(28*28* 10000)
x_test = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
x_test = x_test.reshape(10000, 28*28)

test_label_ubyte.read(8)
buf = test_label_ubyte.read(10000)

y_test = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
y_test = y_test.reshape(10000)


# Transform data into one-hot
def one_hot(x, k, dtype=np.float32):
  # Create a one-hot encoding of x of size k
  return np.array(x[:,None] == np.arange(k), dtype)


# Normalize
x /= 255
x_test /= 255

# One-hot for response(label) variable y
num_labels = 10
examples = y.shape[0]
y = one_hot(y.astype("int32"), num_labels)

examples = y_test.shape[0]
y_test = one_hot(y_test.astype("int32"), num_labels)


# Shuffle then Split data X into train and validation set (55000 : 5000)
np.random.seed(10187)
shuffle_index = np.random.permutation(60000)
x, y = x[shuffle_index], y[shuffle_index]
train_size = 55000
val_size = x.shape[0] - train_size
x_train, x_val = x[:train_size], x[train_size:]
y_train, y_val = y[:train_size], y[train_size:]


# Save train, validation, test data sets as .npy.npz file
np.savez("data_set.npy", x1=x_train, y1=y_train, x2=x_val, y2=y_val, x3=x_test, y3=y_test)


