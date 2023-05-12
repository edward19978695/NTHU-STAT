# -*- coding: utf-8 -*-
"""AlexNet.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1isuxEF1vLD5SosLIe1r-JH7QCdy7rz96
"""

import h5py
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

# Load training data
# hf_train = h5py.File('/content/drive/MyDrive/DeepLearning/Homework/HW3/Dataset/Signs_Data_Training.h5', 'r')
hf_train = h5py.File('Dataset/Signs_Data_Training.h5', 'r')
train_image = np.array(hf_train["train_set_x"])
train_label = np.array(hf_train["train_set_y"])

# Data pre-process
onehotencoder = OneHotEncoder()
train_label = onehotencoder.fit_transform(train_label.reshape(-1, 1)).toarray()
train_image = torch.from_numpy(train_image).permute(0,3,1,2)
train_image = 2*(train_image/255) - 1

# 5-fold CV
cv_5_fold = KFold(n_splits=5, shuffle=True, random_state=1)
train_image_cv = dict() ; train_label_cv = dict()
val_image_cv = dict() ; val_label_cv = dict()
for i, (train_index, val_index) in enumerate(cv_5_fold.split(train_image,train_label)):
  key = "f"+str(i+1)
  train_image_cv[key] = train_image[train_index]
  train_label_cv[key] = train_label[train_index]
  val_image_cv[key] = train_image[val_index]
  val_label_cv[key] = train_label[val_index]

# Accuracy function
def cal_accuracy(prediction, label):
    ''' Calculate Accuracy, please don't modify this part
        Args:
            prediction (with dimension N): Predicted Value
            label  (with dimension N): Label
        Returns:
            accuracy:　Accuracy
    '''

    accuracy = 0
    number_of_data = len(prediction)
    for i in range(number_of_data):
        accuracy += float(prediction[i] == label[i])
    accuracy = (accuracy / number_of_data) * 100

    return accuracy

## wrap train dataset
class Train_Loader(Dataset):
    def __init__(self, img_arr, label_arr):
        '''
            define img_arr, label_arr
        '''
        self.img_arr = img_arr
        self.label_arr = label_arr

    def __len__(self):
        '''
            return length
        '''
        return len(self.img_arr)

    def __getitem__(self, index):
        '''
            process each img and label
        '''
        img = self.img_arr[index]
        label = self.label_arr[index]
        return img,label

# Load AlexNet model
AlexNet_model = torch.hub.load('pytorch/vision:v0.6.0', 'alexnet', pretrained=True)

#Updating the second classifier
AlexNet_model.classifier[6] = nn.Linear(4096,6)

# "cuda" only when GPUs are available.
device = "cuda" if torch.cuda.is_available() else "cpu"

# The number of training epochs and patience.
n_epochs = 50

# For the classification task, we use cross-entropy as the measurement of performance.
criterion = nn.CrossEntropyLoss()

batch_size = 128

# Train final model
train_set = Train_Loader(train_image, train_label)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
# Initialize a model, and put it on the device specified.
model = AlexNet_model.to(device)

# Initialize optimizer, you may fine-tune some hyperparameters such as learning rate on your own.
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)

for epoch in range(n_epochs):
    # ---------- Training ----------
    # Make sure the model is in train mode before training.
    model.train()

    # These are used to record information in training.
    train_loss = []
    train_accs = []
    for batch in tqdm(train_loader):

        # A batch consists of image data and corresponding labels.
        imgs, labels = batch

        # Forward the data. (Make sure data and model are on the same device.)
        logits = model(imgs.to(device))

        # Calculate the cross-entropy loss.
        loss = criterion(logits, labels.to(device))

        # Gradients stored in the parameters in the previous step should be cleared out first.
        optimizer.zero_grad()

        # Compute the gradients for parameters.
        loss.backward()

        # Clip the gradient norms for stable training.
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)

        # Update the parameters with computed gradients.
        optimizer.step()

        # Compute the accuracy for current batch.
        acc = cal_accuracy(np.argmax(logits.cpu().data.numpy(), axis=-1),np.argmax(labels.cpu(), axis=-1))

        # Record the loss and accuracy.
        train_loss.append(loss.item())
        train_accs.append(acc)

    train_loss = sum(train_loss) / len(train_loss)
    train_acc = sum(train_accs) / len(train_accs)

    # Print the information.
    print(f"[ Train | {epoch + 1:03d}/{n_epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")

    # save models
    if epoch == n_epochs-1:
      torch.save(model, "AlexNet_weights.pth") # save last model

# Load in testing image and label
# hf_test = h5py.File("/content/drive/MyDrive/DeepLearning/Homework/HW3/Dataset/Signs_Data_Testing.h5", 'r')
hf_test = h5py.File("Dataset/Signs_Data_Testing.h5", 'r')
test_image = np.array(hf_test["test_set_x"])
test_image = torch.from_numpy(test_image).permute(0,3,1,2)
test_image = 2*(test_image/255) - 1  # mapping to [-1,1]
test_label = np.array(hf_test["test_set_y"])

# Compute predicted probability
pred_prob = model(test_image.cuda()).cpu().detach().numpy()
pred_label = np.argmax(pred_prob, axis=1) # predicted label

# Check prediction performance
test_acc = cal_accuracy(pred_label, test_label)
print(f"Test accuracy : {test_acc:.2f}%")