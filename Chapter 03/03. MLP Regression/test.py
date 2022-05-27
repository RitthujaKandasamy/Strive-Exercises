#import the needed libraries

import data_handler as dh        
import torch
import train as t
import torch.nn as nn
import model as m
import matplotlib.pyplot as plt
import numpy as np



# put pth and batchsize
pth = "C:\\Users\\ritth\\code\\Strive\\Strive-Exercises\\Chapter 03\\03. MLP Regression\\data\\turkish_stocks.csv"
x_train, x_test, y_train, y_test, x = dh.to_batches(pth, batch_size = 64)


# import model
model = m.NeuralNetwork(x.shape[1], 15, 1)
train_model = t.train(x_train=x_train, y_train=y_train, num_epochs = 200, model=model, lr= 0.01, print_every= 40, batch= 64)


# test


