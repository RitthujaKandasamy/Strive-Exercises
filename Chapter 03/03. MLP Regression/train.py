#import the needed libraries

from cProfile import label
import data_handler as dh        
import torch
#import torch.optim as optim
import torch.nn as nn
import model as m
import matplotlib.pyplot as plt
import numpy as np




# put pth and batchsize
pth = "C:\\Users\\ritth\\code\\Strive\\Strive-Exercises\\Chapter 03\\03. MLP Regression\\data\\turkish_stocks.csv"
x_train, x_test, y_train, y_test, x = dh.to_batches(pth, batch_size = 64)


# import model
model = m.NeuralNetwork(x.shape[1], 10, 1)


# train
def train(x_train, y_train, x_test, y_test, num_epochs, model, lr, print_every, batch):

    optimizer = torch.optim.Adam(model.parameters(), lr)
    criterion = nn.MSELoss()
    trainloss = []
    testloss = []

    for epoch in range(num_epochs):

        epoch_list = []
        running_loss = 0

        for i in range(x_train.shape[1]):
            model.train()              # model in training model
            y_pred = model.forward(x_train)
            loss = criterion(y_pred, y_train)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            # store loss
            epoch_list.append(loss.item())
            running_loss += loss.item()

            if i % print_every == 0:
                #print(f'epoch: {epoch + 1} | loss: {running_loss/print_every}')

                running_loss = 0
        mean_epoch_losses = sum(epoch_list)/len(epoch_list)
        trainloss.append(mean_epoch_losses)

        
        # test
        model.eval()
        with torch.no_grad():
            test_pred = model.forward(x_test)
            test_loss = criterion(test_pred, y_test)
            testloss.append(test_loss.item())
        #model.train()

    #print(f'Epoch: {epoch + 1}, train loss: {trainloss:.2f}, test loss: {test_losses:.2f}')


    plt.plot(trainloss, label = 'Train Loss')
    plt.plot(testloss, label = 'Test Loss')
    plt.legend()
    plt.show()


train(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, num_epochs = 200, model=model, lr= 0.01, print_every= 40, batch= 10)