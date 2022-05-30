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
x_train, x_test, y_train, y_test = dh.load_data(pth)

x_train, x_test, y_train, y_test = dh.to_batches(x_train, x_test, y_train, y_test, batch_size= 15)

#print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)



# import model
model = m.NeuralNetwork(x_train.shape[2], hidden_sizes= [600, 400, 200, 100], output_size= 1)

#print(model)




# train
def train(x_train, y_train, x_test, y_test, num_epochs, model, lr, print_every):

    optimizer = torch.optim.Adam(model.parameters(), lr)
    criterion = nn.L1Loss()             # L1 is mean absoult error used in small value errors
    trainloss = []
    testloss = []

    for epoch in range(num_epochs):

        epoch_list = []
        running_loss = 0

        for  i, (x_train_batches, y_train_batches) in enumerate(zip(x_train, y_train)):
            model.train()              # model in training model
            y_pred = model(x_train_batches)
            loss = criterion(y_pred, y_train_batches)
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

            test_epoch_list = []
            acc = []
            

            for j, (x_test_batches,y_test_batches) in enumerate(zip(x_test, y_test)):
                test_pred = model(x_test_batches)
                #preds = model.view(len(y_test_batches))
                test_loss = criterion(test_pred, y_test_batches)

                # store loss
                test_epoch_list.append(test_loss.item())
                n_correct = torch.sum((torch.abs(x_test_batches[0] - y_test_batches[0]) < torch.abs(10 * y_test_batches[0])))
                acc.append(n_correct.item() * 100.0 / len(y_test_batches[0]))

        mean_epoch_losses_test = sum(test_epoch_list)/len(test_epoch_list)
        testloss.append(mean_epoch_losses_test)

        

    #print(f'Epoch: {epoch + 1}, train loss: {trainloss:.2f}, test loss: {test_losses:.2f}')



# plot
    plt.title("Loss for Train and Test")
    plt.plot(trainloss, marker = 'o', label = 'Train Loss')
    plt.plot(testloss, marker = 'o', label = 'Test Loss')
    #plt.plot(acc, marker = 'o', label = 'Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


train(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, num_epochs = 50, model=model, lr= 0.01, print_every= 40)