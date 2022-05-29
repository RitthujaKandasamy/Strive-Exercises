from operator import index
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split

torch.manual_seed(0)


# load data for model
def load_data(pth):
    
    # read data from pth
    data = pd.read_csv(pth)
    # print(data)
    # print('**************************************** \n')
    # print(f'Shape: {data.shape} \n')
    # print(data.info)
    # print('**************************************** \n')

    # drop date column from data
    data.drop(['date'], axis = 1, inplace = True)
    

    # create x and y, with target 'DAX'
    y = data['DAX'].values
    x = data.drop(['DAX'], axis = 1).values
    # print('X data without Target \n')
    # print(x)
    # print('**************************************** \n')

    
    # split data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.1, random_state = 0)
    #print(f'x_train, x_test, y_train, y_test : {x_train.shape, x_test.shape, y_train.shape, y_test.shape} \n') 
    # x_train, x_test, y_train, y_test : ((482, 8), (54, 8), (482,), (54,))

    
    # convert numpy into tensor
    x_train = torch.tensor(x_train.astype(np.float32))
    x_test = torch.tensor(x_test.astype(np.float32))

    y_train = torch.tensor(y_train.astype(np.float32))
    y_test = torch.tensor(y_test.astype(np.float32))
    #print(x_train)


    return x_train, x_test, y_train, y_test


#load_data("C:\\Users\\ritth\\code\\Strive\\Strive-Exercises\\Chapter 03\\03. MLP Regression\\data\\turkish_stocks.csv")




#(100, 7) -> (10,10,7)
#(100, 1) -> (10,10,1)


def to_batches(x_train, x_test, y_train, y_test, batch_size):


    n_batches = x_train.shape[0] // batch_size # 482 / 64 = 7.53125 --> 7 
    n_batches_test = x_test.shape[0] // batch_size
    #print(n_batches)


    indexes = np.random.permutation(x_train.shape[0])
    indexes_test = np.random.permutation(x_test.shape[0])
    #print(indexes)


    x_train = x_train[indexes]
    y_train = y_train[indexes]

    x_test = x_test[indexes_test]
    y_test = y_test[indexes_test]



    x_train = x_train[ :batch_size * n_batches ].reshape(n_batches, batch_size, x_train.shape[1])
    y_train = y_train[ :batch_size * n_batches ].reshape(n_batches, batch_size, 1)
    #print(x_train)
    
    x_test = x_test[ :batch_size * n_batches_test ].reshape(n_batches_test, batch_size, x_test.shape[1])
    y_test = y_test[ :batch_size * n_batches_test ].reshape(n_batches_test, batch_size, 1)


    return x_train, x_test, y_train, y_test


#to_batches("C:\\Users\\ritth\\code\\Strive\\Strive-Exercises\\Chapter 03\\03. MLP Regression\\data\\turkish_stocks.csv", 64)