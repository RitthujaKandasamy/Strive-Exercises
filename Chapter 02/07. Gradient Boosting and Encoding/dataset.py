import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer


hello = "<3"

def get_data(pth):

    data = pd.read_csv(pth)
    print(data[:3])
    
    # split
    x_train, x_test, y_train, y_test = train_test_split(data.values[:, :-1], data.values[:, -1], test_size = 0.2, random_state = 0)


    # preprocessing data
    # OrdinalEncoding and Standardscaler
    ct = ColumnTransformer( [('ordinal', OrdinalEncoder(handle_unknown= 'use_encoded_value', unknown_value = -1) , [1, 4, 5] ), 
                             ('scaler', StandardScaler() , [0, 2] )], 
                                remainder = 'passthrough')

    
    # fit and transform
    x_train = ct.fit_transform(x_train)
    x_test = ct.transform(x_test)
    

    return x_train, x_test, y_train, y_test