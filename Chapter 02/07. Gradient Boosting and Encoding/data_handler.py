import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor

hello = "<3"

def get_data(pth):

    data = pd.read_csv(pth)

    # splittind the data
    x_train, x_test, y_train, y_test = train_test_split(data.values[:,:-1], data.values[:,-1], test_size=0.2, random_state = 0)

    # OE the new split data
    ct = ColumnTransformer( [('ordinal', OrdinalEncoder(handle_unknown= 'use_encoded_value', unknown_value = -1), [1,4,5] )] )

    # fit and transform
    x_train = ct.fit_transform(x_train)
    x_test = ct.transform(x_test)

    # creating randomforest
    Rf_clf = RandomForestRegressor(random_state=0)
    Rf_clf.fit(x_train,y_train)

    # predict the RF
    predictions = Rf_clf.predict(x_test)



    return x_train, x_test, y_train, y_test, Rf_clf, predictions

get_data('C:\\Users\\ritth\\code\\Strive\\Strive-Exercises\\Chapter 02\\07. Gradient Boosting and Encoding\\insurance.csv')
