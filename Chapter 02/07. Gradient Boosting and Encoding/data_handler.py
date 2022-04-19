import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor


hello = "<3"

def get_data(pth):

    data = pd.read_csv(pth)
    print(data[:3])

    X, y = data.iloc[:, :-1], data.iloc[:, -1]
    print(X[:3])
    print(y[:3])

    # split 
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)



    # OE data, because we want good tree model
    # handle_unknown is used for giving the value for data that is not in the training set, if in further it appear in dataset
    # eg: we trained with some data, but in further we want to train same extract, for that we use it
    # by unknown_value we put the value for that data

    oe = OrdinalEncoder(handle_unknown= 'use_encoded_value', unknown_value = -1)
    scalar = StandardScaler()



    # columntransform is used to transform
    # ordinal and standard are name given to them , we can give any name we like
    ct = ColumnTransformer(('ordinal', oe, ['sex', 'smoker', 'region']), 
                               ('standard', scalar, ['age', 'bmi']),
                                    remainder = 'passthrough')



    # fit dataset
    data_full = ct.fit(data)
    # [:5, :] used to print first 5 full dataset
    print(data_full[:5, :])


    # fit and transform
    x_train = ct.fit_transform(x_train)
    x_test = ct.transform(x_test)
    

    # creating randomforest
    Rf_clf = RandomForestRegressor(random_state=0)
    Rf_clf.fit(x_train,y_train)

    # predict the RF
    pred_rf = Rf_clf.predict(x_test)

    # creating Adaboost
    regr = AdaBoostRegressor(random_state=0, n_estimators=100)
    regr.fit(x_train, y_train)

    # predict Ada
    pred_ada = regr.predict(x_test)

    # creating GB
    GB_clf = GradientBoostingRegressor(random_state=0)
    GB_clf.fit(x_train,y_train)

    # predict Gb
    pred_GB = GB_clf.predict(x_test)

    return x_train, x_test, y_train, y_test
