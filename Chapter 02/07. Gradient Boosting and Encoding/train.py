import dataset as dh
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.model_selection import cross_validate



x_train, x_test, y_train, y_test = dh.get_data("C:\\Users\\ritth\\code\\Strive\\Strive-Exercises\\Chapter 02\\07. Gradient Boosting and Encoding\\insurance.csv")


# check shape 
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)


rf = RandomForestRegressor()
ada = AdaBoostRegressor()
gb = GradientBoostingRegressor()
xgb = XGBRegressor()


# create RF, Ada, GB, XGB in one line
models = [rf, ada, gb, xgb]


# fit and evaluating the performance
for train_model in models:
    fit = train_model.fit(x_train, y_train)
    score = train_model.score(x_test, y_test)
    cv = cross_validate(train_model, x_train, y_train)

    print('{} : \n Score = {}, \n Crossvalidation = {} \n'.format(fit, score, cv))
    print('Mean train cross validation score {} \n'.format(cv['test_score'].mean()))