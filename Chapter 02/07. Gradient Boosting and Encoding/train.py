import data_handler as dh
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.model_selection import cross_val_score


# split data
x_train, x_test, y_train, y_test = dh.get_data("C:\\Users\\ritth\\code\\Strive\\Strive-Exercises\\Chapter 02\\07. Gradient Boosting and Encoding\\insurance.csv")

print(x_train, y_train, x_test, y_test)


# check with RF, Ada, GB
models = [RandomForestRegressor(), GradientBoostingRegressor(), AdaBoostRegressor()]


for model in models:
    print(cross_val_score(model, x_train, y_train).mean())