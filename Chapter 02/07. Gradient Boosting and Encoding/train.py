import dataset as dh
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.model_selection import cross_validate, GridSearchCV



x_train, x_test, y_train, y_test, ct = dh.get_data("C:\\Users\\ritth\\code\\Strive\\Strive-Exercises\\Chapter 02\\07. Gradient Boosting and Encoding\\insurance.csv")


# check shape 
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)


def train(rf, ada, gb, xgb):


    rf = RandomForestRegressor()
    ada = AdaBoostRegressor()
    gb = GradientBoostingRegressor()
    xgb = XGBRegressor()


    # create RF, Ada, GB, XGB in one line
    model = [rf, ada, gb, xgb]


    # checking performance
    for train_model in model:
        fit = train_model.fit(x_train, y_train)
        predictions = train_model.predict(x_test)
        score = train_model.score(x_test, y_test)
        cv = cross_validate(train_model, x_train, y_train)


        # dt = train_model.get_params().keys()
        parameters = {
                    'n_estimators':[50, 100, 100, 50],
                        'random_state': [0, 0, 0, 0]
                }
                
        grd = GridSearchCV(train_model, parameters)
        grid_train = grd.fit(x_train, y_train)
        accuracy = grd.best_score_


        print('{} : \n Predication = {}, \n Score = {}, \n Crossvalidation = {}, \n Gridaccuracy = {} \n'.format(fit, predictions[:3], score, cv, accuracy))
        print('Mean train cross validation score {} \n'.format(cv['test_score'].mean()))

    
    
    return model, score, predictions, accuracy