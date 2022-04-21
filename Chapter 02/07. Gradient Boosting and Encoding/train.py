import dataset as dh
import numpy as np
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.model_selection import cross_validate, GridSearchCV



x_train, x_test, y_train, y_test, ct = dh.get_data("C:\\Users\\ritth\\code\\Strive\\Strive-Exercises\\Chapter 02\\07. Gradient Boosting and Encoding\\insurance.csv")
   


class Insurance:


     """
       Create a class called Insurance that has following attributes.

       Attributes: x_train, x_test, y_train, y_test
    """



     def __init__(self, x_train, x_test, y_train, y_test):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        

    
     def model_shape(self):
        X_train = self.x_train.shape
        X_test = self.x_test.shape
        Y_train = self.y_train.shape
        Y_test = self.y_test.shape
        
        #print(x_train[:3])
        #print(x_test[:3])

        return(X_train, X_test, Y_train, Y_test)




class Sub(Insurance):


 
    """
       Create a class called Sub that has following attributes.

       Attributes: x_train, x_test, y_train, y_test, fits, predictions, score, cv, accuracy
    """



    def __init__(self, x_train, x_test, y_train, y_test):
        super().__init__(x_train, x_test, y_train, y_test)
        


    def train_test(self):
        

        rf_reg = RandomForestRegressor()
        ada_reg = AdaBoostRegressor()
        gb_reg = GradientBoostingRegressor()
        xgb_reg = XGBRegressor()


        # create RF, Ada, GB, XGB in one line
        self.models = [rf_reg, ada_reg, gb_reg, xgb_reg]


        
        for train_model in self.models:
            self.fits = train_model.fit(x_train, y_train)
            self.predictions = train_model.predict(x_test)
            self.score = train_model.score(x_test, y_test)
            
            

            # check crossvalidation for better result
            self.cv = cross_validate(train_model, x_train, y_train)
            

            # dt = train_model.get_params().keys()
            parameters = {
                            'n_estimators':[50, 100, 100, 50],
                            'random_state': [0, 0, 0, 0],
                         }
                    
            grd = GridSearchCV(train_model, parameters)
            grd.fit(x_train, y_train)
            self.accuracy = grd.best_score_
            

            #print('{} : \n Predication = {}, \n Score = {}, \n Crossvalidation = {}, \n Gridaccuracy = {} \n'.format(self.fits, self.predictions[:3], self.score, self.cv, self.accuracy))
            #print('Mean train cross validation score {} \n'.format(self.cv['test_score'].mean()))
            print("model name: {} \n, best model: {} \n".format(self.fits, self.accuracy))
            

        return rf_reg, ada_reg, gb_reg, ct, self.models


    
    def tune_model(self):


        # selected best model, gb accuracy is 84.9 
        gb_reg_model = GradientBoostingRegressor()

              
        #  hyparamater tuning
        params  = {   
                        'n_estimators': np.linspace(50, 70, 100),
                         'max_features': np.linspace(3, 6).astype(int),   
                       'max_depth': np.linspace(3, 7).astype(int), 
                       'learning_rate': np.linspace(0.001, 0.1, 10)
                    }



        grid = GridSearchCV(gb_reg_model, params)
        grid.fit(x_train, y_train)


        print(f'grid best score: {grid.best_score_}') 
        print(f'grid best parameters: {grid.best_params_}')

    
   




