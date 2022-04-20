from train import Insurance
from train import Sub
import dataset as dh
import numpy as np 
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV



x_train, x_test, y_train, y_test, ct = dh.get_data("C:\\Users\\ritth\\code\\Strive\\Strive-Exercises\\Chapter 02\\07. Gradient Boosting and Encoding\\insurance.csv")



insurance_model = Insurance(x_train, x_test, y_train, y_test)
#print(insurance_model.model_shape())


insurance_sub_model = Sub(x_train, x_test, y_train, y_test)
#print(insurance_sub_model.train_test())
#print(insurance_sub_model.tune_model())



# selected best model, gb accuracy is 0.8462032882175 
# output grid best score: 0.8470578323403585
# grid best score: {'learning_rate': 0.05600000000000, 'max_depth': 3}


gb = GradientBoostingRegressor(random_state= 0, learning_rate= 0.056)
gb.fit(x_train, y_train)


best_model = [gb, ct]