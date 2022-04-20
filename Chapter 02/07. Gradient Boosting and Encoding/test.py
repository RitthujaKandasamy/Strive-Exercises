from train import Insurance
from train import Sub
import dataset as dh 


x_train, x_test, y_train, y_test, ct = dh.get_data("C:\\Users\\ritth\\code\\Strive\\Strive-Exercises\\Chapter 02\\07. Gradient Boosting and Encoding\\insurance.csv")



insurance_model = Insurance(x_train, x_test, y_train, y_test)
print(insurance_model.model_shape())



insurance_sub_model = Sub(x_train, x_test, y_train, y_test)
print(insurance_sub_model.train_test())



#best_model = g