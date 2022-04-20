import train as tr
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor




model = tr.train(RandomForestRegressor(), AdaBoostRegressor(), GradientBoostingRegressor(), XGBRegressor())

   
# create new data input 
age = int(input("How old are you? \n"))
sex = str(input("What is your sex? \n"))
child = int(input("How many children do you have? \n"))
smoke = bool(input("Do you smoke? \n"))
bmi = float(input("What is your bmi? \n"))
region = str(input("Choose one of the regions from this: Southwest, Southeast, Northwest, Northeast \n"))

# create dataframe
df = pd.DataFrame({"age":age, "sex":sex, "bmi":bmi, "child":child, "smoke":smoke, "region":region}, index = [0] )


# split
x_train, x_test, y_train, y_test = train_test_split(df.values[:, :-1], df.values[:, -1], test_size = 0.2, train_size = 0.2, random_state = 0)


ct = ColumnTransformer( [('ordinal', OrdinalEncoder(handle_unknown= 'use_encoded_value', unknown_value = -1) , [1, 4, 5] ), 
                        ('scaler', StandardScaler() , [0, 2] )], 
                            remainder = 'passthrough')


x_train = ct.fit_transform(x_train)
x_test = ct.transform(x_test)


# predict data    
for train_model in model:
    predictions = train_model.predict(x_test)
    


print("Prediction : {}".format(predictions.mean()))