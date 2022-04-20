import train as tr
import pandas as pd
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor



model = tr.train(RandomForestRegressor(), AdaBoostRegressor(), GradientBoostingRegressor(), XGBRegressor())
new_model = GradientBoostingRegressor(random_state = 0, n_estimators=100)


while True: 


    # create new data input 
    age = int(input("How old are you? \n"))
    sex = str(input("What is your sex? \n"))
    child = int(input("How many children do you have? \n"))
    smoke = bool(input("Do you smoke? \n"))
    bmi = float(input("What is your bmi? \n"))
    region = str(input("Choose one of the regions from this: Southwest, Southeast, Northwest, Northeast \n"))



    # create dataframe
    df = pd.DataFrame({"age":age, "sex":sex, "bmi":bmi, "child":child, "smoke":smoke, "region":region}, index = [0] )



    # predict data    
    for new_train_model in new_model:
        predictions = new_train_model.predict(df)
        


    print("Prediction : {}".format(predictions))