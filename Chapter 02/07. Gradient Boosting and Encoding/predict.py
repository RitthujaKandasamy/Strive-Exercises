from test import best_model
import numpy as np
import pandas as pd



model = best_model


def model_full():

    # create new data input 
    age = int(input("How old are you? \n"))
    sex = str(input("What is your sex? \n"))
    child = int(input("How many children do you have? \n"))
    smoke = str(input("Do you smoke? \n"))
    bmi = float(input("What is your bmi? \n"))
    region = str(input("Choose one of the regions from this: southwest, southeast, northwest, northeast \n"))



    # create dataframe
    df = pd.DataFrame({"age":age, "sex":sex, "bmi":bmi, "child":child, "smoke":smoke, "region":region}, index = [0] )
    

    
    # in model 0 is Gradientdescentregression
    # model 1 is columntransform
    gb_reg_new = model[0]
    ct_new = model[1] 
    

    # transform new data
    x_trans = ct_new.transform(df.values)

    
    # predict data    
    y_pred = np.array(gb_reg_new.predict(x_trans))


    print("\n Your predicted Charges : {}".format(round(y_pred.mean(), 3)))
    

model_full()




