import train as tr
import numpy as np
import pandas as pd



model = tr.train_test()


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

    
    # in model 3 is Gradientbooster
    # model -1 is columntransform
    gb_reg_new = model[3]
    ct_new = model[-1] 
    

    # transform new data
    x_trans = ct_new.transform(df)

    
    # predict data    
    y_pred = np.array(gb_reg_new.predict(x_trans))
        


    print("Charges : {}".format(y_pred.mean()))