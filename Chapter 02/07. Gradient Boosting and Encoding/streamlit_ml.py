import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.model_selection import cross_validate, GridSearchCV




st.title("Webscraping  of Top 100 Adventure Movies in IMDB")
st.image("Downloads\\lord.jpg", use_column_width = True)



data = pd.read_csv("C:\\Users\\ritth\\code\\Strive\\Strive-Exercises\\Chapter 02\\07. Gradient Boosting and Encoding\\insurance.csv")

x_train, x_test, y_train, y_test = train_test_split(data.values[:, :-1], data.values[:, -1], test_size = 0.2, random_state = 0)


    # preprocessing data
    # OrdinalEncoding and Standardscaler
ct = ColumnTransformer( [('ordinal', OrdinalEncoder(handle_unknown= 'use_encoded_value', unknown_value = -1) , [1, 4, 5] ), 
                             ('scaler', StandardScaler() , [0, 2] )], 
                                remainder = 'passthrough')

    
x_train = ct.fit_transform(x_train)
x_test = ct.transform(x_test)



def train_test():
    

    rf_reg = RandomForestRegressor()
    ada_reg = AdaBoostRegressor()
    gb_reg = GradientBoostingRegressor()
    xgb_reg = XGBRegressor()


    # create RF, Ada, GB, XGB in one line
    models = [rf_reg, ada_reg, gb_reg, xgb_reg]

    
    for train_model in models:
        fits = train_model.fit(x_train, y_train)
        predictions = train_model.predict(x_test)
        score = train_model.score(x_test, y_test)
        

        # check crossvalidation for better result
        cv = cross_validate(train_model, x_train, y_train)
        

        # dt = train_model.get_params().keys()
        parameters = {
                    'n_estimators':[50, 100, 100, 50],
                        'random_state': [0, 0, 0, 0]
                }
                
        grd = GridSearchCV(train_model, parameters)
        grid_train = grd.fit(x_train, y_train)

        accuracy = grd.best_score_
        


        #print('{} : \n Predication = {}, \n Score = {}, \n Crossvalidation = {}, \n Gridaccuracy = {} \n'.format(self.fits, self.predictions[:3], self.score, self.cv, self.accuracy))
        #print('Mean train cross validation score {} \n'.format(self.cv['test_score'].mean()))

        return rf_reg, ada_reg, gb_reg, xgb_reg, fits, predictions, score, cv, accuracy
        
train_test()

    


def model_full():

    # create new data input 
    age = st.number_input("How old are you?")
    sex = st.radio("What is your sex? ", ('male', 'female'))
    child = st.number_input("How many children do you have?")
    smoke = st.radio("Do you smoke?", ('yes', 'no'))
    bmi = st.number_input("What is your bmi?")
    region = st.radio("Choose one of the regions from this:", 
               ('southwest', 'southeast', 'northwest', 'northeast'))



    # create dataframe
    df = pd.DataFrame({"age":age, "sex":sex, "bmi":bmi, "child":child, "smoke":smoke, "region":region}, index = [0] )
    
    
    gb_reg = GradientBoostingRegressor()
    fits_new = gb_reg.fit(x_train, y_train)

    
    # transform new data
    x_trans = ct.transform(df.values)

    
    # predict data    
    y_pred = np.array(gb_reg.predict(x_trans))


    st.write("\n Your predicted Charges : {}".format(round(y_pred.mean(), 3)))
    st.write("Thank visit our app again")
    
   

model_full()


st.subheader("Give some heart for us")
my_range = range(1, 6)
number = st.select_slider("Choose a number", options = my_range, value = 1)
st.write("You given us %s hearts:" %number, number*":heart:")


if number == 5:
    st.balloons()



