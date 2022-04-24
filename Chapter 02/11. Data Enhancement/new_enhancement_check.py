import numpy    as np
from numpy.testing._private.utils import decorate_methods
import pandas   as pd
import seaborn  as sns
import matplotlib.pyplot as plt
import sklearn  as skl
import time

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OrdinalEncoder, PowerTransformer
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier




# Data attributes

"""

"timestamp" - timestamp field for grouping the data
"cnt" - the count of a new bike shares
"t1" - real temperature in C
"t2" - temperature in C "feels like"
"hum" - humidity in percentage
"windspeed" - wind speed in km/h
"weathercode" - category of the weather
"isholiday" - boolean field - 1 holiday / 0 non holiday
"isweekend" - boolean field - 1 if the day is weekend
"season" - category field meteorological seasons: 0-spring ; 1-summer; 2-fall; 3-winter.

"weathe_code" category description:
1 = Clear ; mostly clear but have some values with haze/fog/patches of fog/ fog in vicinity 
2 = scattered clouds / few clouds 
3 = Broken clouds 
4 = Cloudy 
7 = Rain/ light Rain shower/ Light rain 
10 = rain with thunderstorm 
26 = snowfall 
94 = Freezing Fog

"""



# read data
data = pd.read_csv(r'data\london_merged.csv')
print("Data Shape: {}".format(data.shape))


# Take a look at nulls 0 nulls
print(data.isnull().sum())


# lets create a 2 new feautures
# Hour time stamp contains the year and the month,
# we will create different columns for each one

data['year'] = data['timestamp'].apply(lambda row: row[:4])
data['month'] = data['timestamp'].apply(lambda row: row.split('-')[1] )
data['hour'] = data['timestamp'].apply(lambda row: row.split(':')[0][-2:] )

data.drop('timestamp', axis=1, inplace=True)

print("After adding New Feature: {}\n".format(data.shape))



# convert data values or data generation 
def data_enhancement(data):
    
    gen_data = data.copy()
    
    for season in data['season'].unique():
        seasonal_data =  gen_data[gen_data['season'] == season]
        hum_std = seasonal_data['hum'].mean()
        wind_speed_std = seasonal_data['wind_speed'].mean()
        t1_std = seasonal_data['t1'].mean()
        t2_std = seasonal_data['t2'].mean()
        

       # add and subtract the values by mean
        for i in seasonal_data.index:
            if np.random.randint(2) == 1:
                gen_data['hum'].values[i] += hum_std/10
                gen_data['wind_speed'].values[i] += wind_speed_std/10
                gen_data['t1'].values[i] += t1_std/10
                gen_data['t2'].values[i] += t2_std/10
            else:
                gen_data['hum'].values[i] -= hum_std/10
                gen_data['wind_speed'].values[i] -= wind_speed_std/10
                gen_data['t1'].values[i] -= t1_std/10
                gen_data['t2'].values[i] -= t2_std/10


    return gen_data



gen = data_enhancement(data)
print("Dataframe After data Enhancement: {}\n".format(gen.shape))
print(gen.head(3))



# final_data = data
y = data['cnt']
x = data.drop(['cnt'], axis=1)

print("Feature shape: {}\n".format(x.shape))
print("Feature shape: {}\n".format(y.shape))




# categorical and numerical data
cat_vars = ['season','is_weekend','is_holiday','year','month','weather_code']
num_vars = ['t1','t2','hum','wind_speed']



# split data
x_train1, x_val1, y_train1, y_val1 = train_test_split(x, y, test_size=0.2, random_state=0 ) # Recommended for reproducibility
                                


# get 25% data from generated data
extra_sample = gen.sample(gen.shape[0] // 4)



# data and generated data both merged by concatenate in pandas
# extra rows are added in data
x_train = pd.concat([x_train1, extra_sample.drop(['cnt'], axis=1 ) ])
y_train = pd.concat([y_train1, extra_sample['cnt'] ])

print("X_train shape after added New Features and before: {} and {}\n".format(x_train, x_train1))
print("y_train shape after added New Features and before: {} and {}\n".format(y_train, y_train1))






# fit and transform data
transformer = PowerTransformer()
y_train = transformer.fit_transform(y_train1.values.reshape(-1,1))
y_val = transformer.transform(y_val1.values.reshape(-1,1))



rang = abs(y_train.max()) + abs(y_train.min())




# create pipeline
num_4_treeModels = Pipeline(steps=[
                              ('imputer', SimpleImputer(strategy='constant', fill_value=-9999)) ])

cat_4_treeModels = Pipeline(steps=[
                              ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                              ('ordinal', OrdinalEncoder()) ]) # handle_unknown='ignore' ONLY IN VERSION 0.24


tree_prepro = ColumnTransformer(transformers=[
                              ('num', num_4_treeModels, num_vars),
                              ('cat', cat_4_treeModels, cat_vars),
                              ], remainder='drop') # Drop other vars not specified in num_vars or cat_vars




# create tree
tree_classifiers = {
                        "Decision Tree": DecisionTreeRegressor(),
                        "Extra Trees":   ExtraTreesRegressor(n_estimators=100),
                        "Random Forest": RandomForestRegressor(n_estimators=100),
                        "AdaBoost":      AdaBoostRegressor(n_estimators=100),
                        "Skl GBM":       GradientBoostingRegressor(n_estimators=100),
                        "XGBoost":       XGBRegressor(n_estimators=100),
                        "LightGBM":      LGBMRegressor(n_estimators=100),
                        "CatBoost":      CatBoostRegressor(n_estimators=100),
                               }




### END SOLUTION

tree_classifiers = {name: make_pipeline(tree_prepro, model) for name, model in tree_classifiers.items()}

results = pd.DataFrame({'Model': [], 'MSE': [], 'MAB': [], " % error": [], 'Time': []})

for model_name, model in tree_classifiers.items():
    
    start_time = time.time()
    model.fit(x_train, y_train)
    total_time = time.time() - start_time
        
    pred = model.predict(x_val1)
    
    results = results.append({"Model":    model_name,
                              "MSE": mean_squared_error(y_val, pred),
                              "MAB": mean_absolute_error(y_val, pred),
                              " % error": mean_squared_error(y_val, pred) / rang,
                              "Time":     total_time},
                              ignore_index = True)


### END SOLUTION



results_ord = results.sort_values(by = ['MSE'], ascending = True, ignore_index = True)
results_ord.index += 1 
results_ord.style.bar(subset=['MSE', 'MAE'], vmin =0, vmax =100, color ='#5fba7d')

print(results_ord)


print(y_train.max())
print(y_train.min())
print(y_val[3])
print(tree_classifiers['Random Forest'].predict(x_val1)[3])
