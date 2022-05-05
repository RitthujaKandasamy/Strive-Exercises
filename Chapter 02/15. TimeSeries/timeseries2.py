import pandas as pd
import numpy as np
import time

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OrdinalEncoder, PowerTransformer
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor



# Load data
data = pd.read_csv('C:\\Users\\ritth\\code\\Strive\\Strive-Exercises\\Chapter 02\\15. TimeSeries\\climate.csv')
# print(data.shape)

data = data.drop("Date Time", axis = 1)
#print(data.head())


# Function to extract sequences and target variable from given data
def pairing(data, seq_len, target_name):

    feature_target = []
    target = []
    for i in range(0, data.shape[0] - (seq_len + 1), seq_len + 1):

         feature_target.append(data[i: seq_len + i])
         target.append(data[target_name][seq_len + i])

    return np.array(feature_target), np.array(target)

x, y = pairing(data, 6, "T (degC)")

print(x.shape)
print(y.shape)
#print(x[:3])



# Extract features
def getfeatures(data):

    new_data = []

    # get each group by sequences
    for i in range(data.shape[0]):

        seq_feature_new = []   
        
        # get each column within each group
        for j in range(data[4: ]):

            seq_feature_new.append((np.max(data[i][:, j]) - np.min(data[i][:, j])))  
            seq_feature_new.append(np.std(data[i][:, j]))

            

        new_data.append(seq_feature_new)

    return np.array(new_data)


new_x = getfeatures(x)
print(new_x.shape)



# # split data
# x_train, x_val, y_train, y_val = train_test_split(new_x, y, test_size=0.2, random_state=0 )

# rang = abs(y_train.max()) + abs(y_train.min())

# # create tree
# tree_classifiers = {
#                         "Decision Tree": DecisionTreeRegressor(random_state=0),
#                         "Extra Trees":   ExtraTreesRegressor(random_state=0),
#                         "Random Forest": RandomForestRegressor(random_state=0),
#                         "AdaBoost":      AdaBoostRegressor(random_state=0),
#                         "Skl GBM":       GradientBoostingRegressor(random_state=0),
#                         "XGBoost":       XGBRegressor(),
#                         "LightGBM":      LGBMRegressor(random_state=0),
#                         "CatBoost":      CatBoostRegressor(random_state=0),
#                                }





# tree_classifiers = {name: make_pipeline(model) for name, model in tree_classifiers.items()}

# results = pd.DataFrame({'Model': [], 'MSE': [], 'MAB': [], " % error": [], 'Time': []})

# for model_name, model in tree_classifiers.items():
    
#     start_time = time.time()


#     # fit new train values
#     model.fit(x_train, y_train)
#     total_time = time.time() - start_time


#     # predict split values    
#     pred = model.predict(x_val)
    

#     # append the list
#     # to find r square, we are using squared error
#     results = results.append({"Model":    model_name,
#                               "MSE": mean_squared_error(y_val, pred),
#                               "MAB": mean_absolute_error(y_val, pred),
#                               " % error": mean_squared_error(y_val, pred) / rang,
#                               "Time":     total_time},
#                               ignore_index = True)




# results_ord = results.sort_values(by = ['MSE'], ascending = True, ignore_index = True)

# print(results_ord)
