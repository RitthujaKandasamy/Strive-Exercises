from new_features import new_x
from timeseries2 import data, y
import pandas as pd
import joblib
import time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.svm import SVR




# split data
x_train, x_val, y_train, y_val = train_test_split(new_x, y, test_size=0.2, random_state=0 )


# create tree
tree_classifiers = {
                        "Decision Tree": DecisionTreeRegressor(random_state=0),
                        "Random Forest": RandomForestRegressor(random_state=0),
                        "AdaBoost":      AdaBoostRegressor(random_state=0),
                        "Skl GBM":       GradientBoostingRegressor(random_state=0),
                        "XGBoost":       XGBRegressor(),
                        "SVM":           SVR(),
                        "LightGBM":      LGBMRegressor(random_state=0),
                               }



tree_classifier = {name: Pipeline([('scalar', StandardScaler()), ('regressor', model)]) 
                       for name, model in tree_classifiers.items()}

results = pd.DataFrame({'Model': [], 'MSE': [], 'MAB': [], "R2 score": [], 'Time': []})

for name, model in tree_classifiers.items():
    
    start_time = time.time()
    model.fit(x_train, y_train)
    total_time = time.time() - start_time
   
    pred = model.predict(x_val)
    
 
    results = results.append({"Model": name,
                              "MSE": mean_squared_error(y_val, pred),
                              "MAB": mean_absolute_error(y_val, pred),
                              "R2 score": r2_score(y_val, pred),
                              "Time":     total_time},
                              ignore_index = True)




results_ord = results.sort_values(by = ['MSE'], ascending = True, ignore_index = True)

print(results_ord)




# final model
best_model = tree_classifiers.get("LightGBM")

best_model.fit(x_train, y_train)

preds = best_model.predict(x_val)
    
    

# Saving model
joblib.dump(best_model, 'best_model.joblib')






"""
Model       MSE       MAB  R2 score        Time
0       CatBoost  0.222266  0.329613  0.996920   29.226871
1       LightGBM  0.228641  0.330479  0.996832    0.942842
2        XGBoost  0.237994  0.329885  0.996702    8.391728
3        Skl GBM  0.267986  0.369951  0.996287   60.328682
4  Random Forest  0.269823  0.356308  0.996261  180.506934
5    Extra Trees  0.273439  0.363178  0.996211   56.635883
6  Decision Tree  0.525782  0.468577  0.992715    2.983281
7       AdaBoost  0.664730  0.622850  0.990790   21.912367
"""