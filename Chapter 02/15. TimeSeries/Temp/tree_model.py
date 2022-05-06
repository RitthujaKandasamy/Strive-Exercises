from new_features import new_x
from timeseries2 import y
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
                        #"SVM":           SVR(),
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
# best_model = tree_classifiers.get("LightGBM")

# best_model.fit(x_train, y_train)

# preds = best_model.predict(x_val)
    
    

# # Saving model
# joblib.dump(best_model, 'best_model.joblib')




# taking only mean for columns
"""
          Model       MSE       MAB  R2 score       Time
0        XGBoost  0.348288  0.410158  0.995174   5.079666
1       LightGBM  0.352487  0.410403  0.995116   0.390744
2        Skl GBM  0.362038  0.422004  0.994984  24.069801
3  Random Forest  0.370739  0.425475  0.994863  63.722446
4  Decision Tree  0.680752  0.584457  0.990568   1.031579
5       AdaBoost  0.772113  0.672699  0.989302   8.377547

"""

# taking only max-min
"""
          Model        MSE       MAB  R2 score       Time
0        XGBoost   1.178387  0.826855  0.983673   2.922765
1       LightGBM   1.240715  0.833498  0.982809   0.359487
2  Random Forest   1.995252  0.979837  0.972355  40.028166
3        Skl GBM   2.707908  1.187757  0.962481  12.425643
4  Decision Tree   4.549264  1.524329  0.936968   0.625191
5       AdaBoost  10.241070  2.467303  0.858105   5.782983
"""

# both mean and max-min
"""
Model       MSE       MAB  R2 score        Time
0        XGBoost  0.237334  0.331597  0.996712    6.126859
1       LightGBM  0.239219  0.338038  0.996685    0.597890
2        Skl GBM  0.275969  0.376183  0.996176   37.511373
3  Random Forest  0.293422  0.369127  0.995934  119.567511
4  Decision Tree  0.576950  0.491833  0.992006    1.953713
5       AdaBoost  0.713447  0.659259  0.990115   13.738549
"""

# with all features
"""
Model       MSE       MAB  R2 score        Time
0       LightGBM  0.228641  0.330479  0.996832    1.766160
1        XGBoost  0.237994  0.329885  0.996702   10.716631
2        Skl GBM  0.267986  0.369951  0.996287   67.032687
3  Random Forest  0.269823  0.356308  0.996261  194.643708
4  Decision Tree  0.525782  0.468577  0.992715    3.087529
5       AdaBoost  0.664730  0.622850  0.990790   22.256749
"""