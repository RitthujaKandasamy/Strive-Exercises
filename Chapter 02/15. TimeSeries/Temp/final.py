from tree_model import tree_classifiers, x_val
from timeseries2 import  y
from new_features import new_x
import pandas as pd



# select best modle(accuracy = 99.6)
best_model = tree_classifiers["LightGBM"]



# Fit and predict best model with all data, x_test
best_model.fit(new_x, y)
test_pred = best_model.predict(x_val)
#print(test_pred)



# create dataframe for temp
temp = pd.DataFrame(test_pred, columns = ["T (degC)"])
print(temp)






# results
"""
Model       MSE       MAB  R2 score        Time
0       LightGBM  0.228641  0.330479  0.996832    0.973694
1        XGBoost  0.237994  0.329885  0.996702    8.832883
2        Skl GBM  0.267986  0.369951  0.996287   62.643995
3  Random Forest  0.269823  0.356308  0.996261  196.858693
4  Decision Tree  0.525782  0.468577  0.992715    3.338269
5       AdaBoost  0.664730  0.622850  0.990790   22.944460
        T (degC)
0      17.531728
1      20.627831
2      -7.293226
3       4.999601
4      -1.230736
...          ...
12011   5.760372
12012   6.098016
12013  12.109029
12014   1.317273
12015  21.714198

[12016 rows x 1 columns]
"""