import pandas as pd
import time
from kaggle_pipeline import tree_classifiers
from kaggle_dataset import X, y
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split


# split data
x_train, x_val, y_train, y_val = train_test_split(X, y, test_size = 0.2, random_state = 0, stratify = y)

# print(x_train.shape, x_val.shape, y_train.shape, y_val.shape)
# print(type(x_train))


# create new list and dataframe to store values
results = pd.DataFrame({'Model': [], 'Accuracy': [], 'Bal Acc.': [], 'Time': []})



for model_name, model in tree_classifiers.items():
    start_time = time.time()
    
    
    # predict and fit split data
    model.fit(x_train, y_train)
    total_time = time.time() - start_time
    pred =  model.predict(x_val)


    # store values
    results = results.append({
                              "Model":    model_name,
                              "Accuracy": accuracy_score(y_val, pred)*100,
                              "Bal Acc.": balanced_accuracy_score(y_val, pred)*100,
                              "Time":     total_time},
                              ignore_index = True)


results_order = results.sort_values(by = ['Accuracy'], ascending = False, ignore_index = True)


print(results_order)










# with f_27 column
"""
           Model   Accuracy   Bal Acc.         Time
0        Skl GBM  70.797778  71.084717  1027.086682
1       AdaBoost  63.119444  63.402561   242.145665
2  Decision Tree  61.122222  61.131221    64.494024
"""

# without f_27 column
"""
      Model   Accuracy   Bal Acc.       Time
0  LightGBM  82.001111  81.966445  12.972682
"""

# with f_27 column
"""
     Model   Accuracy   Bal Acc.       Time
0  LightGBM  79.444444  79.583503  17.692861
"""

# drop some columns with f_27 column
"""
      Model   Accuracy  Bal Acc.       Time
0  LightGBM  80.947778   80.9109  11.685616
"""

# drop some columns without f_27 column
"""
    Model   Accuracy   Bal Acc.       Time
0  LightGBM  78.348333  78.466054  18.662264
"""

