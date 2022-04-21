from dataset import x, y
import pandas as pd
import time
from pipeline import tree_classifiers
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, balanced_accuracy_score



# split data
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size = 0.2, random_state = 0, stratify = y)

print(x_train.shape, x_val.shape, y_train.shape, y_val.shape)



# create new list and dataframe to store values
results = pd.DataFrame({'Model': [], 'Accuracy': [], 'Bal Acc.': [], 'Time': []})



# loop through the tree to get both keys and values
for model_name, model in tree_classifiers.items():
    start_time = time.time()
    
    
    # predict and fit split data
    model.fit(x_train, y_train)
    pred =  model.predict(x_val)


    # total time calculate how much time we send to get value
    total_time = time.time() - start_time
    

    # store values
    results = results.append({
                              "Model":    model_name,
                              "Accuracy": accuracy_score(y_val, pred)*100,
                              "Bal Acc.": balanced_accuracy_score(y_val, pred)*100, # balanced is used to know the inbalanced data stored
                              "Time":     total_time},
                              ignore_index = True)




#print(results)




# same process for StrarifiedKFold and Cross validation
skf = StratifiedKFold(n_splits = 10, shuffle = True, random_state = 0)



# create new list and dataframe to store values
results_cross = pd.DataFrame({'Model': [], 'Accuracy': [], 'Bal Acc.': [], 'Time': []})



for model_name, model in tree_classifiers.items():
    start_time = time.time()
        

    # TRAIN AND GET PREDICTIONS USING cross_val_predict() and x,y
    pred = cross_val_predict(model, x, y, cv = skf)


    total_time = time.time() - start_time

    results_cross = results_cross.append({
                                            "Model":    model_name,
                                            "Accuracy": accuracy_score(y, pred)*100,
                                            "Bal Acc.": balanced_accuracy_score(y, pred)*100,
                                            "Time":     total_time},
                                            ignore_index = True)



#print(results_cross)